import os
import sys
import json
import time
from collections import deque
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import (
    WEB_HOST, WEB_PORT, DATA_DIR,
    CONFIDENCE_THRESHOLD, UNCERTAINTY_MARGIN, UNKNOWN_LABEL,
    SEQUENCE_LENGTH, BENCHMARK_REPORT_PATH
)
from src.vision.camera_stream import CameraStream
from src.vision.landmark_extractor import LandmarkExtractor
from src.models.static_classifier import StaticSignClassifier
from src.models.dynamic_classifier import DynamicSignClassifier
from src.models.continuous_segmenter import ContinuousSignSegmenter
from src.translation.sentence_builder import SentenceBuilder
from src.audio.speech_synthesizer import SpeechSynthesizer
from src.evaluation.metrics import LatencyProfiler
from src.evaluation.test_benchmark import ISLSystemBenchmark
from src.vision.facial_modality_classifier import FacialModalityClassifier
from src.linguistics.hamnosys_extractor import HamNoSysExtractor
from src.translation.english_to_isl import EnglishToISLTranslator
from src.translation.sign_player_data import SignPlayerData
from src.session.session_manager import SessionManager
from src.models.realtime_finetuner import RealtimeSignFineTuner
from scripts.dataset_importer import DatasetImporter

def create_app():
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))

    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    CORS(app)

    # Core Components
    camera = CameraStream()
    extractor = LandmarkExtractor()
    static_model = StaticSignClassifier()
    dynamic_model = DynamicSignClassifier()
    segmenter = ContinuousSignSegmenter()
    sentence_builder = SentenceBuilder()
    synthesizer = SpeechSynthesizer()
    profiler = LatencyProfiler()
    facial_classifier = FacialModalityClassifier()
    hamnosys_extractor = HamNoSysExtractor()
    english_to_isl = EnglishToISLTranslator()
    session_manager = SessionManager()
    finetuner = RealtimeSignFineTuner(classifier=static_model)
    dataset_importer = DatasetImporter()

    # Shared Runtime State
    runtime_state = {
        "mode": "STATIC",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "uncertainty_margin": UNCERTAINTY_MARGIN,
        "auto_segmenter": True,
        "current_sign": UNKNOWN_LABEL,
        "confidence": 0.0,
        "is_uncertain": True,
        "sequence_buffer": deque(maxlen=SEQUENCE_LENGTH),
        "facial_modality": {
            "modality": "DECLARATIVE_STATEMENT",
            "modality_label": "Declarative Statement",
            "confidence": 0.9,
            "punctuation": ".",
            "cues": ["Neutral brow and head alignment"]
        },
        "phonology": {
            "phonological_summary": "[REST | PALM_OUTWARD | NEUTRAL_SPACE | STATIONARY | RESTING_INACTIVE]",
            "hamnosys_transcription": "hamneutralspace hamhand hampalmout hamnomovement"
        },
        "telemetry": {
            "pipeline_latency_ms": 0,
            "effective_fps": 0
        }
    }

    # Ensure models trained on startup
    if not static_model.is_trained or not dynamic_model.is_trained:
        print("[WebApp] Bootstrapping initial trained models...")
        benchmark = ISLSystemBenchmark()
        benchmark.run_full_benchmark()
        static_model.load()
        dynamic_model.load()

    camera.start()

    @app.route("/")
    def index():
        return render_template("index.html")

    def generate_frames():
        """Generator producing MJPEG frames for browser video element."""
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            t0 = time.perf_counter()
            # 1. Landmark extraction
            features = extractor.process_frame(frame)
            t1 = time.perf_counter()
            ext_ms = (t1 - t0) * 1000.0

            vector_222 = features["holistic_vector"]
            runtime_state["sequence_buffer"].append(vector_222)

            # 2. Inference
            inf_t0 = time.perf_counter()
            if runtime_state["mode"] == "STATIC":
                sign, conf, is_unc, _ = static_model.predict_with_confidence(
                    vector_222,
                    confidence_threshold=runtime_state["confidence_threshold"],
                    uncertainty_margin=runtime_state["uncertainty_margin"]
                )
            else:
                if len(runtime_state["sequence_buffer"]) == SEQUENCE_LENGTH:
                    sign, conf, is_unc, _ = dynamic_model.predict_sequence(
                        np.array(runtime_state["sequence_buffer"]),
                        confidence_threshold=runtime_state["confidence_threshold"],
                        uncertainty_margin=runtime_state["uncertainty_margin"]
                    )
                else:
                    sign = f"BUFFERING ({len(runtime_state['sequence_buffer'])}/{SEQUENCE_LENGTH})"
                    conf = 0.0
                    is_unc = True

            inf_t1 = time.perf_counter()
            inf_ms = (inf_t1 - inf_t0) * 1000.0

            runtime_state["current_sign"] = sign
            runtime_state["confidence"] = conf
            runtime_state["is_uncertain"] = is_unc

            try:
                # 3. Continuous Segmentation Auto-commit
                if runtime_state["auto_segmenter"]:
                    committed, _ = segmenter.update(vector_222, sign, conf, is_unc)
                    if committed:
                        sentence_builder.add_word(committed, conf)

                # 4. Non-manual Facial Expression Classifier
                if "face" in features and len(features["face"]) > 0:
                    face_mod = facial_classifier.process_face_landmarks(features["face"], sign)
                    runtime_state["facial_modality"] = face_mod
                    sentence_builder.set_modality(face_mod["modality"], face_mod["punctuation"])

                # 5. Phonological Sub-Unit Feature Extraction (HamNoSys)
                phon_res = hamnosys_extractor.decompose_frame(features, list(runtime_state["sequence_buffer"]))
                runtime_state["phonology"] = phon_res

                # 6. Latency profiling
                runtime_state["telemetry"] = profiler.record_frame(ext_ms, inf_ms)

                # 7. Visual Landmark Drawing
                annotated = extractor.draw_landmarks(frame, features.get("raw_results"))
            except Exception:
                annotated = frame

            # Encode as JPEG
            success, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                continue

            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    @app.route("/video_feed")
    def video_feed():
        return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/state", methods=["GET"])
    def get_state():
        return jsonify({
            "current_sign": runtime_state["current_sign"],
            "confidence": round(runtime_state["confidence"], 3),
            "is_uncertain": runtime_state["is_uncertain"],
            "telemetry": runtime_state["telemetry"],
            "sentence_state": sentence_builder.get_state(),
            "facial_modality": runtime_state.get("facial_modality", {}),
            "phonology": runtime_state.get("phonology", {})
        })

    @app.route("/api/sentence/add", methods=["POST"])
    def sentence_add():
        data = request.get_json() or {}
        sign = data.get("sign", "")
        conf = float(data.get("confidence", 1.0))
        return jsonify(sentence_builder.add_word(sign, conf))

    @app.route("/api/sentence/undo", methods=["POST"])
    def sentence_undo():
        return jsonify(sentence_builder.undo())

    @app.route("/api/sentence/clear", methods=["POST"])
    def sentence_clear():
        return jsonify(sentence_builder.clear())

    @app.route("/api/sentence/speak", methods=["POST"])
    def sentence_speak():
        data = request.get_json() or {}
        text = data.get("text", "")
        if text:
            synthesizer.speak(text)
            state = sentence_builder.get_state()
            session_manager.log_turn(
                speaker="Signer (ISL)",
                input_content=" ".join(state.get("tokens", [])) or "ISL_INPUT",
                translated_content=text,
                modality=runtime_state.get("facial_modality", {}).get("modality_label", "Declarative Statement"),
                phonology_summary=runtime_state.get("phonology", {}).get("phonological_summary", ""),
                confidence=state.get("average_confidence", 1.0),
                direction="ISL_TO_ENGLISH"
            )
        return jsonify({"status": "vocalized", "text": text})

    @app.route("/api/settings/mode", methods=["POST"])
    def set_mode():
        data = request.get_json() or {}
        mode = data.get("mode", "STATIC")
        runtime_state["mode"] = mode
        runtime_state["sequence_buffer"].clear()
        return jsonify({"mode": runtime_state["mode"]})

    @app.route("/api/settings/threshold", methods=["POST"])
    def set_threshold():
        data = request.get_json() or {}
        th = float(data.get("threshold", CONFIDENCE_THRESHOLD))
        runtime_state["confidence_threshold"] = th
        return jsonify({"threshold": runtime_state["confidence_threshold"]})

    @app.route("/api/settings/privacy", methods=["POST"])
    def set_privacy():
        data = request.get_json() or {}
        enabled = bool(data.get("privacy_blur", False))
        camera.set_privacy_blur(enabled)
        return jsonify({"privacy_blur": enabled})

    @app.route("/api/benchmark", methods=["GET"])
    def get_benchmark():
        if os.path.exists(BENCHMARK_REPORT_PATH):
            with open(BENCHMARK_REPORT_PATH, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        benchmark = ISLSystemBenchmark()
        return jsonify(benchmark.run_full_benchmark())

    @app.route("/api/benchmark/run", methods=["POST"])
    def run_benchmark():
        benchmark = ISLSystemBenchmark()
        report = benchmark.run_full_benchmark()
        static_model.load()
        dynamic_model.load()
        return jsonify(report)

    @app.route("/api/models/retrain", methods=["POST"])
    def retrain_models():
        benchmark = ISLSystemBenchmark()
        report = benchmark.run_full_benchmark()
        static_model.load()
        dynamic_model.load()
        return jsonify({
            "status": "success",
            "static_train_acc": report.get("static_classification", {}).get("accuracy", 0.0)
        })

    @app.route("/api/sentence/space", methods=["POST"])
    def sentence_space():
        return jsonify(sentence_builder.add_space())

    @app.route("/api/sentence/delete", methods=["POST"])
    def sentence_delete():
        return jsonify(sentence_builder.delete_last())

    @app.route("/api/sentence/builder_mode", methods=["POST"])
    def sentence_builder_mode():
        data = request.get_json() or {}
        b_mode = data.get("mode", "GLOSS_TRANSLATION")
        return jsonify(sentence_builder.set_mode(b_mode))

    @app.route("/api/signer_independent", methods=["GET"])
    def get_signer_independent():
        from src.models.signer_independent_eval import SignerIndependentEvaluator
        evaluator = SignerIndependentEvaluator()
        if os.path.exists(evaluator.REPORT_PATH):
            with open(evaluator.REPORT_PATH, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        report = evaluator.run_disjoint_signer_evaluation()
        return jsonify(report)

    @app.route("/api/alphabet", methods=["GET"])
    def get_alphabet():
        alpha_file = DATA_DIR / "isl_alphabet.json"
        if os.path.exists(alpha_file):
            with open(alpha_file, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        return jsonify({})

    @app.route("/api/lexicon", methods=["GET"])
    def get_lexicon():
        lexicon_file = DATA_DIR / "isl_lexicon.json"
        alpha_file = DATA_DIR / "isl_alphabet.json"
        result = {}
        if os.path.exists(lexicon_file):
            with open(lexicon_file, "r", encoding="utf-8") as f:
                result.update(json.load(f))
        if os.path.exists(alpha_file):
            with open(alpha_file, "r", encoding="utf-8") as f:
                result["alphabet_dataset_a"] = json.load(f)
        return jsonify(result)

    @app.route("/api/video/process", methods=["POST"])
    def process_video_endpoint():
        from src.vision.video_batch_processor import VideoBatchProcessor
        from werkzeug.utils import secure_filename
        processor = VideoBatchProcessor()

        # Check if file was uploaded
        if "video_file" in request.files and request.files["video_file"].filename:
            file = request.files["video_file"]
            filename = secure_filename(file.filename)
            upload_dir = DATA_DIR / "uploaded_videos"
            os.makedirs(upload_dir, exist_ok=True)
            saved_path = upload_dir / filename
            file.save(str(saved_path))
            video_input = str(saved_path)
        else:
            # Use sample benchmark clip
            sample_path = DATA_DIR / "sample_benchmark_isl.mp4"
            if not os.path.exists(sample_path):
                processor.generate_synthetic_benchmark_video(str(sample_path), num_frames=120)
            video_input = str(sample_path)

        burn_subs = request.form.get("burn_subtitles", "true").lower() == "true"
        mode = request.form.get("mode", "AUTO")

        try:
            result = processor.process_video(
                video_path=video_input,
                burn_subtitles=burn_subs,
                mode=mode
            )
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/video/download/<format_type>/<video_name>", methods=["GET"])
    def download_subtitle_file(format_type, video_name):
        from flask import send_file
        target_dir = DATA_DIR / "processed_videos" / video_name
        
        file_map = {
            "srt": target_dir / f"{video_name}.srt",
            "vtt": target_dir / f"{video_name}.vtt",
            "json": target_dir / f"{video_name}_transcript.json",
            "mp4": target_dir / f"{video_name}_subtitled.mp4"
        }

        path = file_map.get(format_type.lower())
        if path and os.path.exists(path):
            as_attach = format_type.lower() != "vtt" # Allow in-browser viewing for vtt
            return send_file(str(path), as_attachment=as_attach)
        return jsonify({"error": "File not found"}), 404

    @app.route("/api/two_way/translate_english", methods=["POST"])
    def two_way_translate():
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        result = english_to_isl.translate_to_isl(text)
        if text and result.get("isl_gloss_sequence"):
            session_manager.log_turn(
                speaker="Speaker (Hearing)",
                input_content=text,
                translated_content=result["gloss_string"],
                modality=result.get("modality", "Declarative Statement"),
                confidence=1.0,
                direction="ENGLISH_TO_ISL"
            )
        return jsonify(result)

    @app.route("/api/two_way/sign_animation/<sign_name>", methods=["GET"])
    def get_sign_animation(sign_name):
        return jsonify(SignPlayerData.get_sign_animation(sign_name))

    @app.route("/api/session/log_turn", methods=["POST"])
    def log_session_turn():
        data = request.get_json() or {}
        turn = session_manager.log_turn(
            speaker=data.get("speaker", "Signer (ISL)"),
            input_content=data.get("input_content", ""),
            translated_content=data.get("translated_content", ""),
            modality=data.get("modality", "Declarative Statement"),
            phonology_summary=data.get("phonology_summary", ""),
            confidence=float(data.get("confidence", 1.0)),
            direction=data.get("direction", "ISL_TO_ENGLISH")
        )
        return jsonify(turn)

    @app.route("/api/session/history", methods=["GET"])
    def get_session_history():
        return jsonify(session_manager.get_session_summary())

    @app.route("/api/session/clear", methods=["POST"])
    def clear_session_history():
        session_manager.clear()
        return jsonify({"status": "cleared", "session_id": session_manager.session_id})

    @app.route("/api/session/export/<format_type>", methods=["GET"])
    def export_session_transcript(format_type):
        from flask import send_file
        export_dir = DATA_DIR / "transcripts"
        os.makedirs(export_dir, exist_ok=True)
        sid = session_manager.session_id
        fmt = format_type.lower()
        if fmt == "pdf":
            out_file = export_dir / f"{sid}_transcript.pdf"
            session_manager.export_pdf(str(out_file))
            return send_file(str(out_file), as_attachment=True, download_name=f"{sid}_transcript.pdf")
        elif fmt == "txt":
            out_file = export_dir / f"{sid}_transcript.txt"
            session_manager.export_text(str(out_file))
            return send_file(str(out_file), as_attachment=True, download_name=f"{sid}_transcript.txt")
        elif fmt == "json":
            out_file = export_dir / f"{sid}_transcript.json"
            session_manager.export_json(str(out_file))
            return send_file(str(out_file), as_attachment=True, download_name=f"{sid}_transcript.json")
        return jsonify({"error": "Unsupported export format"}), 400

    @app.route("/api/finetune/status", methods=["GET"])
    def get_finetune_status():
        return jsonify(finetuner.get_calibration_status())

    @app.route("/api/finetune/capture_sample", methods=["POST"])
    def capture_finetune_sample():
        data = request.get_json() or {}
        label = data.get("label", "").strip().upper()
        if not label:
            return jsonify({"error": "Missing sign label"}), 400

        ret, frame = camera.read()
        if not ret or frame is None:
            return jsonify({"error": "Camera frame unavailable"}), 500

        features = extractor.process_frame(frame)
        vector_222 = features["holistic_vector"]
        if np.max(np.abs(vector_222)) < 1e-4:
            return jsonify({"error": "No hand landmarks detected. Position your hand clearly in camera view."}), 400

        sample_count = finetuner.add_user_sample(label, vector_222)
        return jsonify({
            "status": "sample_added",
            "label": label,
            "sample_count": sample_count,
            "total_user_samples": sum(len(v) for v in finetuner.user_samples.values())
        })

    @app.route("/api/finetune/train", methods=["POST"])
    def execute_finetune():
        data = request.get_json() or {}
        model_type = data.get("model_type", "rf")
        if not any(len(v) > 0 for v in finetuner.user_samples.values()):
            return jsonify({"error": "No calibration samples captured yet. Capture at least 1 sample before training."}), 400

        result = finetuner.fine_tune(model_type=model_type)
        return jsonify(result)

    @app.route("/api/finetune/reset", methods=["POST"])
    def reset_finetune():
        finetuner.clear_calibration()
        static_model.load()
        return jsonify({"status": "reset_to_generic_baseline"})

    @app.route("/api/dataset/import_directory", methods=["POST"])
    def import_dataset_directory():
        data = request.get_json() or {}
        input_dir = data.get("input_dir", "").strip()
        auto_train = data.get("auto_train", True)
        if not input_dir or not os.path.isdir(input_dir):
            return jsonify({"error": f"Invalid or non-existent directory path: {input_dir}"}), 400

        result = dataset_importer.import_image_directory(input_dir)
        if result.get("status") == "success" and auto_train:
            train_res = dataset_importer.train_model_on_dataset(result["archive_path"])
            result["training_result"] = train_res
            static_model.load()
        return jsonify(result)

    return app

if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)
