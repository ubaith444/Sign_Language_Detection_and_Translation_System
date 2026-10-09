import os
import sys
import time
from collections import deque
import cv2
import numpy as np

# Add project root to Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS,
    CONFIDENCE_THRESHOLD, UNCERTAINTY_MARGIN, UNKNOWN_LABEL,
    SEQUENCE_LENGTH, FEATURE_DIM_HOLISTIC
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

def run_desktop_app():
    print("=================================================================")
    print(" INDIAN SIGN LANGUAGE (ISL) DETECTION AND TRANSLATION SYSTEM     ")
    print("=================================================================")
    print("Controls:")
    print("  [A] : Add detected sign to sentence builder")
    print("  [U] : Undo last added word")
    print("  [C] : Clear current sentence")
    print("  [S] : Speak sentence aloud via Text-To-Speech")
    print("  [P] : Toggle privacy blur mode")
    print("  [M] : Toggle mode (Static Signs <-> Dynamic Sequences)")
    print("  [Q] : Quit application")
    print("=================================================================\n")

    # Initialize Components
    camera = CameraStream(width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=TARGET_FPS)
    extractor = LandmarkExtractor()
    static_model = StaticSignClassifier()
    dynamic_model = DynamicSignClassifier()
    segmenter = ContinuousSignSegmenter()
    sentence_builder = SentenceBuilder()
    synthesizer = SpeechSynthesizer()
    profiler = LatencyProfiler()

    # If models are untrained, bootstrap with benchmark generator
    if not static_model.is_trained or not dynamic_model.is_trained:
        print("[Desktop App] Bootstrapping initial ISL model weights...")
        benchmark = ISLSystemBenchmark()
        benchmark.run_full_benchmark()
        static_model.load()
        dynamic_model.load()

    camera.start()

    # Temporal Sequence Buffer
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    mode = "STATIC"  # "STATIC" or "DYNAMIC"
    privacy_blur = False

    current_label = UNKNOWN_LABEL
    current_confidence = 0.0
    current_uncertain = True

    try:
        while True:
            frame_start = time.perf_counter()
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # 1. Landmark Extraction
            ext_t0 = time.perf_counter()
            features = extractor.process_frame(frame)
            ext_t1 = time.perf_counter()
            extraction_ms = (ext_t1 - ext_t0) * 1000.0

            vector_222 = features["holistic_vector"]
            sequence_buffer.append(vector_222)

            # 2. Classification Inference
            inf_t0 = time.perf_counter()
            if mode == "STATIC":
                label, conf, is_unc, _ = static_model.predict_with_confidence(
                    vector_222,
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    uncertainty_margin=UNCERTAINTY_MARGIN
                )
            else: # DYNAMIC
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    label, conf, is_unc, _ = dynamic_model.predict_sequence(
                        np.array(sequence_buffer),
                        confidence_threshold=CONFIDENCE_THRESHOLD,
                        uncertainty_margin=UNCERTAINTY_MARGIN
                    )
                else:
                    label = f"BUFFERING ({len(sequence_buffer)}/{SEQUENCE_LENGTH})"
                    conf = 0.0
                    is_unc = True

            inf_t1 = time.perf_counter()
            inference_ms = (inf_t1 - inf_t0) * 1000.0

            current_label = label
            current_confidence = conf
            current_uncertain = is_unc

            # 3. Continuous Sign Segmentation
            committed, seg_info = segmenter.update(
                vector_222, current_label, current_confidence, current_uncertain
            )
            if committed:
                print(f"[Auto-Segmenter Committed]: {committed} (conf: {current_confidence:.2f})")
                sentence_builder.add_word(committed, current_confidence)

            # 4. Telemetry Profiling
            telemetry = profiler.record_frame(extraction_ms, inference_ms)

            # 5. Visual HUD Rendering
            annotated_frame = extractor.draw_landmarks(frame, features["raw_results"])
            hud = render_hud(
                frame=annotated_frame,
                mode=mode,
                predicted_label=current_label,
                confidence=current_confidence,
                is_uncertain=current_uncertain,
                sentence_state=sentence_builder.get_state(),
                telemetry=telemetry,
                privacy_blur=privacy_blur
            )

            cv2.imshow("ISL Detection and Translation System", hud)

            # 6. Key Event Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('a'):  # Add word
                if current_label and current_label != UNKNOWN_LABEL and not current_label.startswith("UNCERTAIN"):
                    sentence_builder.add_word(current_label, current_confidence)
                    print(f"[User Action] Added: {current_label}")
            elif key == ord('u'):  # Undo
                sentence_builder.undo()
                print("[User Action] Undid last word")
            elif key == ord('c'):  # Clear
                sentence_builder.clear()
                print("[User Action] Cleared sentence")
            elif key == ord('s'):  # Speak
                st = sentence_builder.get_state()
                to_speak = st.get("translated_text") or st.get("raw_text")
                if to_speak:
                    print(f"[User Action] Speaking: {to_speak}")
                    synthesizer.speak(to_speak)
            elif key == ord('p'):  # Privacy Blur
                privacy_blur = not privacy_blur
                camera.set_privacy_blur(privacy_blur)
                print(f"[User Action] Privacy Blur: {privacy_blur}")
            elif key == ord('m'):  # Toggle Mode
                mode = "DYNAMIC" if mode == "STATIC" else "STATIC"
                sequence_buffer.clear()
                print(f"[User Action] Mode switched to: {mode}")

    finally:
        camera.stop()
        extractor.close()
        synthesizer.stop()
        cv2.destroyAllWindows()
        print("[Desktop App] Gracefully shut down.")

def render_hud(
    frame: np.ndarray,
    mode: str,
    predicted_label: str,
    confidence: float,
    is_uncertain: bool,
    sentence_state: dict,
    telemetry: dict,
    privacy_blur: bool
) -> np.ndarray:
    """Renders professional on-screen display (HUD) banner with clean styling."""
    h, w, _ = frame.shape
    hud = frame.copy()

    # Top Banner (Header & Prediction Status)
    cv2.rectangle(hud, (0, 0), (w, 85), (250, 250, 250), -1)
    cv2.line(hud, (0, 85), (w, 85), (210, 215, 225), 2)

    # Title & Mode
    cv2.putText(hud, f"ISL SYSTEM [{mode}]", (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 41, 59), 2)
    fps_text = f"FPS: {telemetry.get('effective_fps', 0)} | Latency: {telemetry.get('pipeline_latency_ms', 0)}ms"
    cv2.putText(hud, fps_text, (w - 260, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 116, 139), 1)

    # Prediction Badge
    if is_uncertain or predicted_label.startswith("UNCERTAIN") or predicted_label == UNKNOWN_LABEL:
        pred_color = (60, 130, 240)  # Orange/amber
        badge_text = f"Sign: {predicted_label} ({int(confidence * 100)}%)"
    else:
        pred_color = (20, 150, 40)   # Clean Green
        badge_text = f"Sign: {predicted_label} ({int(confidence * 100)}%)"

    cv2.putText(hud, badge_text, (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.85, pred_color, 2)

    # Privacy Indicator
    if privacy_blur:
        cv2.putText(hud, "[PRIVACY BLUR ON]", (w - 220, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 80, 20), 2)

    # Bottom Banner (Sentence Accumulator & English Translation)
    cv2.rectangle(hud, (0, h - 90), (w, h), (250, 250, 250), -1)
    cv2.line(hud, (0, h - 90), (w, h - 90), (210, 215, 225), 2)

    raw_text = sentence_state.get("raw_text", "")
    translated_text = sentence_state.get("translated_text", "")

    # ISL Gloss Sequence
    gloss_display = f"ISL Signs: {raw_text if raw_text else '(none)'}"
    cv2.putText(hud, gloss_display, (15, h - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (71, 85, 105), 1)

    # Grammatical English Translation
    en_display = f"English Translation: {translated_text if translated_text else '(waiting for signs...)'}"
    cv2.putText(hud, en_display, (15, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 23, 42), 2)

    return hud

if __name__ == "__main__":
    run_desktop_app()
