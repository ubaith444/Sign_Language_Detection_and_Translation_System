import os
import cv2
import json
import time
import numpy as np
from pathlib import Path
from collections import deque
from typing import Dict, Any, List, Optional, Tuple

from src.config import (
    SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD, UNCERTAINTY_MARGIN,
    UNKNOWN_LABEL, DATA_DIR
)
from src.vision.landmark_extractor import LandmarkExtractor
from src.translation.sentence_builder import SentenceBuilder

class VideoBatchProcessor:
    """
    Offline batch processor for pre-recorded ISL video files.
    Performs frame-by-frame landmark extraction, continuous segmentation,
    syntactic English translation, and exports aligned subtitles (.srt, .vtt, .json)
    and subtitle-burned video files.
    """

    def __init__(
        self,
        static_model: Optional[Any] = None,
        dynamic_model: Optional[Any] = None,
        translator: Optional[Any] = None
    ):
        from src.models.static_classifier import StaticSignClassifier
        from src.models.dynamic_classifier import DynamicSignClassifier
        from src.models.continuous_segmenter import ContinuousSignSegmenter
        from src.translation.seq2seq_translator import ISLTranslator

        self.extractor = LandmarkExtractor()
        self.static_model = static_model or StaticSignClassifier()
        self.dynamic_model = dynamic_model or DynamicSignClassifier()
        self.translator = translator or ISLTranslator()
        self.segmenter = ContinuousSignSegmenter()

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        burn_subtitles: bool = True,
        mode: str = "AUTO"  # "AUTO", "STATIC", or "DYNAMIC"
    ) -> Dict[str, Any]:
        """
        Processes a video file and generates subtitle tracks and transcripts.
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_name = Path(video_path).stem
        target_dir = Path(output_dir or (DATA_DIR / "processed_videos" / video_name))
        os.makedirs(target_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video stream: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if total_frames > 0 else 0.0

        # Video Writer for subtitle burning
        burned_video_path = target_dir / f"{video_name}_subtitled.mp4"
        writer = None
        if burn_subtitles:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(burned_video_path), fourcc, fps, (width, height))

        # Runtime structures
        sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.segmenter.reset()

        frame_idx = 0
        aligned_segments = []
        sentence_builder = SentenceBuilder(translator=self.translator)

        current_segment_start_sec = 0.0
        current_active_sign = None
        current_active_conf = 0.0

        t_start_processing = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_time_sec = frame_idx / fps

            # 1. Landmark extraction
            features = self.extractor.process_frame(frame)
            vector_222 = features["holistic_vector"]
            sequence_buffer.append(vector_222)

            # 2. Inference based on mode
            if mode == "DYNAMIC" or (mode == "AUTO" and len(sequence_buffer) == SEQUENCE_LENGTH):
                # Check dynamic model
                label_dyn, conf_dyn, is_unc_dyn, _ = self.dynamic_model.predict_sequence(
                    np.array(sequence_buffer),
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    uncertainty_margin=UNCERTAINTY_MARGIN
                )
                # Static check
                label_stat, conf_stat, is_unc_stat, _ = self.static_model.predict_with_confidence(
                    vector_222,
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    uncertainty_margin=UNCERTAINTY_MARGIN
                )

                if conf_dyn > conf_stat and not is_unc_dyn:
                    pred_label, pred_conf, is_unc = label_dyn, conf_dyn, is_unc_dyn
                else:
                    pred_label, pred_conf, is_unc = label_stat, conf_stat, is_unc_stat
            else: # Static mode
                pred_label, pred_conf, is_unc, _ = self.static_model.predict_with_confidence(
                    vector_222,
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    uncertainty_margin=UNCERTAINTY_MARGIN
                )

            # 3. Continuous segmentation
            committed_sign, seg_info = self.segmenter.update(
                vector_222, pred_label, pred_conf, is_unc
            )

            if committed_sign:
                # Close previous sign segment if open
                if current_active_sign is not None:
                    aligned_segments.append({
                        "sign": current_active_sign,
                        "confidence": round(float(current_active_conf), 3),
                        "start_time_sec": round(float(current_segment_start_sec), 2),
                        "end_time_sec": round(float(current_time_sec), 2)
                    })

                current_active_sign = committed_sign
                current_active_conf = pred_conf
                current_segment_start_sec = max(0.0, current_time_sec - 0.4)
                sentence_builder.add_word(committed_sign, pred_conf)

            # 4. Burn subtitles onto frame if requested
            if burn_subtitles and writer is not None:
                annotated = self.extractor.draw_landmarks(frame, features["raw_results"])
                state = sentence_builder.get_state()
                rendered_frame = self._render_subtitle_hud(
                    frame=annotated,
                    current_sign=pred_label,
                    confidence=pred_conf,
                    is_uncertain=is_unc,
                    sentence_state=state,
                    time_sec=current_time_sec
                )
                writer.write(rendered_frame)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        # Close final segment
        if current_active_sign is not None:
            aligned_segments.append({
                "sign": current_active_sign,
                "confidence": round(float(current_active_conf), 3),
                "start_time_sec": round(float(current_segment_start_sec), 2),
                "end_time_sec": round(float(duration_sec), 2)
            })

        processing_duration = time.perf_counter() - t_start_processing

        # Final sentence state
        final_state = sentence_builder.get_state()

        # Build sentence-level subtitle cues
        sentence_cues = self._generate_sentence_cues(aligned_segments, final_state, duration_sec)

        # Export Subtitles (.srt, .vtt) and Transcript (.json)
        srt_path = target_dir / f"{video_name}.srt"
        vtt_path = target_dir / f"{video_name}.vtt"
        json_path = target_dir / f"{video_name}_transcript.json"

        self._export_srt(sentence_cues, srt_path)
        self._export_vtt(sentence_cues, vtt_path)

        result_summary = {
            "video_name": video_name,
            "video_path": video_path,
            "fps": round(float(fps), 2),
            "total_frames": frame_idx,
            "duration_sec": round(float(duration_sec), 2),
            "processing_time_sec": round(float(processing_duration), 2),
            "processing_fps": round(frame_idx / max(processing_duration, 0.001), 1),
            "recognized_gloss_tokens": final_state.get("glosses", final_state.get("tokens", [])),
            "translated_english": final_state.get("translated_text", ""),
            "translation_method": final_state.get("translation_method", "none"),
            "aligned_sign_segments": aligned_segments,
            "sentence_subtitle_cues": sentence_cues,
            "exported_files": {
                "srt_file": str(srt_path),
                "vtt_file": str(vtt_path),
                "json_transcript": str(json_path),
                "burned_video": str(burned_video_path) if burn_subtitles else None
            }
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_summary, f, indent=2)

        return result_summary

    def _generate_sentence_cues(
        self,
        aligned_segments: List[Dict[str, Any]],
        final_state: Dict[str, Any],
        duration_sec: float
    ) -> List[Dict[str, Any]]:
        """Constructs timestamped subtitle cues for full translated sentences."""
        if not aligned_segments:
            return [{
                "cue_index": 1,
                "start_time_sec": 0.5,
                "end_time_sec": max(1.0, duration_sec),
                "start_timestamp": self._format_timestamp(0.5, is_srt=True),
                "end_timestamp": self._format_timestamp(max(1.0, duration_sec), is_srt=True),
                "text": final_state.get("translated_text", "") or "No signs detected."
            }]

        start_sec = aligned_segments[0]["start_time_sec"]
        end_sec = max(aligned_segments[-1]["end_time_sec"], start_sec + 1.5)
        text = final_state.get("translated_text", "")

        return [{
            "cue_index": 1,
            "start_time_sec": start_sec,
            "end_time_sec": end_sec,
            "start_timestamp": self._format_timestamp(start_sec, is_srt=True),
            "end_timestamp": self._format_timestamp(end_sec, is_srt=True),
            "gloss_sequence": " ".join([seg["sign"] for seg in aligned_segments]),
            "text": text
        }]

    @staticmethod
    def _format_timestamp(seconds: float, is_srt: bool = True) -> str:
        """Converts float seconds to 00:00:00,000 (SRT) or 00:00:00.000 (VTT)."""
        millis = int(round((seconds - int(seconds)) * 1000))
        total_seconds = int(seconds)
        s = total_seconds % 60
        m = (total_seconds // 60) % 60
        h = total_seconds // 3600
        sep = "," if is_srt else "."
        return f"{h:02d}:{m:02d}:{s:02d}{sep}{millis:03d}"

    def _export_srt(self, cues: List[Dict[str, Any]], output_path: Path):
        """Writes standard SubRip (.srt) subtitle file."""
        lines = []
        for i, cue in enumerate(cues, 1):
            start_ts = self._format_timestamp(cue["start_time_sec"], is_srt=True)
            end_ts = self._format_timestamp(cue["end_time_sec"], is_srt=True)
            lines.append(str(i))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(cue.get("text", ""))
            lines.append("")  # Blank separator

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _export_vtt(self, cues: List[Dict[str, Any]], output_path: Path):
        """Writes WebVTT (.vtt) subtitle file for HTML5 video players."""
        lines = ["WEBVTT", ""]
        for i, cue in enumerate(cues, 1):
            start_ts = self._format_timestamp(cue["start_time_sec"], is_srt=False)
            end_ts = self._format_timestamp(cue["end_time_sec"], is_srt=False)
            lines.append(f"{i}")
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(cue.get("text", ""))
            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _render_subtitle_hud(
        self,
        frame: np.ndarray,
        current_sign: str,
        confidence: float,
        is_uncertain: bool,
        sentence_state: Dict[str, Any],
        time_sec: float
    ) -> np.ndarray:
        """Renders subtitle bar and HUD directly onto output video frames."""
        h, w, _ = frame.shape
        out = frame.copy()

        # Top Information Banner
        cv2.rectangle(out, (0, 0), (w, 55), (250, 250, 250), -1)
        cv2.line(out, (0, 55), (w, 55), (200, 205, 215), 1)

        time_str = f"Time: {time_sec:.1f}s"
        cv2.putText(out, time_str, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 116, 139), 1)

        if not is_uncertain and current_sign != UNKNOWN_LABEL:
            cv2.putText(out, f"Sign: {current_sign} ({int(confidence * 100)}%)", (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 150, 40), 2)
        else:
            cv2.putText(out, f"Sign: {current_sign}", (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 100, 30), 1)

        # Bottom Subtitle Banner
        cv2.rectangle(out, (0, h - 75), (w, h), (15, 23, 42), -1)
        sub_text = sentence_state.get("translated_text", "") or sentence_state.get("raw_text", "")
        if sub_text:
            cv2.putText(out, sub_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            cv2.putText(out, "(Signing in progress...)", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1)

        return out

    @staticmethod
    def generate_synthetic_benchmark_video(
        output_path: str,
        num_frames: int = 150,
        fps: float = 30.0
    ) -> str:
        """
        Creates an animated test video for automated validation when a physical
        pre-recorded camera file is not immediately available.
        Simulates: [NAMASTE] -> [PLEASE] -> [HELP] -> [ME]
        """
        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        w, h = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        signs = ["NAMASTE", "PLEASE", "HELP", "ME"]
        frames_per_sign = num_frames // len(signs)

        for i in range(num_frames):
            sign_idx = min(len(signs) - 1, i // frames_per_sign)
            sign_name = signs[sign_idx]

            frame = np.full((h, w, 3), 245, dtype=np.uint8)
            cx, head_y = w // 2, 140
            cv2.circle(frame, (cx, head_y), 45, (210, 210, 215), -1)
            cv2.ellipse(frame, (cx, 300), (90, 110), 0, 0, 180, (185, 185, 195), -1)

            # Oscillate hands
            t = i * 0.1
            hand_y = int(240 + np.sin(t) * 20)
            cv2.circle(frame, (cx - 70, hand_y), 24, (230, 180, 160), -1)
            cv2.circle(frame, (cx + 70, hand_y), 24, (230, 180, 160), -1)

            cv2.putText(frame, f"Benchmark Dataset Clip: {sign_name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
            writer.write(frame)

        writer.release()
        return output_path
