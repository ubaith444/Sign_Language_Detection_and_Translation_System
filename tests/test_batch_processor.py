"""
Comprehensive Unit and Integration Tests for ISL Video Batch Processing & Subtitle Exporter.
Runs with standard library unittest (no external test runner dependency required).
"""

import os
import json
import shutil
import unittest
from pathlib import Path

from src.vision.video_batch_processor import VideoBatchProcessor
from src.config import DATA_DIR


class TestVideoBatchProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = VideoBatchProcessor()
        cls.test_output_dir = DATA_DIR / "test_batch_output"
        os.makedirs(cls.test_output_dir, exist_ok=True)
        cls.test_video_path = cls.test_output_dir / "test_clip.mp4"
        # Generate a lightweight 60-frame synthetic clip
        cls.processor.generate_synthetic_benchmark_video(
            str(cls.test_video_path), num_frames=60, fps=30.0
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir, ignore_errors=True)

    def test_timestamp_formatting_srt(self):
        """Test SRT timestamp formatting with comma millisecond separator."""
        ts_zero = self.processor._format_timestamp(0.0, is_srt=True)
        self.assertEqual(ts_zero, "00:00:00,000")

        ts_complex = self.processor._format_timestamp(3723.456, is_srt=True)
        self.assertEqual(ts_complex, "01:02:03,456")

    def test_timestamp_formatting_vtt(self):
        """Test WebVTT timestamp formatting with dot millisecond separator."""
        ts_zero = self.processor._format_timestamp(0.0, is_srt=False)
        self.assertEqual(ts_zero, "00:00:00.000")

        ts_complex = self.processor._format_timestamp(3723.456, is_srt=False)
        self.assertEqual(ts_complex, "01:02:03.456")

    def test_synthetic_video_exists(self):
        """Verify synthetic benchmark generator produces a valid readable video."""
        self.assertTrue(os.path.exists(self.test_video_path))
        self.assertGreater(os.path.getsize(self.test_video_path), 1000)

    def test_end_to_end_batch_processing(self):
        """Verify full offline batch pipeline produces SRT, VTT, JSON, and subtitled MP4."""
        result = self.processor.process_video(
            video_path=str(self.test_video_path),
            output_dir=str(self.test_output_dir),
            burn_subtitles=True,
            mode="AUTO"
        )

        self.assertIn("video_name", result)
        self.assertIn("fps", result)
        self.assertIn("total_frames", result)
        self.assertIn("duration_sec", result)
        self.assertIn("processing_fps", result)
        self.assertIn("exported_files", result)

        files = result["exported_files"]
        self.assertTrue(os.path.exists(files["srt_file"]), "SRT file must exist")
        self.assertTrue(os.path.exists(files["vtt_file"]), "VTT file must exist")
        self.assertTrue(os.path.exists(files["json_transcript"]), "JSON transcript must exist")
        self.assertTrue(os.path.exists(files["burned_video"]), "Subtitled MP4 must exist")

        # Verify SRT formatting
        with open(files["srt_file"], "r", encoding="utf-8") as f:
            srt_content = f.read()
            self.assertIn("-->", srt_content)
            self.assertIn(",", srt_content)

        # Verify VTT formatting
        with open(files["vtt_file"], "r", encoding="utf-8") as f:
            vtt_content = f.read()
            self.assertTrue(vtt_content.startswith("WEBVTT"))
            self.assertIn(".", vtt_content)

        # Verify JSON transcript validity
        with open(files["json_transcript"], "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
            self.assertEqual(transcript_data["video_name"], result["video_name"])
            self.assertIn("aligned_sign_segments", transcript_data)
            self.assertIn("translated_english", transcript_data)


if __name__ == "__main__":
    unittest.main()
