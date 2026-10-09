import os
import sys
import argparse
from pathlib import Path

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DATA_DIR
from src.vision.video_batch_processor import VideoBatchProcessor

def main():
    parser = argparse.ArgumentParser(description="ISL Pre-Recorded Video Batch Processor & Subtitle Exporter")
    parser.add_argument("--video", type=str, default=None, help="Path to input sign language video (.mp4, .avi, .mov)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save exported subtitles and transcript")
    parser.add_argument("--no-burn", action="store_true", help="Disable burning subtitles directly onto video frames")
    parser.add_argument("--mode", type=str, default="AUTO", choices=["AUTO", "STATIC", "DYNAMIC"], help="Inference mode")
    parser.add_argument("--sample", action="store_true", help="Generate and process an automated benchmark sample video")

    args = parser.parse_args()

    processor = VideoBatchProcessor()

    input_video = args.video
    if args.sample or not input_video:
        sample_path = DATA_DIR / "sample_benchmark_isl.mp4"
        print(f"[VideoBatchProcessor] Generating automated test video at: {sample_path}")
        input_video = processor.generate_synthetic_benchmark_video(str(sample_path), num_frames=120)

    print("=================================================================")
    print(" ISL VIDEO BATCH PROCESSOR & SUBTITLE EXPORTER                   ")
    print(f" Processing: {input_video}                                      ")
    print(f" Mode:       {args.mode}                                         ")
    print(f" Burn Video: {not args.no_burn}                                  ")
    print("=================================================================\n")

    result = processor.process_video(
        video_path=input_video,
        output_dir=args.output_dir,
        burn_subtitles=not args.no_burn,
        mode=args.mode
    )

    print("--- 1. VIDEO METRICS ---")
    print(f"Duration:            {result['duration_sec']} seconds ({result['total_frames']} frames @ {result['fps']} FPS)")
    print(f"Processing Time:     {result['processing_time_sec']} seconds ({result['processing_fps']} FPS throughput)")

    print("\n--- 2. LINGUISTIC RECOGNITION & TRANSLATION ---")
    print(f"Recognized Glosses:  {' '.join(result['recognized_gloss_tokens']) if result['recognized_gloss_tokens'] else '(none)'}")
    print(f"English Translation: {result['translated_english']}")
    print(f"Syntactic Method:    {result['translation_method']}")

    print("\n--- 3. TIMESTAMPED ALIGNED SIGN SEGMENTS ---")
    for seg in result["aligned_sign_segments"]:
        print(f" [{seg['start_time_sec']}s - {seg['end_time_sec']}s] Sign: {seg['sign']} (Confidence: {int(seg['confidence']*100)}%)")

    print("\n--- 4. EXPORTED ARTIFACTS ---")
    for k, v in result["exported_files"].items():
        if v:
            print(f" - {k}: {v}")

    print("\n=================================================================")
    print(" Subtitle export and video batch processing complete.")
    print("=================================================================")

if __name__ == "__main__":
    main()
