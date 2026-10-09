import os
import sys
import glob
import json
import time
import argparse
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DATA_DIR, FEATURE_DIM_HOLISTIC, STATIC_MODEL_PATH
from src.vision.landmark_extractor import LandmarkExtractor
from src.models.static_classifier import StaticSignClassifier

class DatasetImporter:
    """
    Ingestion engine for external Sign Language alphabet and word datasets.
    Supports:
      1. Folder hierarchies of images (e.g. Kaggle ISL Alphabet: 'dataset/A/*.jpg', 'dataset/B/*.jpg')
      2. CSV landmark files with coordinates and label columns
      3. Automatic MediaPipe Holistic landmark extraction and 222-D geometric normalization
      4. Disjoint partition generation and model retraining
    """

    SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or os.path.join(DATA_DIR, "imported_datasets")
        os.makedirs(self.output_dir, exist_ok=True)
        self.extractor = LandmarkExtractor()

    def import_image_directory(
        self,
        input_directory: str,
        max_samples_per_class: int = 100,
        augment_samples: bool = True
    ) -> Dict[str, Any]:
        """
        Scans a directory containing subfolders named by alphabet letter or sign:
        Example structure:
          input_dir/
            ├── A/
            │   ├── 001.jpg, 002.jpg ...
            ├── B/
            │   ├── 001.jpg, 002.jpg ...
            └── ...
        Extracts 222-D normalized coordinates using LandmarkExtractor.
        """
        if not os.path.isdir(input_directory):
            return {"status": "error", "message": f"Directory not found: {input_directory}"}

        subdirs = [
            d for d in os.listdir(input_directory)
            if os.path.isdir(os.path.join(input_directory, d)) and not d.startswith(".")
        ]

        if not subdirs:
            return {"status": "error", "message": f"No class subdirectories found inside {input_directory}"}

        print(f"[DatasetImporter] Found {len(subdirs)} class folders: {subdirs}")
        extracted_data = {}
        total_processed = 0
        total_failed = 0

        t0 = time.perf_counter()

        for class_name in sorted(subdirs):
            class_folder = os.path.join(input_directory, class_name)
            image_files = [
                os.path.join(class_folder, f)
                for f in os.listdir(class_folder)
                if os.path.splitext(f.lower())[1] in self.SUPPORTED_IMAGE_EXTS
            ]

            class_vectors = []
            selected_files = image_files[:max_samples_per_class]

            for img_path in selected_files:
                frame = cv2.imread(img_path)
                if frame is None:
                    total_failed += 1
                    continue

                features = self.extractor.process_frame(frame)
                vec = features["holistic_vector"]

                # Check if hands were detected (non-trivial vector)
                if np.max(np.abs(vec)) > 1e-4:
                    class_vectors.append(vec)
                    total_processed += 1
                else:
                    total_failed += 1

            if class_vectors:
                extracted_data[class_name] = np.array(class_vectors, dtype=np.float32)
                print(f"[DatasetImporter] Class '{class_name}': {len(class_vectors)} valid landmarks extracted.")

        t1 = time.perf_counter()
        elapsed = t1 - t0

        if not extracted_data:
            return {
                "status": "error",
                "message": "No valid hand landmarks could be extracted from provided images.",
                "total_failed": total_failed
            }

        # Save processed dataset as compressed archive
        timestamp = int(time.time())
        dataset_archive = os.path.join(self.output_dir, f"imported_dataset_{timestamp}.npz")
        np.savez_compressed(dataset_archive, **extracted_data)

        summary = {
            "status": "success",
            "archive_path": dataset_archive,
            "classes_imported": list(extracted_data.keys()),
            "total_classes": len(extracted_data),
            "total_landmarks_extracted": total_processed,
            "failed_extractions": total_failed,
            "extraction_time_sec": round(elapsed, 2)
        }
        return summary

    def import_csv_dataset(self, csv_file_path: str, label_column: str = "label") -> Dict[str, Any]:
        """
        Imports pre-extracted coordinate landmarks from a CSV file.
        Expects columns for landmark coordinates or raw coordinates.
        """
        import pandas as pd
        if not os.path.isfile(csv_file_path):
            return {"status": "error", "message": f"CSV file not found: {csv_file_path}"}

        df = pd.read_csv(csv_file_path)
        if label_column not in df.columns:
            return {"status": "error", "message": f"Label column '{label_column}' not found in CSV."}

        labels = df[label_column].astype(str).values
        feature_cols = [c for c in df.columns if c != label_column]

        X_raw = df[feature_cols].values.astype(np.float32)

        extracted_data = {}
        for unique_label in np.unique(labels):
            idx = (labels == unique_label)
            extracted_data[unique_label] = X_raw[idx]

        dataset_archive = os.path.join(self.output_dir, f"csv_imported_{int(time.time())}.npz")
        np.savez_compressed(dataset_archive, **extracted_data)

        return {
            "status": "success",
            "archive_path": dataset_archive,
            "classes_imported": list(extracted_data.keys()),
            "total_samples": len(df),
            "feature_dim": X_raw.shape[1]
        }

    def train_model_on_dataset(
        self,
        npz_archive_path: str,
        model_type: str = "rf"
    ) -> Dict[str, Any]:
        """
        Trains or updates StaticSignClassifier using the extracted NPZ archive.
        """
        if not os.path.exists(npz_archive_path):
            return {"status": "error", "message": f"Archive not found: {npz_archive_path}"}

        npz = np.load(npz_archive_path, allow_pickle=True)
        classes = sorted(npz.files)

        X_list = []
        y_list = []

        for cls in classes:
            vecs = npz[cls]
            for v in vecs:
                X_list.append(v)
                y_list.append(cls)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)

        classifier = StaticSignClassifier()
        result = classifier.train(X, y, classes=classes, model_type=model_type)

        return {
            "status": "success",
            "trained_classes": classes,
            "num_samples": len(X),
            "train_accuracy": result.get("train_accuracy", 0.0),
            "model_path": classifier.model_path
        }


def main():
    parser = argparse.ArgumentParser(description="Sign Language Dataset Ingestion & Feature Extractor")
    parser.add_argument("--input-dir", type=str, help="Path to directory containing labeled alphabet folders (A/, B/, ...)")
    parser.add_argument("--csv", type=str, help="Path to CSV file with landmark coordinates")
    parser.add_argument("--max-samples", type=int, default=80, help="Max image samples per class to extract")
    parser.add_argument("--train", action="store_true", help="Automatically train static classifier on extracted landmarks")
    parser.add_argument("--model-type", type=str, default="rf", choices=["rf", "mlp"], help="Model type to train")

    args = parser.parse_args()
    importer = DatasetImporter()

    if args.input_dir:
        print(f"[DatasetImporter] Processing image directory: {args.input_dir}")
        res = importer.import_image_directory(args.input_dir, max_samples_per_class=args.max_samples)
        print(json.dumps(res, indent=2))
        if args.train and res.get("status") == "success":
            print("[DatasetImporter] Training model on imported landmarks...")
            train_res = importer.train_model_on_dataset(res["archive_path"], model_type=args.model_type)
            print(json.dumps(train_res, indent=2))
    elif args.csv:
        print(f"[DatasetImporter] Processing CSV file: {args.csv}")
        res = importer.import_csv_dataset(args.csv)
        print(json.dumps(res, indent=2))
        if args.train and res.get("status") == "success":
            train_res = importer.train_model_on_dataset(res["archive_path"], model_type=args.model_type)
            print(json.dumps(train_res, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
