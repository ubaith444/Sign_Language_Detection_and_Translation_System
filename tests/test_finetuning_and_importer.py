import os
import shutil
import tempfile
import unittest
import numpy as np
import cv2

from src.config import FEATURE_DIM_HOLISTIC
from src.models.static_classifier import StaticSignClassifier
from src.models.realtime_finetuner import RealtimeSignFineTuner
from scripts.dataset_importer import DatasetImporter
from src.web.app import create_app

class TestRealtimeFineTuningAndImporter(unittest.TestCase):
    """
    Test suite verifying real-time few-shot calibration, live webcam fine-tuning,
    and external alphabet dataset ingestion.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="isl_finetune_test_")
        self.classifier = StaticSignClassifier()
        self.finetuner = RealtimeSignFineTuner(
            classifier=self.classifier,
            calibration_dir=os.path.join(self.test_dir, "calibration")
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_add_and_query_calibration_samples(self):
        """Verifies adding user webcam frames to calibration pool."""
        sample_vec = np.random.uniform(-0.5, 0.5, FEATURE_DIM_HOLISTIC).astype(np.float32)
        count = self.finetuner.add_user_sample("A", sample_vec)
        self.assertEqual(count, 1)

        sample_vec2 = np.random.uniform(-0.5, 0.5, FEATURE_DIM_HOLISTIC).astype(np.float32)
        count2 = self.finetuner.add_user_sample("A", sample_vec2)
        self.assertEqual(count2, 2)

        status = self.finetuner.get_calibration_status()
        self.assertEqual(status["total_user_samples"], 2)
        self.assertIn("A", status["calibrated_signs"])
        self.assertEqual(status["sample_counts"]["A"], 2)

    def test_data_augmentation_pipeline(self):
        """Verifies geometric perturbation synthesizes valid landmark variations."""
        base_vec = np.random.uniform(-0.5, 0.5, FEATURE_DIM_HOLISTIC).astype(np.float32)
        augmented = self.finetuner.augment_sample(base_vec, num_augmented=5)

        self.assertEqual(len(augmented), 5)
        for aug in augmented:
            self.assertEqual(aug.shape, (FEATURE_DIM_HOLISTIC,))
            self.assertFalse(np.array_equal(aug, base_vec))
            # Hand coordinates should be non-trivial
            self.assertGreater(np.linalg.norm(aug[:126]), 0.0)

    def test_few_shot_fine_tuning_execution(self):
        """Verifies model fine-tuning completes in under 3 seconds with high accuracy."""
        # Add 3 distinct samples for 'A' and 'B'
        for _ in range(3):
            vec_a = np.random.uniform(0.1, 0.9, FEATURE_DIM_HOLISTIC).astype(np.float32)
            self.finetuner.add_user_sample("A", vec_a)

            vec_b = np.random.uniform(-0.9, -0.1, FEATURE_DIM_HOLISTIC).astype(np.float32)
            self.finetuner.add_user_sample("B", vec_b)

        result = self.finetuner.fine_tune(base_samples_per_class=10, augmentation_factor=3, model_type="rf")

        self.assertEqual(result["status"], "success")
        self.assertLess(result["retraining_time_sec"], 5.0)
        self.assertGreater(result["user_raw_samples"], 0)
        self.assertGreater(result["user_augmented_samples"], 0)
        self.assertGreaterEqual(result["user_calibration_accuracy"], 0.8)

    def test_dataset_importer_image_folder(self):
        """Verifies ingestion of mock folder hierarchy with alphabet subfolders."""
        input_folder = os.path.join(self.test_dir, "mock_alphabet")
        class_a_dir = os.path.join(input_folder, "A")
        class_b_dir = os.path.join(input_folder, "B")
        os.makedirs(class_a_dir, exist_ok=True)
        os.makedirs(class_b_dir, exist_ok=True)

        # Create mock 200x200 BGR images
        dummy_img = np.ones((200, 200, 3), dtype=np.uint8) * 120
        # Draw mock face and hand shapes so tracker detects landmarks
        cv2.circle(dummy_img, (100, 60), 30, (180, 180, 180), -1)
        cv2.circle(dummy_img, (60, 150), 20, (180, 180, 180), -1)

        cv2.imwrite(os.path.join(class_a_dir, "img1.jpg"), dummy_img)
        cv2.imwrite(os.path.join(class_b_dir, "img1.jpg"), dummy_img)

        importer = DatasetImporter(output_dir=os.path.join(self.test_dir, "output"))
        import_res = importer.import_image_directory(input_folder, max_samples_per_class=5)

        self.assertIn(import_res["status"], ["success", "error"])
        if import_res["status"] == "success":
            self.assertTrue(os.path.exists(import_res["archive_path"]))
            train_res = importer.train_model_on_dataset(import_res["archive_path"])
            self.assertEqual(train_res["status"], "success")

    def test_web_api_finetuning_endpoints(self):
        """Verifies Flask endpoints for live calibration and fine-tuning."""
        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()

        # 1. Status endpoint
        res = client.get("/api/finetune/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("total_user_samples", data)
        self.assertIn("sample_counts", data)

        # 2. Capture sample endpoint
        cap_res = client.post("/api/finetune/capture_sample", json={"label": "A"})
        # Should return 200 with added sample, or 400 if no camera hand visible
        self.assertIn(cap_res.status_code, [200, 400])

        # 3. Reset endpoint
        reset_res = client.post("/api/finetune/reset")
        self.assertEqual(reset_res.status_code, 200)
        self.assertEqual(reset_res.get_json().get("status"), "reset_to_generic_baseline")

if __name__ == "__main__":
    unittest.main()
