import os
import json
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from src.config import DATA_DIR, FEATURE_DIM_HOLISTIC, CONFIDENCE_THRESHOLD, UNCERTAINTY_MARGIN
from src.evaluation.metrics import EvaluationMetrics

class SignerIndependentEvaluator:
    """
    Evaluates ISL gesture models using disjoint, signer-independent data splits
    (Leave-One-Signer-Out / Cross-Signer validation) to prevent artificially inflated
    test metrics caused by intra-signer data leakage.
    """

    REPORT_PATH = DATA_DIR / "signer_independent_report.json"

    def __init__(self, classes: Optional[List[str]] = None):
        self.classes = classes or [
            "A", "B", "C", "D", "E", "L", "O", "V", "Y",
            "NAMASTE", "PLEASE", "HELP", "YES", "NO", "HOUSE", "WATER"
        ]

    def generate_multi_signer_dataset(
        self,
        num_signers: int = 4,
        samples_per_signer_class: int = 20
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generates structured synthetic benchmark data partitioned strictly by signer_id.
        Each signer has distinct anatomical hand scale, joint ratios, and signing speed/noise.
        Signers: ['signer_01', 'signer_02', 'signer_03', 'signer_04']
        """
        np.random.seed(42)
        dataset = {}

        # Base prototypes for each class
        base_prototypes = {}
        for c_idx, c_name in enumerate(self.classes):
            np.random.seed(200 + c_idx)
            proto = np.random.uniform(-0.8, 0.8, FEATURE_DIM_HOLISTIC).astype(np.float32)
            proto[:126] = proto[:126] / (np.linalg.norm(proto[:126]) + 1e-4)
            base_prototypes[c_name] = proto

        for s_idx in range(num_signers):
            signer_id = f"signer_{s_idx + 1:02d}"
            # Unique signer physical bias: hand size variation (scale) and postural habit
            np.random.seed(300 + s_idx)
            signer_scale_bias = float(np.random.uniform(0.88, 1.15))
            signer_posture_bias = np.random.normal(0, 0.04, FEATURE_DIM_HOLISTIC).astype(np.float32)

            X_signer = []
            y_signer = []

            for c_name in self.classes:
                base = base_prototypes[c_name]
                for _ in range(samples_per_signer_class):
                    # Sample noise representing natural execution variance
                    noise = np.random.normal(0, 0.035, FEATURE_DIM_HOLISTIC).astype(np.float32)
                    sample = (base * signer_scale_bias) + signer_posture_bias + noise
                    sample[:126] = sample[:126] / (np.linalg.norm(sample[:126]) + 1e-4)
                    X_signer.append(sample)
                    y_signer.append(c_name)

            # Generate out-of-vocabulary unknown gestures for this signer
            X_unknown = []
            for _ in range(15):
                rand_u = np.random.uniform(-2.0, 2.0, FEATURE_DIM_HOLISTIC).astype(np.float32)
                X_unknown.append(rand_u)

            dataset[signer_id] = {
                "X": np.array(X_signer, dtype=np.float32),
                "y": y_signer,
                "X_unknown": np.array(X_unknown, dtype=np.float32),
                "scale_bias": signer_scale_bias
            }

        return dataset

    def run_disjoint_signer_evaluation(
        self,
        test_signer_id: str = "signer_04"
    ) -> Dict[str, Any]:
        """
        Executes Signer-Independent benchmark where training is performed strictly
        on signers {S1, S2, S3} and testing is evaluated exclusively on unseen {S4}.
        Also computes comparison with Signer-Dependent (random mixed split) to measure
        the empirical generalization drop.
        """
        print(f"[SignerIndependentEvaluator] Partitioning dataset. Held-out test signer: {test_signer_id}")
        data = self.generate_multi_signer_dataset(num_signers=4, samples_per_signer_class=25)

        # 1. Signer-Independent Partition
        train_signers = [s for s in data if s != test_signer_id]
        X_train_independent = np.vstack([data[s]["X"] for s in train_signers])
        y_train_independent = [lbl for s in train_signers for lbl in data[s]["y"]]

        X_test_independent = data[test_signer_id]["X"]
        y_test_independent = data[test_signer_id]["y"]
        X_test_unknown = data[test_signer_id]["X_unknown"]

        # Train model on independent signers
        clf_independent = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42)
        clf_independent.fit(X_train_independent, y_train_independent)

        # Evaluate on unseen test signer
        preds_independent = clf_independent.predict(X_test_independent)
        metrics_independent = EvaluationMetrics.compute_classification_metrics(
            y_true=y_test_independent,
            y_pred=list(preds_independent),
            classes=self.classes
        )

        # Evaluate Unknown Rejection on test signer
        unknown_preds = []
        for u_vec in X_test_unknown:
            probs = clf_independent.predict_proba([u_vec])[0]
            max_p = np.max(probs)
            pred_class = clf_independent.classes_[np.argmax(probs)]
            label = "UNKNOWN" if max_p < CONFIDENCE_THRESHOLD else pred_class
            unknown_preds.append(label)

        rejection_metrics = EvaluationMetrics.compute_unknown_rejection_metrics(
            known_true=y_test_independent,
            known_pred=list(preds_independent),
            unknown_true=["UNKNOWN"] * len(X_test_unknown),
            unknown_pred=unknown_preds
        )

        # 2. Signer-Dependent Baseline (Random 80/20 mixed split pooling all signers together)
        all_X = np.vstack([data[s]["X"] for s in data])
        all_y = [lbl for s in data for lbl in data[s]["y"]]
        indices = np.random.permutation(len(all_X))
        split_point = int(0.75 * len(all_X))
        train_idx, test_idx = indices[:split_point], indices[split_point:]

        clf_dependent = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42)
        clf_dependent.fit(all_X[train_idx], [all_y[i] for i in train_idx])
        preds_dependent = clf_dependent.predict(all_X[test_idx])
        acc_dependent = float(np.mean(preds_dependent == [all_y[i] for i in test_idx]))

        generalization_gap = round(acc_dependent - metrics_independent["accuracy"], 4)

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol": "Signer-Independent (Disjoint Signers)",
            "train_signers": train_signers,
            "held_out_test_signer": test_signer_id,
            "classes_evaluated": len(self.classes),
            "total_independent_train_samples": len(X_train_independent),
            "total_independent_test_samples": len(X_test_independent),
            "signer_independent_metrics": metrics_independent,
            "unknown_rejection_metrics": rejection_metrics,
            "signer_dependent_accuracy_baseline": round(acc_dependent, 4),
            "generalization_gap": generalization_gap,
            "finding_summary": (
                f"Signer-independent accuracy is {metrics_independent['accuracy'] * 100:.1f}%, "
                f"compared to {acc_dependent * 100:.1f}% when signers are randomly mixed across splits. "
                f"The measured generalization gap of {generalization_gap * 100:.1f}% substantiates the data collection rule "
                f"that keeping all clips from one signer in only one split is mandatory to prevent inflated evaluation."
            )
        }

        with open(self.REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report
