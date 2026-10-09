import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

class EvaluationMetrics:
    """
    Computes rigorous quantitative performance metrics for ISL recognition
    and translation: Accuracy, Macro F1, Confusion Matrix, WER, and Latency.
    """

    @staticmethod
    def compute_classification_metrics(
        y_true: List[str],
        y_pred: List[str],
        classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculates accuracy, macro precision, macro recall, macro F1,
        and full confusion matrix.
        """
        if not y_true or not y_pred:
            return {}

        unique_labels = classes if classes is not None else sorted(list(set(y_true) | set(y_pred)))
        
        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=unique_labels, average="macro", zero_division=0
        )
        per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
            y_true, y_pred, labels=unique_labels, average=None, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        per_class_breakdown = {}
        for i, lbl in enumerate(unique_labels):
            per_class_breakdown[lbl] = {
                "precision": float(per_class_p[i]),
                "recall": float(per_class_r[i]),
                "f1_score": float(per_class_f1[i]),
                "support": int(per_class_supp[i])
            }

        return {
            "accuracy": round(acc, 4),
            "macro_precision": round(float(prec), 4),
            "macro_recall": round(float(rec), 4),
            "macro_f1": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
            "classes": unique_labels,
            "per_class": per_class_breakdown,
            "total_samples": len(y_true)
        }

    @staticmethod
    def compute_word_error_rate(reference: str, hypothesis: str) -> float:
        """
        Calculates Word Error Rate (WER) using Levenshtein distance on words.
        WER = (Substitutions + Deletions + Insertions) / Total Reference Words
        """
        r_words = reference.lower().strip().split()
        h_words = hypothesis.lower().strip().split()

        if not r_words:
            return 0.0 if not h_words else 1.0

        d = np.zeros((len(r_words) + 1, len(h_words) + 1), dtype=np.int32)
        for i in range(len(r_words) + 1):
            d[i, 0] = i
        for j in range(len(h_words) + 1):
            d[0, j] = j

        for i in range(1, len(r_words) + 1):
            for j in range(1, len(h_words) + 1):
                if r_words[i - 1] == h_words[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    substitution = d[i - 1, j - 1] + 1
                    insertion = d[i, j - 1] + 1
                    deletion = d[i - 1, j] + 1
                    d[i, j] = min(substitution, insertion, deletion)

        distance = d[len(r_words), len(h_words)]
        wer = distance / len(r_words)
        return float(round(wer, 4))

    @staticmethod
    def compute_unknown_rejection_metrics(
        known_true: List[str],
        known_pred: List[str],
        unknown_true: List[str],
        unknown_pred: List[str],
        unknown_token: str = "UNKNOWN"
    ) -> Dict[str, Any]:
        """
        Evaluates the model's rejection capabilities:
        - True Rejection Rate (TRR): % of genuine unknown gestures correctly flagged
        - False Acceptance Rate (FAR): % of unknown gestures falsely labeled as a known sign
        - False Rejection Rate (FRR): % of known gestures mistakenly flagged as unknown
        """
        # True unknown test set
        total_unknown = len(unknown_true)
        correctly_rejected = sum(1 for p in unknown_pred if unknown_token in p)
        falsely_accepted = total_unknown - correctly_rejected

        trr = (correctly_rejected / total_unknown) if total_unknown > 0 else 1.0
        far = (falsely_accepted / total_unknown) if total_unknown > 0 else 0.0

        # Known test set
        total_known = len(known_true)
        falsely_rejected = sum(1 for p in known_pred if unknown_token in p)
        frr = (falsely_rejected / total_known) if total_known > 0 else 0.0

        return {
            "true_rejection_rate": round(trr, 4),
            "false_acceptance_rate": round(far, 4),
            "false_rejection_rate": round(frr, 4),
            "total_unknown_samples": total_unknown,
            "total_known_samples": total_known
        }

class LatencyProfiler:
    """Measures inference latency and throughput (FPS)."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.extraction_times: List[float] = []
        self.inference_times: List[float] = []
        self.total_frame_times: List[float] = []
        self._last_frame_timestamp = time.perf_counter()

    def record_frame(
        self,
        extraction_time_ms: float,
        inference_time_ms: float
    ) -> Dict[str, float]:
        """Records telemetry for one frame and returns running averages."""
        now = time.perf_counter()
        delta = (now - self._last_frame_timestamp) * 1000.0
        self._last_frame_timestamp = now

        self.extraction_times.append(extraction_time_ms)
        self.inference_times.append(inference_time_ms)
        self.total_frame_times.append(delta)

        if len(self.extraction_times) > self.window_size:
            self.extraction_times.pop(0)
            self.inference_times.pop(0)
            self.total_frame_times.pop(0)

        avg_ext = float(np.mean(self.extraction_times))
        avg_inf = float(np.mean(self.inference_times))
        avg_frame = float(np.mean(self.total_frame_times))
        fps = (1000.0 / avg_frame) if avg_frame > 0 else 0.0

        return {
            "extraction_latency_ms": round(avg_ext, 2),
            "inference_latency_ms": round(avg_inf, 2),
            "pipeline_latency_ms": round(avg_ext + avg_inf, 2),
            "effective_fps": round(fps, 1)
        }
