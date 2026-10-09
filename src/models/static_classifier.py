import os
import joblib
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from src.config import STATIC_MODEL_PATH, CONFIDENCE_THRESHOLD, UNCERTAINTY_MARGIN, UNKNOWN_LABEL

class StaticSignClassifier:
    """
    Classifier for instantaneous static ISL signs and manual postures.
    Uses calibrated ensemble random forest or MLP with uncertainty rejection.
    """

    ALPHABET_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    STATIC_WORD_CLASSES = [
        "NAMASTE", "PLEASE", "HELP", "YES", "NO",
        "HOUSE", "BOOK", "WATER", "FOOD", "PEACE",
        "ME", "YOU", "TEACHER", "DOCTOR", "STUDENT", "FRIEND"
    ]
    DEFAULT_CLASSES = ALPHABET_CLASSES + STATIC_WORD_CLASSES

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = str(model_path or STATIC_MODEL_PATH)
        self.classes: List[str] = self.DEFAULT_CLASSES
        self.model = None
        self.is_trained = False

        if os.path.exists(self.model_path):
            self.load()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: Optional[List[str]] = None,
        model_type: str = "rf"
    ) -> Dict[str, Any]:
        """
        Trains static sign model on landmark feature vectors.
        Args:
            X: shape (N, feature_dim)
            y: shape (N,) integer class indices or string labels
            classes: list of class names
            model_type: 'rf' for RandomForest or 'mlp' for Multi-Layer Perceptron
        """
        if classes is not None:
            self.classes = list(classes)

        # Standardize labels to class indices
        if isinstance(y[0], str):
            label_to_idx = {name: i for i, name in enumerate(self.classes)}
            y_indices = np.array([label_to_idx.get(name, 0) for name in y], dtype=np.int32)
        else:
            y_indices = np.array(y, dtype=np.int32)

        if model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=300,
                activation="relu",
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=80,
                max_depth=16,
                min_samples_split=3,
                random_state=42,
                class_weight="balanced"
            )

        self.model.fit(X, y_indices)
        self.is_trained = True
        self.save()

        # Compute training accuracy
        train_acc = float(np.mean(self.model.predict(X) == y_indices))
        return {
            "num_samples": len(X),
            "num_classes": len(self.classes),
            "train_accuracy": train_acc,
            "classes": self.classes
        }

    def predict_with_confidence(
        self,
        feature_vector: np.ndarray,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        uncertainty_margin: float = UNCERTAINTY_MARGIN
    ) -> Tuple[str, float, bool, Dict[str, float]]:
        """
        Classifies landmark vector with confidence scoring and unknown rejection.
        Returns:
            Tuple of:
                - predicted_label (str)
                - confidence_score (float 0.0 - 1.0)
                - is_uncertain (bool)
                - all_probabilities (dict of class_name -> probability)
        """
        if not self.is_trained or self.model is None:
            return UNKNOWN_LABEL, 0.0, True, {}

        # Vector validation
        vec = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

        # Reject all-zero features (no hands/pose detected)
        if np.all(vec == 0) or np.max(np.abs(vec)) < 1e-4:
            return UNKNOWN_LABEL, 0.0, True, {cls: 0.0 for cls in self.classes}

        probs = self.model.predict_proba(vec)[0]
        prob_dict = {
            self.classes[i]: float(probs[i])
            for i in range(len(self.classes))
            if i < len(probs)
        }

        # Sort probabilities descending
        sorted_indices = np.argsort(probs)[::-1]
        top_1_idx = sorted_indices[0]
        top_1_prob = float(probs[top_1_idx])
        top_2_prob = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0

        top_label = self.classes[top_1_idx] if top_1_idx < len(self.classes) else UNKNOWN_LABEL

        # Uncertainty rules:
        # 1. Top probability below threshold
        # 2. Ambiguity: margin between top-1 and top-2 is smaller than uncertainty_margin
        is_uncertain = False
        if top_1_prob < confidence_threshold:
            is_uncertain = True
            top_label = UNKNOWN_LABEL
        elif (top_1_prob - top_2_prob) < uncertainty_margin:
            is_uncertain = True
            # Flag ambiguity while maintaining informative label
            top_label = f"UNCERTAIN ({top_label}?)"

        return top_label, top_1_prob, is_uncertain, prob_dict

    def save(self, path: Optional[str] = None):
        """Saves trained model and class metadata."""
        target_path = str(path or self.model_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "classes": self.classes,
            "is_trained": self.is_trained
        }, target_path)

    def load(self, path: Optional[str] = None):
        """Loads trained model and class metadata."""
        target_path = str(path or self.model_path)
        if os.path.exists(target_path):
            data = joblib.load(target_path)
            self.model = data.get("model")
            self.classes = data.get("classes", self.DEFAULT_CLASSES)
            self.is_trained = data.get("is_trained", True)
