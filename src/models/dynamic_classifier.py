import os
import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from src.config import (
    DYNAMIC_MODEL_PATH,
    DYNAMIC_METADATA_PATH,
    SEQUENCE_LENGTH,
    CONFIDENCE_THRESHOLD,
    UNCERTAINTY_MARGIN,
    UNKNOWN_LABEL,
    FEATURE_DIM_HOLISTIC
)

class DynamicSignClassifier:
    """
    Temporal sequence classifier for dynamic, movement-based ISL signs.
    Processes ordered landmark sequences (e.g. 30 frames x 222 features)
    using a Bidirectional Temporal Recurrent model.
    """

    DEFAULT_DYNAMIC_CLASSES = [
        "HELLO", "THANK_YOU", "GOODBYE", "WHAT", "WHERE",
        "HOW_ARE_YOU", "SORRY", "MEET", "GO", "COME"
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        sequence_length: int = SEQUENCE_LENGTH,
        feature_dim: int = FEATURE_DIM_HOLISTIC
    ):
        self.model_path = str(model_path or DYNAMIC_MODEL_PATH)
        self.metadata_path = str(metadata_path or DYNAMIC_METADATA_PATH)
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.classes: List[str] = self.DEFAULT_DYNAMIC_CLASSES
        self.is_trained = False
        self.torch_model = None
        self.fallback_model = None

        if os.path.exists(self.model_path) and os.path.exists(self.metadata_path):
            self.load()

    def _build_torch_network(self, num_classes: int):
        """Constructs PyTorch Temporal GRU network."""
        import torch
        import torch.nn as nn

        class TemporalGRUClassifier(nn.Module):
            def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Linear(in_features, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
                self.gru = nn.GRU(
                    input_size=128,
                    hidden_size=hidden_dim,
                    num_layers=2,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.2
                )
                self.attention = nn.Sequential(
                    nn.Linear(hidden_dim * 2, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_dim * 2, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, num_classes)
                )

            def forward(self, x):
                # x: (batch, seq_len, in_features)
                proj = self.proj(x)
                out, _ = self.gru(proj)  # (batch, seq_len, hidden_dim * 2)

                # Temporal Attention Pooling
                att_weights = torch.softmax(self.attention(out), dim=1) # (batch, seq_len, 1)
                pooled = torch.sum(out * att_weights, dim=1)           # (batch, hidden_dim * 2)

                logits = self.head(pooled)
                return logits

        return TemporalGRUClassifier(
            in_features=self.feature_dim,
            hidden_dim=64,
            num_classes=num_classes
        )

    def train_torch(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        classes: Optional[List[str]] = None,
        epochs: int = 35,
        batch_size: int = 16,
        lr: float = 0.001
    ) -> Dict[str, Any]:
        """
        Trains temporal GRU on sequence data.
        X_seq: (N, seq_len, feature_dim)
        y: (N,) class labels or integer indices
        """
        if classes is not None:
            self.classes = list(classes)

        # Standardize labels
        if isinstance(y[0], str):
            label_to_idx = {name: i for i, name in enumerate(self.classes)}
            y_indices = np.array([label_to_idx.get(name, 0) for name in y], dtype=np.int64)
        else:
            y_indices = np.array(y, dtype=np.int64)

        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.torch_model = self._build_torch_network(len(self.classes)).to(device)

            dataset = TensorDataset(
                torch.tensor(X_seq, dtype=torch.float32),
                torch.tensor(y_indices, dtype=torch.long)
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(self.torch_model.parameters(), lr=lr, weight_decay=1e-4)

            self.torch_model.train()
            final_loss = 0.0
            for epoch in range(epochs):
                total_loss = 0.0
                for bx, by in loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    out = self.torch_model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                final_loss = total_loss / max(1, len(loader))

            self.torch_model.eval()
            self.is_trained = True
            self.save()

            return {
                "num_sequences": len(X_seq),
                "num_classes": len(self.classes),
                "final_loss": float(final_loss),
                "device": str(device),
                "classes": self.classes
            }

        except Exception as e:
            print(f"[DynamicSignClassifier] Torch training unavailable: {e}. Falling back to scikit-learn sequence model.")
            return self._train_fallback(X_seq, y_indices)

    def _train_fallback(self, X_seq: np.ndarray, y_indices: np.ndarray) -> Dict[str, Any]:
        """Fallback lightweight classifier using temporal statistical summary + Random Forest."""
        from sklearn.ensemble import RandomForestClassifier
        # Extract temporal mean, std, min, max, delta features
        N, T, D = X_seq.shape
        mean_feat = np.mean(X_seq, axis=1)
        std_feat = np.std(X_seq, axis=1)
        delta_feat = X_seq[:, -1, :] - X_seq[:, 0, :]
        flattened_summary = np.concatenate([mean_feat, std_feat, delta_feat], axis=1)

        self.fallback_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            random_state=42
        )
        self.fallback_model.fit(flattened_summary, y_indices)
        self.is_trained = True
        self.save()

        acc = float(np.mean(self.fallback_model.predict(flattened_summary) == y_indices))
        return {
            "num_sequences": len(X_seq),
            "num_classes": len(self.classes),
            "train_accuracy": acc,
            "classes": self.classes,
            "backend": "scikit-learn fallback"
        }

    def predict_sequence(
        self,
        sequence_window: np.ndarray,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        uncertainty_margin: float = UNCERTAINTY_MARGIN
    ) -> Tuple[str, float, bool, Dict[str, float]]:
        """
        Infers sign label from sequence window of shape (seq_len, feature_dim).
        Returns:
            Tuple of:
                - label (str)
                - confidence (float)
                - is_uncertain (bool)
                - prob_dict (dict of class -> prob)
        """
        if not self.is_trained:
            return UNKNOWN_LABEL, 0.0, True, {}

        seq = np.array(sequence_window, dtype=np.float32)
        if seq.ndim == 2:
            seq = np.expand_dims(seq, axis=0) # (1, seq_len, feature_dim)

        # Check for empty frames
        if np.max(np.abs(seq)) < 1e-4:
            return UNKNOWN_LABEL, 0.0, True, {cls: 0.0 for cls in self.classes}

        probs = None
        # PyTorch Inference
        if self.torch_model is not None:
            try:
                import torch
                self.torch_model.eval()
                with torch.no_grad():
                    device = next(self.torch_model.parameters()).device
                    t_in = torch.tensor(seq, dtype=torch.float32).to(device)
                    logits = self.torch_model(t_in)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            except Exception:
                probs = None

        # Fallback Inference
        if probs is None and self.fallback_model is not None:
            mean_feat = np.mean(seq, axis=1)
            std_feat = np.std(seq, axis=1)
            delta_feat = seq[:, -1, :] - seq[:, 0, :]
            summary = np.concatenate([mean_feat, std_feat, delta_feat], axis=1)
            probs = self.fallback_model.predict_proba(summary)[0]

        if probs is None:
            return UNKNOWN_LABEL, 0.0, True, {}

        prob_dict = {
            self.classes[i]: float(probs[i])
            for i in range(len(self.classes))
            if i < len(probs)
        }

        sorted_indices = np.argsort(probs)[::-1]
        top_1_idx = sorted_indices[0]
        top_1_prob = float(probs[top_1_idx])
        top_2_prob = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0

        top_label = self.classes[top_1_idx] if top_1_idx < len(self.classes) else UNKNOWN_LABEL

        is_uncertain = False
        if top_1_prob < confidence_threshold:
            is_uncertain = True
            top_label = UNKNOWN_LABEL
        elif (top_1_prob - top_2_prob) < uncertainty_margin:
            is_uncertain = True
            top_label = f"UNCERTAIN ({top_label}?)"

        return top_label, top_1_prob, is_uncertain, prob_dict

    def save(self):
        """Saves weights and metadata."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        # Save metadata
        meta = {
            "classes": self.classes,
            "sequence_length": self.sequence_length,
            "feature_dim": self.feature_dim,
            "is_trained": self.is_trained
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Save weights
        if self.torch_model is not None:
            try:
                import torch
                torch.save(self.torch_model.state_dict(), self.model_path)
            except Exception as e:
                print(f"[DynamicSignClassifier] Could not save torch weights: {e}")

        if self.fallback_model is not None:
            import joblib
            fallback_path = self.model_path.replace(".pt", "_fallback.joblib")
            joblib.dump(self.fallback_model, fallback_path)

    def load(self):
        """Loads weights and metadata."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.classes = meta.get("classes", self.DEFAULT_DYNAMIC_CLASSES)
                self.sequence_length = meta.get("sequence_length", SEQUENCE_LENGTH)
                self.feature_dim = meta.get("feature_dim", FEATURE_DIM_HOLISTIC)
                self.is_trained = meta.get("is_trained", True)

        try:
            import torch
            if os.path.exists(self.model_path):
                self.torch_model = self._build_torch_network(len(self.classes))
                state_dict = torch.load(self.model_path, map_location=torch.device("cpu"))
                self.torch_model.load_state_dict(state_dict)
                self.torch_model.eval()
                self.is_trained = True
        except Exception:
            self.torch_model = None

        fallback_path = self.model_path.replace(".pt", "_fallback.joblib")
        if os.path.exists(fallback_path):
            import joblib
            self.fallback_model = joblib.load(fallback_path)
            self.is_trained = True
