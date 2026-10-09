"""
Non-Manual Facial Expression Classifier for Sign Language Modality.
Classifies linguistic sentence modalities based on facial gestures and head pose:
- Declarative Statement (neutral brows, steady head)
- Polar Question (Yes/No question: raised eyebrows, widened eyes, head tilt forward)
- Wh-Question (Content question: furrowed/lowered eyebrows, head tilt)
- Negative Statement (Head shake oscillation, brow constriction)
"""

import numpy as np
from collections import deque
from typing import Dict, Any, List, Optional, Tuple


class FacialModalityClassifier:
    """
    Analyzes non-manual facial landmarks to classify sentence grammatical modality.
    Applies adaptive neutral face baseline calibration and temporal sliding window smoothing.
    """

    MODALITY_STATEMENT = "DECLARATIVE_STATEMENT"
    MODALITY_POLAR_QUESTION = "POLAR_QUESTION"      # Yes/No Question (Raised eyebrows)
    MODALITY_WH_QUESTION = "WH_QUESTION"            # Wh-Question (Furrowed eyebrows)
    MODALITY_NEGATIVE = "NEGATIVE_STATEMENT"        # Negation (Head shake)

    def __init__(self, history_len: int = 15):
        self.history_len = history_len
        self.yaw_history = deque(maxlen=history_len)
        self.pitch_history = deque(maxlen=history_len)
        self.elevation_history = deque(maxlen=history_len)
        self.furrow_history = deque(maxlen=history_len)

        # Adaptive baselines (calibrated dynamically during neutral signing)
        self.baseline_elevation = 0.08
        self.baseline_brow_dist = 0.14
        self.calibration_samples = 0
        self.max_calibration_samples = 60

    def process_face_landmarks(
        self,
        face_landmarks: np.ndarray,
        current_gloss: str = ""
    ) -> Dict[str, Any]:
        """
        Extracts facial features and determines non-manual linguistic modality.
        Expects face landmarks array of shape (N, 3), typically 20 keypoints.
        """
        if face_landmarks is None or len(face_landmarks) < 18 or np.linalg.norm(face_landmarks) < 0.01:
            return self._default_statement_result("No face landmarks detected")

        # Map indices from the standard 20 face keypoints:
        # [0..2]: Left Brow (70, 63, 105)
        # [3..5]: Right Brow (336, 293, 300)
        # [6..13]: Lips (61: left corner, 291: right corner, 8: upper lip, 9: lower lip...)
        # [14]: Nose Tip (1)
        # [15]: Chin (152)
        # [16]: Left face boundary (234)
        # [17]: Right face boundary (454)
        # [18]: Forehead top (10)
        # [19]: Glabella / between brows (168)

        left_brow_mid = face_landmarks[1]     # index 63
        left_brow_inner = face_landmarks[2]   # index 105
        right_brow_inner = face_landmarks[3]  # index 336
        right_brow_mid = face_landmarks[4]    # index 293

        mouth_corner_left = face_landmarks[6]   # index 61
        mouth_corner_right = face_landmarks[7]  # index 291
        upper_lip = face_landmarks[10] if len(face_landmarks) > 10 else face_landmarks[6]  # 13
        lower_lip = face_landmarks[11] if len(face_landmarks) > 11 else face_landmarks[7]  # 14

        nose_tip = face_landmarks[14]           # index 1
        chin = face_landmarks[15]               # index 152
        left_cheek = face_landmarks[16]         # index 234
        right_cheek = face_landmarks[17]        # index 454
        glabella = face_landmarks[19] if len(face_landmarks) > 19 else (left_brow_inner + right_brow_inner) / 2.0

        # Normalization scale: Face width (distance between cheeks)
        face_width = float(np.linalg.norm(right_cheek - left_cheek))
        if face_width < 1e-4:
            face_width = 0.35  # standard fallback scale

        # 1. Eyebrow Elevation (relative to glabella/nose)
        # In image coords, smaller Y is higher on face
        avg_brow_y = (left_brow_mid[1] + right_brow_mid[1]) / 2.0
        eyebrow_elevation = float((nose_tip[1] - avg_brow_y) / face_width)

        # 2. Eyebrow Furrow (distance between inner eyebrows)
        inner_brow_dist = float(np.linalg.norm(right_brow_inner - left_brow_inner) / face_width)

        # 3. Head Pose Estimation
        # Yaw: asymmetry between nose tip and cheeks
        dist_left_cheek = float(np.linalg.norm(nose_tip[:2] - left_cheek[:2]))
        dist_right_cheek = float(np.linalg.norm(nose_tip[:2] - right_cheek[:2]))
        head_yaw = float((dist_right_cheek - dist_left_cheek) / face_width)

        # Pitch: distance from glabella to nose vs nose to chin
        dist_upper = float(np.linalg.norm(glabella[:2] - nose_tip[:2]))
        dist_lower = float(np.linalg.norm(chin[:2] - nose_tip[:2]))
        head_pitch = float((dist_upper - dist_lower) / face_width)

        # 4. Mouth Aperture
        mouth_opening = float(np.linalg.norm(lower_lip - upper_lip) / face_width)

        # Update historical buffers
        self.yaw_history.append(head_yaw)
        self.pitch_history.append(head_pitch)
        self.elevation_history.append(eyebrow_elevation)
        self.furrow_history.append(inner_brow_dist)

        # Dynamic baseline calibration during neutral signing
        if self.calibration_samples < self.max_calibration_samples:
            self.baseline_elevation = (self.baseline_elevation * 0.9) + (eyebrow_elevation * 0.1)
            self.baseline_brow_dist = (self.baseline_brow_dist * 0.9) + (inner_brow_dist * 0.1)
            self.calibration_samples += 1

        # 5. Evaluate Temporal Metrics (Head shake oscillation)
        head_shake_score = self._compute_oscillation(self.yaw_history)

        # Relative differences from baseline
        elevation_delta = eyebrow_elevation - self.baseline_elevation
        furrow_delta = self.baseline_brow_dist - inner_brow_dist  # positive means furrowed / narrowed

        # 6. Modality Classification Logic
        modality = self.MODALITY_STATEMENT
        confidence = 0.85
        cues = []

        # Check for Head Shake Negation
        if head_shake_score > 0.35:
            modality = self.MODALITY_NEGATIVE
            confidence = min(0.98, 0.70 + head_shake_score * 0.5)
            cues.append("Head shake oscillation detected")

        # Check for Polar / Yes-No Question (distinct brow raise)
        elif elevation_delta > 0.035 and furrow_delta < 0.02:
            modality = self.MODALITY_POLAR_QUESTION
            confidence = min(0.99, 0.65 + (elevation_delta / 0.06) * 0.3)
            cues.append(f"Eyebrow raise (+{elevation_delta:.2f})")
            if head_pitch < -0.05:
                cues.append("Head tilted forward")

        # Check for Wh-Question (distinct brow furrow / contraction)
        elif furrow_delta > 0.025 or (current_gloss in ["WHAT", "WHERE", "WHO", "WHY", "HOW", "WHICH"] and elevation_delta <= 0.01):
            modality = self.MODALITY_WH_QUESTION
            confidence = min(0.98, 0.70 + (furrow_delta / 0.05) * 0.25)
            cues.append(f"Eyebrow furrow (+{furrow_delta:.2f})")

        else:
            modality = self.MODALITY_STATEMENT
            confidence = 0.90
            cues.append("Neutral brow and head alignment")

        # Punctuation recommendation
        if modality in [self.MODALITY_POLAR_QUESTION, self.MODALITY_WH_QUESTION]:
            punctuation = "?"
        elif modality == self.MODALITY_NEGATIVE:
            punctuation = "."
        else:
            punctuation = "."

        return {
            "modality": modality,
            "modality_label": self._get_human_label(modality),
            "confidence": round(float(confidence), 2),
            "punctuation": punctuation,
            "cues": cues,
            "metrics": {
                "eyebrow_elevation": round(eyebrow_elevation, 3),
                "elevation_delta": round(elevation_delta, 3),
                "inner_brow_dist": round(inner_brow_dist, 3),
                "furrow_delta": round(furrow_delta, 3),
                "head_yaw": round(head_yaw, 3),
                "head_pitch": round(head_pitch, 3),
                "mouth_opening": round(mouth_opening, 3),
                "head_shake_score": round(head_shake_score, 3)
            }
        }

    def _compute_oscillation(self, history: deque) -> float:
        """Measures frequency of directional sign reversals in yaw (head shaking)."""
        if len(history) < 8:
            return 0.0
        diffs = np.diff(list(history))
        # Count sign flips
        sign_flips = 0
        for i in range(1, len(diffs)):
            if (diffs[i] * diffs[i - 1]) < -1e-5 and abs(diffs[i]) > 0.015:
                sign_flips += 1
        return float(min(1.0, sign_flips / 3.0))

    def _get_human_label(self, modality: str) -> str:
        labels = {
            self.MODALITY_STATEMENT: "Declarative Statement",
            self.MODALITY_POLAR_QUESTION: "Polar Question (Yes/No)",
            self.MODALITY_WH_QUESTION: "Wh-Question (Content)",
            self.MODALITY_NEGATIVE: "Negative Statement"
        }
        return labels.get(modality, "Declarative Statement")

    def _default_statement_result(self, reason: str) -> Dict[str, Any]:
        return {
            "modality": self.MODALITY_STATEMENT,
            "modality_label": "Declarative Statement",
            "confidence": 0.80,
            "punctuation": ".",
            "cues": [reason],
            "metrics": {
                "eyebrow_elevation": 0.0,
                "elevation_delta": 0.0,
                "inner_brow_dist": 0.0,
                "furrow_delta": 0.0,
                "head_yaw": 0.0,
                "head_pitch": 0.0,
                "mouth_opening": 0.0,
                "head_shake_score": 0.0
            }
        }
