import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any, List
from src.config import (
    KINETIC_VELOCITY_THRESHOLD,
    TRANSITION_REST_FRAMES,
    DEBOUNCE_STABLE_FRAMES,
    CONFIDENCE_THRESHOLD,
    UNKNOWN_LABEL
)
from src.vision.normalizer import GeometricNormalizer

class ContinuousSignSegmenter:
    """
    Detects sign boundaries and segments continuous signing without
    requiring the signer to freeze or artificially hold postures.
    Tracks kinetic energy, prediction consistency, and refractory debouncing.
    """

    def __init__(
        self,
        stable_frames_required: int = DEBOUNCE_STABLE_FRAMES,
        velocity_threshold: float = KINETIC_VELOCITY_THRESHOLD,
        rest_frames_required: int = TRANSITION_REST_FRAMES,
        refractory_period: int = 15
    ):
        self.stable_frames_required = stable_frames_required
        self.velocity_threshold = velocity_threshold
        self.rest_frames_required = rest_frames_required
        self.refractory_period = refractory_period

        # State tracking
        self.prev_vector: Optional[np.ndarray] = None
        self.prediction_history = deque(maxlen=20)
        self.confidence_history = deque(maxlen=20)
        self.velocity_history = deque(maxlen=15)

        self.last_emitted_sign: Optional[str] = None
        self.refractory_counter = 0
        self.low_velocity_counter = 0
        self.is_signing_active = False

    def update(
        self,
        current_vector: np.ndarray,
        predicted_label: str,
        confidence: float,
        is_uncertain: bool
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Processes the current frame's observation and determines if a distinct sign
        should be committed to the sentence.
        Returns:
            Tuple of:
                - committed_sign (str or None)
                - segment_info (dict of telemetry metrics)
        """
        # Calculate instantaneous kinetic velocity
        velocity = GeometricNormalizer.compute_kinetic_velocity(self.prev_vector, current_vector)
        self.prev_vector = current_vector.copy() if current_vector is not None else None

        self.velocity_history.append(velocity)
        self.prediction_history.append(predicted_label)
        self.confidence_history.append(confidence)

        if self.refractory_counter > 0:
            self.refractory_counter -= 1

        # Check movement status
        avg_velocity = np.mean(self.velocity_history) if len(self.velocity_history) > 0 else 0.0

        if avg_velocity < self.velocity_threshold:
            self.low_velocity_counter += 1
        else:
            self.low_velocity_counter = 0
            self.is_signing_active = True

        committed_sign: Optional[str] = None

        # Discard invalid/uncertain labels from committing
        if (
            not is_uncertain
            and predicted_label != UNKNOWN_LABEL
            and not predicted_label.startswith("UNCERTAIN")
            and confidence >= CONFIDENCE_THRESHOLD
        ):
            # Check consistency in recent window
            recent_preds = list(self.prediction_history)[-self.stable_frames_required:]
            if len(recent_preds) >= self.stable_frames_required:
                # If all recent frames agree on this sign
                if all(p == predicted_label for p in recent_preds):
                    # Check refractory guard and duplicate suppression
                    if self.refractory_counter == 0 and predicted_label != self.last_emitted_sign:
                        committed_sign = predicted_label
                        self.last_emitted_sign = predicted_label
                        self.refractory_counter = self.refractory_period

        # If user hands dropped or rested for long enough, clear last emitted sign
        # so they can naturally repeat the same sign if desired
        if self.low_velocity_counter >= self.rest_frames_required * 2:
            self.last_emitted_sign = None
            self.is_signing_active = False

        segment_info = {
            "kinetic_velocity": float(velocity),
            "avg_velocity": float(avg_velocity),
            "is_signing_active": self.is_signing_active,
            "refractory_counter": self.refractory_counter,
            "low_velocity_frames": self.low_velocity_counter,
            "last_emitted_sign": self.last_emitted_sign
        }

        return committed_sign, segment_info

    def reset(self):
        """Resets segmenter states."""
        self.prev_vector = None
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.velocity_history.clear()
        self.last_emitted_sign = None
        self.refractory_counter = 0
        self.low_velocity_counter = 0
        self.is_signing_active = False
