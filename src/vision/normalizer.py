import numpy as np
from typing import Optional, Tuple, Dict, Any

class GeometricNormalizer:
    """
    Normalizes raw MediaPipe landmarks to achieve scale, position,
    and signer-distance invariance.
    """

    @staticmethod
    def normalize_hand_landmarks(landmarks_xyz: np.ndarray) -> np.ndarray:
        """
        Normalizes 21 3D hand landmarks (21, 3):
        1. Translates coordinates so the wrist (index 0) is at origin (0, 0, 0).
        2. Scales coordinates by the distance between the wrist (0) and middle MCP (9).
        Returns normalized array of shape (21, 3).
        """
        if landmarks_xyz is None or len(landmarks_xyz) == 0:
            return np.zeros((21, 3), dtype=np.float32)

        coords = np.array(landmarks_xyz, dtype=np.float32)
        if coords.shape != (21, 3):
            coords = coords.reshape((21, 3))

        # Check if hand is detected (not all zeros)
        if np.all(coords == 0):
            return coords

        wrist = coords[0, :].copy()
        translated = coords - wrist

        # Compute reference scale distance: wrist to middle finger base (landmark 9)
        ref_distance = np.linalg.norm(translated[9])
        if ref_distance > 1e-4:
            scaled = translated / ref_distance
        else:
            # Fallback to max bounding distance if MCP distance is degenerate
            max_dist = np.max(np.linalg.norm(translated, axis=1))
            scaled = translated / (max_dist + 1e-6)

        return scaled

    @staticmethod
    def normalize_pose_landmarks(pose_xyz: np.ndarray) -> np.ndarray:
        """
        Normalizes 12 key upper-body landmarks (12, 3):
        Landmarks: Nose, L/R Shoulders, L/R Elbows, L/R Wrists, L/R Hips, etc.
        1. Translates relative to the mid-shoulder point.
        2. Scales by the inter-shoulder distance.
        """
        if pose_xyz is None or len(pose_xyz) == 0:
            return np.zeros((12, 3), dtype=np.float32)

        coords = np.array(pose_xyz, dtype=np.float32)
        if coords.shape != (12, 3):
            coords = coords.reshape((12, 3))

        if np.all(coords == 0):
            return coords

        # Assuming indices: 1 is Left Shoulder, 2 is Right Shoulder
        left_shoulder = coords[1]
        right_shoulder = coords[2]
        mid_shoulder = (left_shoulder + right_shoulder) / 2.0

        translated = coords - mid_shoulder
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
        scale = shoulder_dist if shoulder_dist > 1e-3 else 1.0

        return translated / scale

    @staticmethod
    def normalize_face_landmarks(face_xyz: np.ndarray) -> np.ndarray:
        """
        Normalizes 20 key facial landmarks (20, 3) representing non-manual cues:
        Brows, mouth contour, chin.
        1. Translates relative to center of facial landmarks.
        2. Scales by facial bounding box span.
        """
        if face_xyz is None or len(face_xyz) == 0:
            return np.zeros((20, 3), dtype=np.float32)

        coords = np.array(face_xyz, dtype=np.float32)
        if coords.shape != (20, 3):
            coords = coords.reshape((20, 3))

        if np.all(coords == 0):
            return coords

        center = np.mean(coords, axis=0)
        translated = coords - center
        span = np.ptp(translated[:, :2])
        scale = span if span > 1e-3 else 1.0

        return translated / scale

    @classmethod
    def assemble_holistic_vector(
        cls,
        left_hand: Optional[np.ndarray],
        right_hand: Optional[np.ndarray],
        pose: Optional[np.ndarray],
        face: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Combines and normalizes all holistic streams into a single 1D feature vector.
        Vector layout:
        [0:63]   -> Left Hand (21 * 3)
        [63:126] -> Right Hand (21 * 3)
        [126:162]-> Upper Body Pose (12 * 3)
        [162:222]-> Non-manual Face cues (20 * 3)
        Total length = 222
        """
        norm_lh = cls.normalize_hand_landmarks(left_hand)
        norm_rh = cls.normalize_hand_landmarks(right_hand)
        norm_pose = cls.normalize_pose_landmarks(pose)
        norm_face = cls.normalize_face_landmarks(face)

        holistic_vector = np.concatenate([
            norm_lh.flatten(),
            norm_rh.flatten(),
            norm_pose.flatten(),
            norm_face.flatten()
        ]).astype(np.float32)

        return holistic_vector

    @staticmethod
    def compute_kinetic_velocity(
        prev_vector: Optional[np.ndarray],
        curr_vector: Optional[np.ndarray]
    ) -> float:
        """
        Computes the kinetic displacement / velocity between consecutive frames.
        Focuses primarily on hand landmark displacement (indices 0 to 126).
        Used for continuous sign boundary detection and movement onset.
        """
        if prev_vector is None or curr_vector is None:
            return 0.0

        hand_indices = 126
        p_hand = prev_vector[:hand_indices]
        c_hand = curr_vector[:hand_indices]

        # Calculate average per-landmark Euclidean displacement
        p_hand_reshaped = p_hand.reshape(-1, 3)
        c_hand_reshaped = c_hand.reshape(-1, 3)

        # Ignore if hands are not present in either frame
        p_active = not np.all(p_hand == 0)
        c_active = not np.all(c_hand == 0)

        if not p_active or not c_active:
            return 0.0

        displacements = np.linalg.norm(c_hand_reshaped - p_hand_reshaped, axis=1)
        return float(np.mean(displacements))
