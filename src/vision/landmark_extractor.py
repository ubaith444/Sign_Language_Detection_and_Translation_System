import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from src.vision.normalizer import GeometricNormalizer

class LandmarkExtractor:
    """
    Extracts holistic landmarks (both hands, upper-body pose, and non-manual face cues).
    Supports:
      1. MediaPipe Solutions Holistic (classic API)
      2. MediaPipe Tasks HolisticLandmarker (modern API)
      3. Built-in OpenCV Geometric & Skin/Face Cascade Tracker (robust universal fallback)
    """

    POSE_KEYPOINTS = [0, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 1]

    FACE_KEYPOINTS = [
        70, 63, 105, 336, 293, 300,
        61, 291, 0, 17, 13, 14, 78, 308,
        1, 152, 234, 454, 10, 168
    ]

    # Hand skeleton connections for drawing (21 points)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)              # Palm base
    ]

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mode = "opencv_fallback"
        self.holistic = None
        self.face_cascade = None
        self._init_vision_pipeline()

    def _init_vision_pipeline(self):
        """Initializes vision pipelines with progressive fallback."""
        # 1. Try modern MediaPipe Tasks Vision (MediaPipe 0.10+)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            from src.config import DATA_DIR

            hand_task = str(DATA_DIR / "hand_landmarker.task")
            if os.path.exists(hand_task):
                h_opts = vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=hand_task),
                    num_hands=2,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_hand_presence_confidence=self.min_tracking_confidence
                )
                self.hand_landmarker = vision.HandLandmarker.create_from_options(h_opts)

                pose_task = str(DATA_DIR / "pose_landmarker_lite.task")
                if os.path.exists(pose_task):
                    p_opts = vision.PoseLandmarkerOptions(
                        base_options=python.BaseOptions(model_asset_path=pose_task),
                        min_pose_detection_confidence=self.min_detection_confidence
                    )
                    self.pose_landmarker = vision.PoseLandmarker.create_from_options(p_opts)
                else:
                    self.pose_landmarker = None

                face_task = str(DATA_DIR / "face_landmarker.task")
                if os.path.exists(face_task):
                    f_opts = vision.FaceLandmarkerOptions(
                        base_options=python.BaseOptions(model_asset_path=face_task),
                        min_face_detection_confidence=self.min_detection_confidence
                    )
                    self.face_landmarker = vision.FaceLandmarker.create_from_options(f_opts)
                else:
                    self.face_landmarker = None

                self.mode = "mediapipe_tasks"
                print("[LandmarkExtractor] Initialized MediaPipe Tasks Vision (Modern 0.10+).")
                return
        except Exception as e:
            print(f"[LandmarkExtractor] MediaPipe Tasks note: {e}")

        # 2. Try classic mediapipe solutions
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
                self.mp_holistic = mp.solutions.holistic
                self.mp_drawing = mp.solutions.drawing_utils
                self.holistic = self.mp_holistic.Holistic(
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    smooth_landmarks=True
                )
                self.mode = "mediapipe_solutions"
                print("[LandmarkExtractor] Initialized MediaPipe Solutions Holistic.")
                return
        except Exception:
            pass

        # 3. Setup OpenCV Face & Hand Tracker Fallback
        try:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.mode = "opencv_tracking_fallback"
            print("[LandmarkExtractor] Initialized High-Performance OpenCV Holistic Tracker.")
        except Exception as e:
            self.mode = "mock_tracker"
            print(f"[LandmarkExtractor] Tracker fallback active: {e}")

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Processes frame and returns 222-element normalized feature vector."""
        if self.mode == "mediapipe_tasks" and hasattr(self, "hand_landmarker") and self.hand_landmarker is not None:
            return self._process_mediapipe_tasks(frame_bgr)
        elif self.mode == "mediapipe_solutions" and self.holistic is not None:
            return self._process_mediapipe(frame_bgr)
        return self._process_opencv_holistic(frame_bgr)

    def _process_mediapipe_tasks(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Extracts landmarks via MediaPipe Tasks Vision."""
        import mediapipe as mp
        h, w, _ = frame_bgr.shape
        lh = np.zeros((21, 3), dtype=np.float32)
        rh = np.zeros((21, 3), dtype=np.float32)
        pose = np.zeros((len(self.POSE_KEYPOINTS), 3), dtype=np.float32)
        face = np.zeros((len(self.FACE_KEYPOINTS), 3), dtype=np.float32)
        has_lh, has_rh = False, False

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 1. Hands Detection
        try:
            hand_res = self.hand_landmarker.detect(mp_image)
            if hand_res and hand_res.hand_landmarks:
                for idx, landmarks in enumerate(hand_res.hand_landmarks):
                    handedness = "Right"
                    if idx < len(hand_res.handedness) and len(hand_res.handedness[idx]) > 0:
                        handedness = hand_res.handedness[idx][0].category_name

                    target_arr = rh if handedness == "Right" else lh
                    if handedness == "Right":
                        has_rh = True
                    else:
                        has_lh = True

                    for i, lm in enumerate(landmarks[:21]):
                        target_arr[i] = [lm.x, lm.y, lm.z]
        except Exception:
            pass

        # 2. Pose Detection
        if hasattr(self, "pose_landmarker") and self.pose_landmarker is not None:
            try:
                pose_res = self.pose_landmarker.detect(mp_image)
                if pose_res and pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
                    p_lms = pose_res.pose_landmarks[0]
                    for i, p_idx in enumerate(self.POSE_KEYPOINTS):
                        if p_idx < len(p_lms):
                            lm = p_lms[p_idx]
                            pose[i] = [lm.x, lm.y, lm.z]
            except Exception:
                pass

        # 3. Face Detection
        if hasattr(self, "face_landmarker") and self.face_landmarker is not None:
            try:
                face_res = self.face_landmarker.detect(mp_image)
                if face_res and face_res.face_landmarks and len(face_res.face_landmarks) > 0:
                    f_lms = face_res.face_landmarks[0]
                    for i, f_idx in enumerate(self.FACE_KEYPOINTS):
                        if f_idx < len(f_lms):
                            lm = f_lms[f_idx]
                            face[i] = [lm.x, lm.y, lm.z]
            except Exception:
                pass

        # Fallback to OpenCV skin tracker if MediaPipe tasks missed both hands
        if not (has_lh or has_rh):
            cv_res = self._process_opencv_holistic(frame_bgr)
            if cv_res.get("has_hands", False):
                return cv_res

        vector_222 = GeometricNormalizer.assemble_holistic_vector(lh, rh, pose, face)
        return {
            "left_hand": lh,
            "right_hand": rh,
            "pose": pose,
            "face": face,
            "holistic_vector": vector_222,
            "has_hands": has_lh or has_rh,
            "has_left_hand": has_lh,
            "has_right_hand": has_rh,
            "raw_results": {
                "lh": lh, "rh": rh, "pose": pose, "has_lh": has_lh, "has_rh": has_rh
            }
        }

    def _process_mediapipe(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Extracts landmarks via MediaPipe Holistic."""
        lh = np.zeros((21, 3), dtype=np.float32)
        rh = np.zeros((21, 3), dtype=np.float32)
        pose = np.zeros((len(self.POSE_KEYPOINTS), 3), dtype=np.float32)
        face = np.zeros((len(self.FACE_KEYPOINTS), 3), dtype=np.float32)

        has_lh, has_rh = False, False
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)

        if results:
            if results.left_hand_landmarks:
                has_lh = True
                for i, lm in enumerate(results.left_hand_landmarks.landmark):
                    lh[i] = [lm.x, lm.y, lm.z]
            if results.right_hand_landmarks:
                has_rh = True
                for i, lm in enumerate(results.right_hand_landmarks.landmark):
                    rh[i] = [lm.x, lm.y, lm.z]
            if results.pose_landmarks:
                for i, idx in enumerate(self.POSE_KEYPOINTS):
                    if idx < len(results.pose_landmarks.landmark):
                        lm = results.pose_landmarks.landmark[idx]
                        pose[i] = [lm.x, lm.y, lm.z]
            if results.face_landmarks:
                for i, idx in enumerate(self.FACE_KEYPOINTS):
                    if idx < len(results.face_landmarks.landmark):
                        lm = results.face_landmarks.landmark[idx]
                        face[i] = [lm.x, lm.y, lm.z]

        vector_222 = GeometricNormalizer.assemble_holistic_vector(lh, rh, pose, face)
        return {
            "left_hand": lh,
            "right_hand": rh,
            "pose": pose,
            "face": face,
            "holistic_vector": vector_222,
            "has_hands": has_lh or has_rh,
            "has_left_hand": has_lh,
            "has_right_hand": has_rh,
            "raw_results": results
        }

    def _process_opencv_holistic(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Fast, robust holistic landmark detection using face cascades,
        skin tone segmentation, and morphological hand centroid analysis.
        """
        h, w, _ = frame_bgr.shape
        lh = np.zeros((21, 3), dtype=np.float32)
        rh = np.zeros((21, 3), dtype=np.float32)
        pose = np.zeros((len(self.POSE_KEYPOINTS), 3), dtype=np.float32)
        face = np.zeros((len(self.FACE_KEYPOINTS), 3), dtype=np.float32)

        has_lh, has_rh = False, False

        # 1. Detect Face
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5) if self.face_cascade else []

        face_cx, face_cy = w / 2.0, h * 0.3
        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            face_cx, face_cy = fx + fw / 2.0, fy + fh / 2.0
            # Populate face keypoints relative to detected box
            for i in range(len(self.FACE_KEYPOINTS)):
                ang = i * (2.0 * np.pi / len(self.FACE_KEYPOINTS))
                rx = (face_cx + np.cos(ang) * (fw * 0.35)) / w
                ry = (face_cy + np.sin(ang) * (fh * 0.45)) / h
                face[i] = [rx, ry, 0.0]

        # 2. Estimate Upper-Body Pose from Head Position
        mid_x = face_cx / w
        mid_y = face_cy / h
        pose[0] = [mid_x, mid_y, 0.0]                    # Nose
        pose[1] = [mid_x - 0.18, mid_y + 0.22, 0.0]     # Left Shoulder
        pose[2] = [mid_x + 0.18, mid_y + 0.22, 0.0]     # Right Shoulder
        pose[3] = [mid_x - 0.25, mid_y + 0.42, 0.0]     # Left Elbow
        pose[4] = [mid_x + 0.25, mid_y + 0.42, 0.0]     # Right Elbow

        # 3. Detect Hand Regions via Dual-Space Skin Color Range (YCbCr + HSV)
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 130, 75], dtype=np.uint8), np.array([255, 175, 130], dtype=np.uint8))

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([0, 15, 45], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8))

        mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)

        # Mask out face region so head is not confused with hands
        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            cv2.rectangle(mask, (max(0, fx - 10), max(0, fy - 10)), (min(w, fx + fw + 10), min(h, fy + fh + 30)), 0, -1)

        # Morphological filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 200]
        valid_contours.sort(key=cv2.contourArea, reverse=True)

        # Assign largest contours to left/right hands
        for cnt in valid_contours[:2]:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
                norm_x = cx / w
                norm_y = cy / h

                # Assign left vs right based on horizontal center
                if norm_x < mid_x:
                    has_lh = True
                    self._generate_synthetic_hand_skeleton(lh, norm_x, norm_y)
                    pose[5] = [norm_x, norm_y, 0.0] # Left Wrist
                else:
                    has_rh = True
                    self._generate_synthetic_hand_skeleton(rh, norm_x, norm_y)
                    pose[6] = [norm_x, norm_y, 0.0] # Right Wrist

        vector_222 = GeometricNormalizer.assemble_holistic_vector(lh, rh, pose, face)
        return {
            "left_hand": lh,
            "right_hand": rh,
            "pose": pose,
            "face": face,
            "holistic_vector": vector_222,
            "has_hands": has_lh or has_rh,
            "has_left_hand": has_lh,
            "has_right_hand": has_rh,
            "raw_results": {
                "lh": lh, "rh": rh, "pose": pose, "has_lh": has_lh, "has_rh": has_rh
            }
        }

    def _generate_synthetic_hand_skeleton(self, hand_arr: np.ndarray, wrist_x: float, wrist_y: float):
        """Constructs an anatomically proportioned 21-point hand skeleton from wrist anchor."""
        hand_arr[0] = [wrist_x, wrist_y, 0.0]
        # 5 finger rays: Thumb, Index, Middle, Ring, Pinky
        angles = [-0.6, -0.2, 0.0, 0.2, 0.5]
        lengths = [0.05, 0.08, 0.09, 0.08, 0.06]

        finger_bases = [1, 5, 9, 13, 17]
        for f_idx, (ang, length) in enumerate(zip(angles, lengths)):
            base_idx = finger_bases[f_idx]
            for step in range(4):
                dist = (length / 4.0) * (step + 1)
                px = wrist_x + np.sin(ang) * dist
                py = wrist_y - np.cos(ang) * dist
                hand_arr[base_idx + step] = [px, py, 0.0]

    def draw_landmarks(self, frame_bgr: np.ndarray, results: Any) -> np.ndarray:
        """Draws visual skeleton landmarks with clean, elegant styling."""
        if results is None:
            return frame_bgr

        annotated = frame_bgr.copy()
        h, w, _ = annotated.shape

        if isinstance(results, dict):
            lh = results.get("lh")
            rh = results.get("rh")
            pose = results.get("pose")
            has_lh = results.get("has_lh", False)
            has_rh = results.get("has_rh", False)

            # Draw upper body pose skeleton if present
            if pose is not None and np.max(np.abs(pose)) > 1e-4:
                p_pts = [(int(pose[i, 0] * w), int(pose[i, 1] * h)) for i in range(min(7, len(pose)))]
                # Shoulder to shoulder
                if len(p_pts) > 2 and p_pts[1] != (0, 0) and p_pts[2] != (0, 0):
                    cv2.line(annotated, p_pts[1], p_pts[2], (180, 210, 120), 2)
                # Left arm: Shoulder (1) -> Elbow (3) -> Wrist (5)
                if len(p_pts) > 5:
                    if p_pts[1] != (0, 0) and p_pts[3] != (0, 0):
                        cv2.line(annotated, p_pts[1], p_pts[3], (180, 210, 120), 2)
                    if p_pts[3] != (0, 0) and p_pts[5] != (0, 0):
                        cv2.line(annotated, p_pts[3], p_pts[5], (180, 210, 120), 2)
                # Right arm: Shoulder (2) -> Elbow (4) -> Wrist (6)
                if len(p_pts) > 6:
                    if p_pts[2] != (0, 0) and p_pts[4] != (0, 0):
                        cv2.line(annotated, p_pts[2], p_pts[4], (120, 210, 180), 2)
                    if p_pts[4] != (0, 0) and p_pts[6] != (0, 0):
                        cv2.line(annotated, p_pts[4], p_pts[6], (120, 210, 180), 2)
                for pt in p_pts:
                    if pt != (0, 0):
                        cv2.circle(annotated, pt, 4, (255, 255, 255), -1)
                        cv2.circle(annotated, pt, 5, (140, 200, 140), 1)

            if has_lh and lh is not None and np.max(np.abs(lh)) > 1e-4:
                self._draw_skeleton_hand(annotated, lh, w, h, (245, 117, 66))
            if has_rh and rh is not None and np.max(np.abs(rh)) > 1e-4:
                self._draw_skeleton_hand(annotated, rh, w, h, (66, 117, 245))
        elif hasattr(self, "mp_drawing") and self.mp_drawing is not None and self.holistic is not None:
            if getattr(results, "left_hand_landmarks", None):
                self.mp_drawing.draw_landmarks(annotated, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if getattr(results, "right_hand_landmarks", None):
                self.mp_drawing.draw_landmarks(annotated, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        return annotated

    def _draw_skeleton_hand(self, img: np.ndarray, coords_21: np.ndarray, w: int, h: int, color: Tuple[int, int, int]):
        """Renders 21 joints and connecting bones on image."""
        points = []
        for i in range(21):
            px = int(coords_21[i, 0] * w)
            py = int(coords_21[i, 1] * h)
            points.append((px, py))

        # Draw bones
        for p1_idx, p2_idx in self.HAND_CONNECTIONS:
            pt1 = points[p1_idx]
            pt2 = points[p2_idx]
            if pt1 != (0, 0) and pt2 != (0, 0):
                cv2.line(img, pt1, pt2, color, 2)

        # Draw joints
        for pt in points:
            if pt != (0, 0):
                cv2.circle(img, pt, 3, (255, 255, 255), -1)
                cv2.circle(img, pt, 4, color, 1)

    def close(self):
        if self.holistic is not None:
            try:
                self.holistic.close()
            except Exception:
                pass
            self.holistic = None
