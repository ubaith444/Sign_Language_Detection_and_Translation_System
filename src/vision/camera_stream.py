import cv2
import threading
import time
import numpy as np
from typing import Optional, Tuple
from src.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS

class CameraStream:
    """
    Threaded OpenCV webcam capture with privacy masking controls
    and fallback synthetic generator for headless or camera-less environments.
    """

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        fps: int = TARGET_FPS
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.current_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.worker_thread: Optional[threading.Thread] = None

        # Privacy Controls
        self.privacy_blur = False
        self.privacy_landmarks_only = False
        self.is_camera_enabled = True
        self.mirror_view = True

        # Fallback simulation counter
        self.sim_frame_idx = 0
        self.using_fallback_stream = False
        self.consecutive_failures = 0
        self.last_reconnect_attempt = 0.0

    def start(self) -> bool:
        """Starts background frame grabber thread with DirectShow resilience."""
        if self.is_running:
            return True

        # Try DirectShow first on Windows for low latency and instant open
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        except Exception:
            self.cap = cv2.VideoCapture(self.camera_index)

        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            # Test 1 read
            ret, _ = self.cap.read()
            if ret:
                self.using_fallback_stream = False
                print(f"[CameraStream] Initialized hardware webcam at index {self.camera_index} (DirectShow).")
            else:
                self.using_fallback_stream = False
                print(f"[CameraStream] Camera opened at index {self.camera_index}, warming up sensors.")
        else:
            print("[CameraStream] Hardware webcam unavailable. Activating synthetic video simulator.")
            self.using_fallback_stream = True

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.worker_thread.start()
        return True

    def _capture_loop(self):
        """Thread loop continuously polling frames with drop tolerance and auto-reconnect."""
        frame_interval = 1.0 / self.fps

        while self.is_running:
            start_time = time.time()

            if not self.is_camera_enabled:
                blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(
                    blank, "Camera Paused (Privacy Mode)",
                    (self.width // 4, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2
                )
                with self.lock:
                    self.current_frame = blank
                time.sleep(0.05)
                continue

            # Auto-reconnection attempt if currently in fallback
            if self.using_fallback_stream and (start_time - self.last_reconnect_attempt > 3.0):
                self.last_reconnect_attempt = start_time
                try:
                    reconnect_cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                    if reconnect_cap.isOpened():
                        ret_test, frame_test = reconnect_cap.read()
                        if ret_test and frame_test is not None:
                            if self.cap:
                                self.cap.release()
                            self.cap = reconnect_cap
                            self.using_fallback_stream = False
                            self.consecutive_failures = 0
                            print(f"[CameraStream] Reconnected to hardware webcam at index {self.camera_index}.")
                        else:
                            reconnect_cap.release()
                except Exception:
                    pass

            frame = None
            if not self.using_fallback_stream and self.cap is not None:
                ret, raw_frame = self.cap.read()
                if ret and raw_frame is not None:
                    frame = raw_frame
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1
                    # Tolerate up to 30 consecutive dropped frames before declaring fallback
                    if self.consecutive_failures > 30:
                        self.using_fallback_stream = True

            if self.using_fallback_stream or frame is None:
                frame = self._generate_simulated_frame()

            # Resize if necessary
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            # Mirror view for intuitive user experience
            if self.mirror_view:
                frame = cv2.flip(frame, 1)

            # Apply Privacy Blur if enabled
            if self.privacy_blur:
                # High kernel Gaussian blur obscuring environmental details
                blurred = cv2.GaussianBlur(frame, (35, 35), 0)
                frame = blurred

            with self.lock:
                self.current_frame = frame

            elapsed = time.time() - start_time
            sleep_time = max(0.001, frame_interval - elapsed)
            time.sleep(sleep_time)

    def _generate_simulated_frame(self) -> np.ndarray:
        """Generates an animated test frame for demonstration."""
        self.sim_frame_idx += 1
        frame = np.full((self.height, self.width, 3), 245, dtype=np.uint8)

        # Draw signer silhouette
        center_x = self.width // 2
        head_y = 140
        cv2.circle(frame, (center_x, head_y), 45, (215, 215, 220), -1)
        cv2.ellipse(frame, (center_x, 300), (90, 110), 0, 0, 180, (190, 190, 200), -1)

        # Draw oscillating hands simulating gesture
        t = self.sim_frame_idx * 0.08
        offset_y = int(np.sin(t) * 35)
        offset_x = int(np.cos(t) * 20)

        # Left hand
        cv2.circle(frame, (center_x - 90 + offset_x, 260 + offset_y), 24, (230, 180, 160), -1)
        # Right hand
        cv2.circle(frame, (center_x + 90 - offset_x, 260 + offset_y), 24, (230, 180, 160), -1)

        cv2.putText(
            frame, "ISL Interactive Simulation Stream",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 2
        )
        return frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the latest captured frame."""
        with self.lock:
            if self.current_frame is None:
                return True, self._generate_simulated_frame()
            return True, self.current_frame.copy()

    def set_privacy_blur(self, enabled: bool):
        """Toggles environmental background privacy blur."""
        self.privacy_blur = enabled

    def set_camera_enabled(self, enabled: bool):
        """Toggles video acquisition."""
        self.is_camera_enabled = enabled

    def stop(self):
        """Releases camera and stops worker thread."""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
