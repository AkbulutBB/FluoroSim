"""
core/camera.py — Threaded webcam capture and intrinsic calibration.

Each CameraCapture instance runs a background thread that continuously
reads frames. The UI thread calls get_frame() without blocking.
"""

import cv2
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from config import (
    CHECKERBOARD_COLS, CHECKERBOARD_ROWS, CHECKERBOARD_SQ_MM,
    INTRINSIC_CALIB_FRAMES, CAMERAS_DIR, MAX_CAMERAS_TO_SCAN,
    PREVIEW_W, PREVIEW_H,
)


class CameraCapture:
    """Non-blocking webcam wrapper with threaded frame capture."""

    def __init__(self, index: int):
        self.index      = index
        self._cap       = None
        self._frame     = None
        self._running   = False
        self._lock      = threading.Lock()
        self._thread    = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Open the capture device and start the background thread."""
        print(f"[DEBUG] Opening camera index {self.index}…")
        # Try DirectShow first (Windows); fall back to default
        for backend, name in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY")):
            print(f"[DEBUG]   Trying backend {name}…")
            cap = cv2.VideoCapture(self.index, backend)
            if cap.isOpened():
                print(f"[DEBUG]   ✔ Opened with {name}")
                self._cap = cap
                break
            cap.release()
        if self._cap is None or not self._cap.isOpened():
            print(f"[DEBUG]   ✘ All backends failed for camera {self.index}")
            return False

        self._try_set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._try_set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self._try_set(cv2.CAP_PROP_AUTOFOCUS,       0)
        self._try_set(cv2.CAP_PROP_BUFFERSIZE,       1)

        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[DEBUG] Camera {self.index} capture thread started")
        return True

    def _try_set(self, prop, value):
        """Set a camera property, ignoring errors (virtual cameras may reject them)."""
        try:
            self._cap.set(prop, value)
        except Exception as e:
            print(f"[DEBUG]   cap.set({prop}, {value}) ignored: {e}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    # ── Frame access ───────────────────────────────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent frame, or None if unavailable."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_open(self) -> bool:
        return self._running and self._cap is not None and self._cap.isOpened()

    # ── Internal ───────────────────────────────────────────────────────────

    def _capture_loop(self):
        while self._running:
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    with self._lock:
                        self._frame = frame


# ── Camera enumeration ─────────────────────────────────────────────────────────

def list_available_cameras() -> list[int]:
    """Probe camera indices and return those that open successfully."""
    available = []
    for i in range(MAX_CAMERAS_TO_SCAN):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()
        else:
            cap.release()
    return available


# ── Intrinsic calibration ──────────────────────────────────────────────────────

class IntrinsicCalibrator:
    """
    Collects checkerboard views and computes lens intrinsics.

    Usage:
        cal = IntrinsicCalibrator()
        while not cal.is_done:
            frame = camera.get_frame()
            annotated, accepted = cal.process_frame(frame)
            display(annotated)
        mtx, dist, rms = cal.compute()
    """

    def __init__(self):
        self._obj_pts: list[np.ndarray] = []
        self._img_pts: list[np.ndarray] = []
        self._img_size: Optional[Tuple[int, int]] = None
        self._cooldown = 0   # frames to skip after a successful capture

        # Physical 3-D corners in checkerboard space (Z=0 plane)
        cols, rows = CHECKERBOARD_COLS, CHECKERBOARD_ROWS
        self._board_3d = np.zeros((cols * rows, 3), np.float32)
        self._board_3d[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        self._board_3d *= CHECKERBOARD_SQ_MM

    @property
    def n_frames(self) -> int:
        return len(self._obj_pts)

    @property
    def is_done(self) -> bool:
        return self.n_frames >= INTRINSIC_CALIB_FRAMES

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Detect checkerboard in frame.
        Returns (annotated_frame, was_accepted).
        Accepted frames are throttled to every ~15 frames for diversity.
        """
        annotated = frame.copy()
        accepted  = False

        if self._cooldown > 0:
            self._cooldown -= 1
            cv2.putText(annotated, f"Collected {self.n_frames}/{INTRINSIC_CALIB_FRAMES}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
            return annotated, accepted

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS), None)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(annotated, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS),
                                      corners_refined, ret)

            self._obj_pts.append(self._board_3d)
            self._img_pts.append(corners_refined)
            self._img_size = (gray.shape[1], gray.shape[0])
            self._cooldown  = 15
            accepted = True

        status = f"Collected {self.n_frames}/{INTRINSIC_CALIB_FRAMES} — move board slowly"
        color  = (0, 255, 0) if ret else (0, 120, 255)
        cv2.putText(annotated, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return annotated, accepted

    def compute(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Solve for intrinsic matrix and distortion coefficients."""
        if not self.is_done:
            raise RuntimeError("Not enough calibration frames collected.")
        rms, mtx, dist, _, _ = cv2.calibrateCamera(
            self._obj_pts, self._img_pts, self._img_size, None, None
        )
        return mtx, dist.flatten(), rms


# ── Persistence ────────────────────────────────────────────────────────────────

def save_intrinsics(camera_id: str, mtx: np.ndarray, dist: np.ndarray):
    path = Path(CAMERAS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(str(path / f"intrinsics_{camera_id}.npz"), mtx=mtx, dist=dist)


def load_intrinsics(camera_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    p = Path(CAMERAS_DIR) / f"intrinsics_{camera_id}.npz"
    if not p.exists():
        return None
    data = np.load(str(p))
    return data["mtx"], data["dist"]
