"""
core/camera.py — Camera capture and intrinsic calibration.

CameraCapture runs a background thread that continuously grabs frames so
the UI never blocks waiting for a camera read.  IntrinsicCalibrator collects
checkerboard views and computes the lens matrix and distortion coefficients.
"""

import threading
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from config import (
    CHECKERBOARD_COLS, CHECKERBOARD_ROWS, CHECKERBOARD_SQ_MM,
    INTRINSIC_CALIB_FRAMES, CAMERAS_DIR, MAX_CAMERAS,
)


# ── Threaded camera capture ────────────────────────────────────────────────────

class CameraCapture:
    """
    Wraps an OpenCV VideoCapture in a background thread.

    The latest frame is always available via get_frame() without blocking.
    """

    def __init__(self, index: int):
        self._index   = index
        self._cap     = None
        self._frame   = None
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

    def start(self) -> bool:
        """Open the camera and start the capture thread. Returns True on success."""
        self._cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self._index)   # fallback (Linux / macOS)
        if not self._cap.isOpened():
            return False

        # Prefer 1080p; camera will fall back to its best supported resolution.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS,       0)   # disable autofocus

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop the capture thread and release the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    def get_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent frame, or None if not yet available."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_open(self) -> bool:
        return self._running and self._cap is not None and self._cap.isOpened()

    def _loop(self):
        while self._running:
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    with self._lock:
                        self._frame = frame


# ── Camera enumeration ─────────────────────────────────────────────────────────

def list_available_cameras() -> list[int]:
    """Probe camera indices 0–MAX_CAMERAS and return those that open successfully."""
    available = []
    for i in range(MAX_CAMERAS):
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
    Collects checkerboard frames from a live camera feed and computes
    the camera matrix and distortion coefficients.

    Usage
    -----
    cal = IntrinsicCalibrator()
    while not cal.is_done:
        annotated, accepted = cal.process_frame(frame)
        display(annotated)
    mtx, dist, rms = cal.compute()
    """

    _COOLDOWN_FRAMES = 15   # frames to skip after a successful capture (avoids duplicates)

    def __init__(self):
        self._obj_pts:  list[np.ndarray] = []
        self._img_pts:  list[np.ndarray] = []
        self._img_size: Optional[Tuple[int, int]] = None
        self._cooldown  = 0

        cols, rows = CHECKERBOARD_COLS, CHECKERBOARD_ROWS
        board = np.zeros((cols * rows, 3), np.float32)
        board[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        board *= CHECKERBOARD_SQ_MM
        self._board_3d = board

    @property
    def n_frames(self) -> int:
        return len(self._obj_pts)

    @property
    def is_done(self) -> bool:
        return self.n_frames >= INTRINSIC_CALIB_FRAMES

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Attempt to detect a checkerboard in frame.

        Returns
        -------
        annotated : frame with detection overlay drawn on it
        accepted  : True if this frame was added to the calibration set
        """
        out = frame.copy()
        accepted = False

        if self._cooldown > 0:
            self._cooldown -= 1
            self._annotate_status(out)
            return out, False

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(
            gray, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS), flags
        )

        if found:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(out, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS), corners, found)

            self._img_size = (gray.shape[1], gray.shape[0])
            self._obj_pts.append(self._board_3d.copy())
            self._img_pts.append(corners)
            self._cooldown = self._COOLDOWN_FRAMES
            accepted = True

        self._annotate_status(out)
        return out, accepted

    def compute(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute intrinsics from collected frames.

        Returns
        -------
        mtx  : (3, 3) camera matrix
        dist : (1, 5) distortion coefficients
        rms  : reprojection RMS in pixels
        """
        if not self.is_done:
            raise RuntimeError(f"Need {INTRINSIC_CALIB_FRAMES} frames, have {self.n_frames}.")
        rms, mtx, dist, _, _ = cv2.calibrateCamera(
            self._obj_pts, self._img_pts, self._img_size, None, None
        )
        return mtx, dist, rms

    def _annotate_status(self, frame: np.ndarray):
        remaining = INTRINSIC_CALIB_FRAMES - self.n_frames
        text  = f"Calibration: {self.n_frames}/{INTRINSIC_CALIB_FRAMES} frames"
        color = (0, 255, 0) if self.n_frames > 0 else (0, 180, 255)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if remaining > 0:
            cv2.putText(frame, "Move checkerboard slowly", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)


# ── Intrinsic persistence ──────────────────────────────────────────────────────

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
