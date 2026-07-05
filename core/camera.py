"""
core/camera.py — Threaded webcam capture + intrinsic calibration I/O.

CameraStream  : background-thread frame grabber (non-blocking latest-frame).
load_intrinsics / save_intrinsics : persist camera matrix + distortion.
calibrate_intrinsics : checkerboard intrinsic calibration helper.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import config as cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Threaded capture
# ─────────────────────────────────────────────────────────────────────────────

class CameraStream:
    """
    Non-blocking webcam capture on a background thread.
    read() always returns the most recent frame (or None before first grab).
    """

    def __init__(self, device_index: int, width: int, height: int, fps: int):
        self._idx     = device_index
        self._width   = width
        self._height  = height
        self._fps     = fps
        self._cap:   Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray]       = None
        self._lock    = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        # CAP_DSHOW gives far faster startup on Windows; harmless elsewhere
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self._idx, backend)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self._idx)   # fallback to default backend
        if not self._cap.isOpened():
            logger.error("Cannot open camera %d", self._idx)
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsic calibration persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_intrinsics(path: str, cam_mtx: np.ndarray, dist: np.ndarray,
                    rms: float = 0.0):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(p), cam_mtx=cam_mtx, dist=dist, rms=rms)
    logger.info("Saved intrinsics to %s (RMS %.3f px)", path, rms)


def load_intrinsics(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    p = Path(path)
    if not p.exists():
        return None
    data = np.load(str(p))
    return data["cam_mtx"], data["dist"]


# ─────────────────────────────────────────────────────────────────────────────
# Checkerboard intrinsic calibration
# ─────────────────────────────────────────────────────────────────────────────

def make_object_points() -> np.ndarray:
    """Generate 3-D object points for the checkerboard (Z=0 plane)."""
    cols, rows = cfg.CHECKER_COLS, cfg.CHECKER_ROWS
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= cfg.CHECKER_SQ_MM
    return objp


def find_checkerboard(frame: np.ndarray):
    """Detect checkerboard corners; return (found, refined_corners, gray)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, (cfg.CHECKER_COLS, cfg.CHECKER_ROWS),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners, gray


def compute_intrinsics(
    obj_points: list, img_points: list, image_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run cv2.calibrateCamera; return (cam_mtx, dist, rms)."""
    rms, cam_mtx, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    return cam_mtx, dist, float(rms)


# ─────────────────────────────────────────────────────────────────────────────
# ChArUco intrinsic calibration (uses the same board the tracker uses)
# ─────────────────────────────────────────────────────────────────────────────

def make_charuco_board():
    """Build the ChArUco board object from config (matches BoardTracker)."""
    return cv2.aruco.CharucoBoard(
        (cfg.CHARUCO_COLS, cfg.CHARUCO_ROWS),
        cfg.CHARUCO_SQUARE_MM,
        cfg.CHARUCO_MARKER_MM,
        cfg.CHARUCO_BOARD_DICT,
    )


def detect_charuco(gray: np.ndarray, detector) -> tuple:
    """
    Detect ChArUco corners in a grayscale frame.
    Returns (found_bool, charuco_corners, charuco_ids, n_corners).
    """
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    n = 0 if ch_ids is None else len(ch_ids)
    found = n >= cfg.MIN_CHARUCO_CORNERS
    return found, ch_corners, ch_ids, n


def compute_intrinsics_charuco(
    board, captured: list, image_size: Tuple[int, int]
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Compute intrinsics from a list of captured (charuco_corners, charuco_ids).
    Returns (cam_mtx, dist, rms) or None if too few usable views.
    """
    all_obj, all_img = [], []
    for ch_corners, ch_ids in captured:
        obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
        if obj_pts is not None and len(obj_pts) >= cfg.MIN_CHARUCO_CORNERS:
            all_obj.append(obj_pts)
            all_img.append(img_pts)
    if len(all_obj) < 8:
        return None
    rms, cam_mtx, dist, _, _ = cv2.calibrateCamera(
        all_obj, all_img, image_size, None, None
    )
    return cam_mtx, dist, float(rms)
