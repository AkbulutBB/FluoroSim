"""
core/intrinsics.py  —  Per-camera lens calibration
===================================================

A camera's intrinsics (focal lengths, principal point, lens distortion) depend
only on the lens/sensor, NOT on where the camera sits.  So this is a one-time
job per webcam: show the ChArUco board to the camera from several angles, and
we solve for the intrinsics.  Repositioning the camera afterwards needs no
recalibration — the live board tracker handles that every frame.

Calibration quality is reported as the RMS reprojection error in pixels;
under ~1 px is good for this working distance.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import json

import cv2
import numpy as np

import config
from core import markers, paths


@dataclass
class CameraIntrinsics:
    mtx:        np.ndarray   # 3x3 camera matrix
    dist:       np.ndarray   # distortion coefficients
    image_size: Tuple[int, int]   # (w, h)
    rms:        float        # calibration RMS reprojection error (px)
    n_views:    int

    # ---- persistence -------------------------------------------------------
    def save(self, role: str) -> str:
        paths.ensure_dirs()
        fn = paths.camera_file(role)
        with open(fn, "w") as f:
            json.dump({
                "mtx": self.mtx.tolist(),
                "dist": self.dist.tolist(),
                "image_size": list(self.image_size),
                "rms": self.rms,
                "n_views": self.n_views,
            }, f, indent=2)
        return fn

    @classmethod
    def load(cls, role: str) -> Optional["CameraIntrinsics"]:
        fn = paths.camera_file(role)
        try:
            with open(fn) as f:
                d = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return cls(
            mtx=np.asarray(d["mtx"], dtype=np.float64),
            dist=np.asarray(d["dist"], dtype=np.float64),
            image_size=tuple(d["image_size"]),
            rms=float(d.get("rms", 0.0)),
            n_views=int(d.get("n_views", 0)),
        )


class CharucoCalibrator:
    """Accumulates board captures, then solves for intrinsics."""

    def __init__(self):
        self._board    = markers.make_charuco_board()
        self._detector = cv2.aruco.CharucoDetector(self._board)
        self._all_corners: List[np.ndarray] = []
        self._all_ids:     List[np.ndarray] = []
        self._image_size:  Optional[Tuple[int, int]] = None

    @property
    def n_views(self) -> int:
        return len(self._all_corners)

    def try_add_view(self, frame: np.ndarray) -> Tuple[bool, int]:
        """
        Attempt to use ``frame`` as a calibration view.
        Returns (accepted, n_corners_found).  A view is accepted only if enough
        ChArUco corners are interpolated to be useful.
        """
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _, _ = self._detector.detectBoard(gray)
        n = 0 if ch_ids is None else len(ch_ids)
        if ch_ids is None or n < max(config.MIN_CHARUCO_CORNERS, 8):
            return False, n
        self._all_corners.append(ch_corners)
        self._all_ids.append(ch_ids)
        self._image_size = (gray.shape[1], gray.shape[0])
        return True, n

    def reset(self):
        self._all_corners.clear()
        self._all_ids.clear()
        self._image_size = None

    def calibrate(self) -> CameraIntrinsics:
        if self.n_views < config.MIN_CALIB_VIEWS:
            raise ValueError(f"Need at least {config.MIN_CALIB_VIEWS} views "
                             f"(have {self.n_views}).")
        obj_all, img_all = [], []
        for corners, ids in zip(self._all_corners, self._all_ids):
            obj, img = self._board.matchImagePoints(corners, ids)
            if obj is None or len(obj) < 4:
                continue
            obj_all.append(obj.astype(np.float32))
            img_all.append(img.astype(np.float32))
        rms, mtx, dist, _, _ = cv2.calibrateCamera(
            obj_all, img_all, self._image_size, None, None)
        return CameraIntrinsics(mtx, dist, self._image_size, float(rms), self.n_views)
