"""
core/markers.py — ArUco / ChArUco detection and pose estimation.

Two trackers
------------
BoardTracker : estimates the ChArUco board pose (board → camera) for each
               camera. The board is the fixed reference frame on the platform.

ProbeTracker : estimates the two-cube probe pose (probe → camera) by fusing
               all visible cube-face markers in a single solvePnP. Fusing
               every visible face from both cubes gives far stronger angular
               constraints than a single-face solve.

Coordinate frames (all in mm)
-----------------------------
    probe body frame  →  camera frame   (ProbeTracker.estimate)
    board frame       →  camera frame   (BoardTracker.estimate)

The probe pose can then be expressed in board frame via:
    T_probe_board = inv(T_board_cam) @ T_probe_cam
and from there into gVXR world via BOARD_TO_WORLD (in xray_sim / app).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import config as cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Rigid transform helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compose a 4×4 homogeneous transform from 3×3 R and 3-vector t."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = np.asarray(t, dtype=np.float64).flatten()
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a rigid-body 4×4 transform (R, t) → (Rᵀ, −Rᵀt)."""
    R = T[:3, :3]
    t = T[:3,  3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3,  3] = -(R.T @ t)
    return Ti


def transform_point(T: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Apply a 4×4 transform to a 3-vector, return a 3-vector."""
    ph = np.append(np.asarray(pt, dtype=np.float64).flatten(), 1.0)
    return (T @ ph)[:3]


# ─────────────────────────────────────────────────────────────────────────────
# Board (reference frame) tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoardPose:
    """ChArUco board pose in camera frame."""
    rvec: np.ndarray         # (3,1)
    tvec: np.ndarray         # (3,1)
    R:    np.ndarray         # (3,3)
    n_corners: int           # number of charuco corners used

    @property
    def T_board_cam(self) -> np.ndarray:
        return make_transform(self.R, self.tvec)


class BoardTracker:
    """Detects the ChArUco platform board and estimates its 6-DOF pose."""

    def __init__(self):
        self._board = cv2.aruco.CharucoBoard(
            (cfg.CHARUCO_COLS, cfg.CHARUCO_ROWS),
            cfg.CHARUCO_SQUARE_MM,
            cfg.CHARUCO_MARKER_MM,
            cfg.CHARUCO_BOARD_DICT,
        )
        params         = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(cfg.CHARUCO_BOARD_DICT, params)
        # CharucoDetector available in OpenCV ≥ 4.7
        self._charuco_detector = cv2.aruco.CharucoDetector(self._board)

    def estimate(
        self, frame: np.ndarray, cam_mtx: np.ndarray, dist: np.ndarray
    ) -> Optional[BoardPose]:
        """Return BoardPose in camera frame, or None if board not seen clearly."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _, _ = self._charuco_detector.detectBoard(gray)

        if ch_ids is None or len(ch_ids) < cfg.MIN_CHARUCO_CORNERS:
            return None

        obj_pts, img_pts = self._board.matchImagePoints(ch_corners, ch_ids)
        if obj_pts is None or len(obj_pts) < cfg.MIN_CHARUCO_CORNERS:
            return None

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None

        R, _ = cv2.Rodrigues(rvec)
        return BoardPose(rvec=rvec, tvec=tvec, R=R, n_corners=int(len(ch_ids)))

    def annotate(
        self, frame: np.ndarray, pose: Optional[BoardPose],
        cam_mtx: np.ndarray, dist: np.ndarray,
    ) -> np.ndarray:
        out = frame.copy()
        if pose is None:
            cv2.putText(out, "Board not detected", (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
            return out
        cv2.drawFrameAxes(out, cam_mtx, dist, pose.rvec, pose.tvec, 30.0)
        cv2.putText(out, f"Board: {pose.n_corners} corners", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 2)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Probe (two-cube) tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProbePose:
    """Two-cube probe pose in camera frame, plus derived K-wire geometry."""
    rvec:  np.ndarray
    tvec:  np.ndarray
    R:     np.ndarray
    n_faces: int                # number of marker faces fused
    rms_reproj: float           # reprojection RMS error (px)
    tip_cam:  np.ndarray        # K-wire tip in camera frame (mm)
    base_cam: np.ndarray        # K-wire base in camera frame (mm)

    @property
    def T_probe_cam(self) -> np.ndarray:
        return make_transform(self.R, self.tvec)


class ProbeTracker:
    """
    Detects two-cube ArUco markers (IDs 0–7) and fuses all visible faces
    into a single robust probe pose via multi-point solvePnP.
    """

    def __init__(self):
        params = cv2.aruco.DetectorParameters()
        # Relaxed thresholds for printed markers under variable lighting
        params.adaptiveThreshWinSizeMin  = 3
        params.adaptiveThreshWinSizeMax  = 23
        params.adaptiveThreshWinSizeStep = 4
        params.minMarkerPerimeterRate    = 0.02
        params.cornerRefinementMethod    = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(cfg.ARUCO_PROBE_DICT, params)

    def estimate(
        self, frame: np.ndarray, cam_mtx: np.ndarray, dist: np.ndarray
    ) -> Optional[ProbePose]:
        """Fuse all visible probe faces into one pose. None if none seen."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None:
            return None

        obj_all: list[np.ndarray] = []
        img_all: list[np.ndarray] = []
        n_faces = 0
        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)
            if mid not in cfg.PROBE_FACE_OBJ_PTS:
                continue
            obj_all.append(cfg.PROBE_FACE_OBJ_PTS[mid])
            img_all.append(corners[i][0].astype(np.float32))
            n_faces += 1

        if n_faces < cfg.MIN_PROBE_MARKERS:
            return None

        obj_pts = np.concatenate(obj_all, axis=0).reshape(-1, 3).astype(np.float32)
        img_pts = np.concatenate(img_all, axis=0).reshape(-1, 2).astype(np.float32)

        # Single-face → IPPE_SQUARE; multi-face → iterative refinement
        if n_faces == 1:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, cam_mtx, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, cam_mtx, dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        if not ok:
            return None

        # Refine with VVS for sub-pixel accuracy when multiple faces present
        if n_faces >= 2:
            rvec, tvec = cv2.solvePnPRefineVVS(
                obj_pts, img_pts, cam_mtx, dist, rvec, tvec
            )

        rms = self._reprojection_rms(obj_pts, img_pts, rvec, tvec, cam_mtx, dist)
        R, _ = cv2.Rodrigues(rvec)

        tip_cam  = (R @ cfg.ROD_TIP_IN_PROBE.reshape(3, 1)  + tvec).flatten()
        base_cam = (R @ cfg.ROD_BASE_IN_PROBE.reshape(3, 1) + tvec).flatten()

        return ProbePose(
            rvec=rvec, tvec=tvec, R=R,
            n_faces=n_faces, rms_reproj=rms,
            tip_cam=tip_cam, base_cam=base_cam,
        )

    @staticmethod
    def _reprojection_rms(obj_pts, img_pts, rvec, tvec, cam_mtx, dist) -> float:
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam_mtx, dist)
        proj = proj.reshape(-1, 2)
        err  = np.linalg.norm(proj - img_pts, axis=1)
        return float(np.sqrt(np.mean(err ** 2)))

    def annotate(
        self, frame: np.ndarray, pose: Optional[ProbePose],
        cam_mtx: np.ndarray, dist: np.ndarray,
    ) -> np.ndarray:
        out = frame.copy()
        # Always draw detected markers for visual feedback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(out, corners, ids)

        if pose is None:
            cv2.putText(out, "No probe", (16, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
            return out

        cv2.drawFrameAxes(out, cam_mtx, dist, pose.rvec, pose.tvec, 25.0)

        # Project K-wire shaft.
        # A degenerate/diverged PnP solve can yield non-finite rvec/tvec, which
        # makes projectPoints return NaN or +-inf. Casting those to int is
        # undefined (RuntimeWarning: invalid value encountered in cast) and
        # yields garbage pixel coordinates, so validate before drawing.
        pts3d = np.array([cfg.ROD_BASE_IN_PROBE, cfg.ROD_TIP_IN_PROBE], np.float32)
        proj, _ = cv2.projectPoints(pts3d, pose.rvec, pose.tvec, cam_mtx, dist)
        proj = proj.reshape(-1, 2)

        h, w = out.shape[:2]
        # Generous bound: allow off-screen but reject absurd magnitudes that
        # indicate a diverged solve rather than a probe merely out of view.
        limit = 10 * max(h, w)
        if np.all(np.isfinite(proj)) and np.all(np.abs(proj) < limit):
            proj = proj.astype(int)
            cv2.line(out, tuple(proj[0]), tuple(proj[1]),
                     cfg.OVERLAY_SHAFT_COLOR, 3, cv2.LINE_AA)
            cv2.circle(out, tuple(proj[1]), 7, cfg.OVERLAY_TIP_COLOR, -1, cv2.LINE_AA)
        else:
            cv2.putText(out, "Probe pose unstable", (16, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)

        col = (80, 220, 80) if pose.rms_reproj < 15 else (0, 140, 255)
        cv2.putText(out, f"Probe: {pose.n_faces} faces  RMS {pose.rms_reproj:.1f}px",
                    (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        return out
