"""
core/tracking.py  —  Live pose pipeline
=======================================

Three pieces, used identically by the Cameras screen (verification) and the
Simulation screen (overlay):

  BoardTracker  : ChArUco board in a frame  -> camera->model RigidTransform
  ProbeTracker  : ArUco cube in a frame      -> rod tip/base in CAMERA space
  fuse_model_points : combine two cameras' model-space estimates into one

Per-frame board tracking is what makes the rig forgiving: the webcams can be
moved between (or during) sessions and the camera->model transform is simply
recomputed.  Only the one-time lens intrinsics are persisted.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

import cv2
import numpy as np

import config
from core import markers
from core.geometry import RigidTransform


# ── Board ────────────────────────────────────────────────────────────────────
@dataclass
class BoardPose:
    transform: RigidTransform
    n_corners: int
    rvec: np.ndarray
    tvec: np.ndarray


class BoardTracker:
    def __init__(self):
        self._board    = markers.make_charuco_board()
        self._detector = cv2.aruco.CharucoDetector(self._board)

    @property
    def board(self):
        return self._board

    def estimate(self, frame: np.ndarray, mtx: np.ndarray, dist: np.ndarray) -> Optional[BoardPose]:
        gray = _as_gray(frame)
        ch_corners, ch_ids, _, _ = self._detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < config.MIN_CHARUCO_CORNERS:
            return None

        obj_pts, img_pts = self._board.matchImagePoints(ch_corners, ch_ids)
        if obj_pts is None or len(obj_pts) < config.MIN_CHARUCO_CORNERS:
            return None

        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        # solvePnP gives MODEL->CAMERA; from_model_to_camera inverts it for us.
        return BoardPose(RigidTransform.from_model_to_camera(rvec, tvec), int(len(ch_ids)),
                         np.asarray(rvec, float), np.asarray(tvec, float))


# ── Probe cube ─────────────────────────────────────────────────────────────—
@dataclass
class ProbeObservation:
    rod_tip_cam:  np.ndarray   # (3,) camera-space mm
    rod_base_cam: np.ndarray   # (3,) camera-space mm
    n_faces:      int
    primary_id:   int
    corners:      np.ndarray   # (4,2) image px of the primary face (for drawing)
    rvec:         np.ndarray
    tvec:         np.ndarray


class ProbeTracker:
    def __init__(self):
        self._detector = markers.make_aruco_detector()

    def estimate(self, frame: np.ndarray, mtx: np.ndarray, dist: np.ndarray) -> Optional[ProbeObservation]:
        gray = _as_gray(frame)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return None

        obj_list, img_list, faces = [], [], []
        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)
            if mid not in markers.CUBE_FACE_OBJ_PTS:
                continue
            c = corners[i][0].astype(np.float64)
            obj_list.append(markers.CUBE_FACE_OBJ_PTS[mid])
            img_list.append(c)
            faces.append((mid, c))
        if not faces:
            return None

        obj_pts = np.vstack(obj_list).astype(np.float64)
        img_pts = np.vstack(img_list).astype(np.float64)

        # SQPNP handles our (non-canonical) cube-face object points correctly for
        # any face count; IPPE_SQUARE assumes a marker centred at the origin and
        # is wrong for our geometry (~20 mm tip error vs ~0.3 mm).  We then refine
        # with an iterative solve.  Note: a *single* face is still geometrically
        # ambiguous (it can tilt toward or away), so two faces, or fusion across
        # the two cameras, is what makes the tip trustworthy.
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, dist,
                                      flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, dist,
                                          rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None

        R, _ = cv2.Rodrigues(rvec)
        tip_cam  = (R @ markers.ROD_TIP_IN_CUBE  + tvec.flatten())
        base_cam = (R @ markers.ROD_BASE_IN_CUBE + tvec.flatten())
        primary_id, primary_corners = max(faces, key=lambda f: cv2.contourArea(f[1].astype(np.float32)))
        return ProbeObservation(tip_cam, base_cam, len(faces), primary_id,
                                primary_corners, rvec, tvec)


# ── Fusion ───────────────────────────────────────────────────────────────────
@dataclass
class FusedProbe:
    tip_model:   np.ndarray
    base_model:  np.ndarray
    direction:   np.ndarray          # unit vector base->tip
    n_cameras:   int
    agreement_mm: Optional[float]    # spread between the two cameras' tip estimates


def fuse_model_points(tips: List[np.ndarray], bases: List[np.ndarray]) -> Optional[FusedProbe]:
    """Average per-camera model-space tip/base estimates.  1 or 2 cameras OK."""
    if not tips:
        return None
    tips  = [np.asarray(t, dtype=np.float64) for t in tips]
    bases = [np.asarray(b, dtype=np.float64) for b in bases]
    tip  = np.mean(tips,  axis=0)
    base = np.mean(bases, axis=0)
    d = tip - base
    n = np.linalg.norm(d)
    direction = d / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    agreement = float(np.linalg.norm(tips[0] - tips[1])) if len(tips) == 2 else None
    return FusedProbe(tip, base, direction, len(tips), agreement)


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    c = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _as_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# ── diagnostic drawing ───────────────────────────────────────────────────────
def draw_board_pose(frame, board: BoardPose, mtx, dist, axis_len=30.0):
    """Draw the board's model axes (X=red, Y=green, Z=blue) at its origin."""
    try:
        cv2.drawFrameAxes(frame, mtx, dist, board.rvec, board.tvec, axis_len, 2)
    except cv2.error:
        pass


def draw_probe_pose(frame, obs: ProbeObservation, mtx, dist, axis_len=18.0):
    """Draw the cube axes and the rod (base->tip) as the tracker sees them.
    If the red rod line + dot land on the physical K-wire and its tip in the
    image, the probe pose is correct.  If they point off in some other
    direction, the cube pose is wrong (gluing / single-face ambiguity)."""
    try:
        cv2.drawFrameAxes(frame, mtx, dist, obs.rvec, obs.tvec, axis_len, 2)
        pts = np.vstack([markers.ROD_BASE_IN_CUBE, markers.ROD_TIP_IN_CUBE]).astype(np.float64)
        proj, _ = cv2.projectPoints(pts, obs.rvec, obs.tvec, mtx, dist)
        p = proj.reshape(-1, 2)
        if np.all(np.isfinite(p)):
            a = tuple(np.round(p[0]).astype(int))
            b = tuple(np.round(p[1]).astype(int))
            cv2.line(frame, a, b, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, b, 6, (0, 0, 255), -1, cv2.LINE_AA)
    except cv2.error:
        pass
