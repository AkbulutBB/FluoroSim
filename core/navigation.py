"""
core/navigation.py — Pose fusion: cameras → board → gVXR world.

NavigationEngine takes the per-camera board and probe detections and produces
a single probe tip + base position in gVXR world (CAD platform) space, ready
to feed XRaySimulator.overlay_probe().

Pipeline per camera
-------------------
    T_probe_cam   (from ProbeTracker)
    T_board_cam   (from BoardTracker)
    T_probe_board = inv(T_board_cam) @ T_probe_cam
    → tip_board, base_board

Then board → world via BOARD_TO_WORLD (config / xray_sim).

Two-camera fusion
-----------------
Each camera that sees BOTH board and probe yields an independent estimate of
tip/base in board frame. We weight by probe reprojection quality (lower RMS =
higher weight) and average. A median-of-history buffer then suppresses
single-frame pose flips while staying responsive to real motion.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config as cfg
from core.markers import (
    BoardPose, ProbePose,
    invert_transform, transform_point,
)

logger = logging.getLogger(__name__)


@dataclass
class FusedPose:
    """Probe tip + base in gVXR world (CAD platform) frame, mm."""
    tip_world:  np.ndarray
    base_world: np.ndarray
    n_cameras:  int            # how many cameras contributed
    mean_rms:   float          # mean probe reprojection RMS (px)
    valid:      bool


class NavigationEngine:
    """Fuses dual-camera detections into a single world-space probe pose."""

    def __init__(self, board_to_world: np.ndarray, history: int = 5):
        self._board_to_world = np.asarray(board_to_world, dtype=np.float64).reshape(4, 4)
        self._tip_hist:  deque = deque(maxlen=history)
        self._base_hist: deque = deque(maxlen=history)

    # ── Per-camera board-frame estimate ──────────────────────────────────

    @staticmethod
    def _probe_in_board(
        board: BoardPose, probe: ProbePose
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (tip_board, base_board) for one camera."""
        T_cam_board   = invert_transform(board.T_board_cam)
        # Tip / base are already in camera frame inside ProbePose
        tip_board  = transform_point(T_cam_board, probe.tip_cam)
        base_board = transform_point(T_cam_board, probe.base_cam)
        return tip_board, base_board

    # ── Fusion ───────────────────────────────────────────────────────────

    def fuse(
        self,
        board_a: Optional[BoardPose], probe_a: Optional[ProbePose],
        board_b: Optional[BoardPose], probe_b: Optional[ProbePose],
    ) -> FusedPose:
        """
        Combine detections from both cameras into a world-space pose.
        A camera contributes only if it sees BOTH the board and the probe.
        """
        tips, bases, weights, rmss = [], [], [], []

        for board, probe in ((board_a, probe_a), (board_b, probe_b)):
            if board is None or probe is None:
                continue
            tip_b, base_b = self._probe_in_board(board, probe)
            # Weight: inverse reprojection error (clamped)
            w = 1.0 / max(probe.rms_reproj, 1.0)
            tips.append(tip_b)
            bases.append(base_b)
            weights.append(w)
            rmss.append(probe.rms_reproj)

        if not tips:
            return FusedPose(
                tip_world=np.zeros(3), base_world=np.zeros(3),
                n_cameras=0, mean_rms=float("nan"), valid=False,
            )

        weights = np.array(weights)
        weights /= weights.sum()
        tip_board  = np.average(np.vstack(tips),  axis=0, weights=weights)
        base_board = np.average(np.vstack(bases), axis=0, weights=weights)

        # Median-history smoothing (robust to single-frame flips)
        self._tip_hist.append(tip_board)
        self._base_hist.append(base_board)
        tip_smooth  = np.median(np.vstack(self._tip_hist),  axis=0)
        base_smooth = np.median(np.vstack(self._base_hist), axis=0)

        # Board → world
        tip_world  = transform_point(self._board_to_world, tip_smooth)
        base_world = transform_point(self._board_to_world, base_smooth)

        return FusedPose(
            tip_world=tip_world, base_world=base_world,
            n_cameras=len(tips), mean_rms=float(np.mean(rmss)), valid=True,
        )

    def reset_history(self):
        self._tip_hist.clear()
        self._base_hist.clear()
