"""
core/pose_fusion.py — Dual-camera probe pose fusion.

Each camera independently estimates the probe pose via solvePnP on the
ArUco cube.  PoseFusion converts both estimates into model space using
their respective CameraModelTransforms and computes a weighted average.

If only one camera detects the probe, that single estimate is used as-is.
If both cameras detect the probe, their estimates are averaged equally —
this reduces noise and improves robustness against partial occlusion or
marker-edge ambiguity on any one camera.

The result is a FusedPose: rod tip, rod base, and direction vector all
expressed in model space, ready for direct X-ray projection.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from core.tracker   import ProbeDetection
from core.transform import CameraModelTransform


@dataclass
class FusedPose:
    """
    Probe pose in model space after fusing estimates from both cameras.

    Attributes
    ----------
    tip_model   : (3,) K-wire tip position in model space [mm]
    base_model  : (3,) rod base position (at cube front face) [mm]
    dir_model   : (3,) unit direction vector from base toward tip
    n_cameras   : number of cameras that contributed (1 or 2)
    """
    tip_model:  np.ndarray
    base_model: np.ndarray
    dir_model:  np.ndarray
    n_cameras:  int

    @property
    def insertion_depth_mm(self) -> float:
        """Euclidean distance from rod base to tip in model space."""
        return float(np.linalg.norm(self.tip_model - self.base_model))

    @property
    def confidence_label(self) -> str:
        return "✓✓ dual camera" if self.n_cameras == 2 else "✓ single camera"


def fuse_poses(
    det1: Optional[ProbeDetection],
    xfm1: Optional[CameraModelTransform],
    det2: Optional[ProbeDetection],
    xfm2: Optional[CameraModelTransform],
) -> Optional[FusedPose]:
    """
    Fuse probe detections from camera 1 (cranial) and camera 2 (oblique).

    For each camera with a valid detection and a valid board transform,
    the rod tip and base are converted from camera space to model space.
    The resulting model-space estimates are then averaged.

    Returns None if no camera produced a valid detection.

    Parameters
    ----------
    det1 / det2 : ProbeDetection or None — result of ArucoTracker.detect()
    xfm1 / xfm2 : CameraModelTransform or None — result of PlatformBoardTracker.estimate_pose()
    """
    tips:  list[np.ndarray] = []
    bases: list[np.ndarray] = []

    if det1 is not None and xfm1 is not None:
        tips.append( xfm1.point_cam_to_model(det1.rod_tip_cam))
        bases.append(xfm1.point_cam_to_model(det1.rod_base_cam))

    if det2 is not None and xfm2 is not None:
        tips.append( xfm2.point_cam_to_model(det2.rod_tip_cam))
        bases.append(xfm2.point_cam_to_model(det2.rod_base_cam))

    if not tips:
        return None

    tip_model  = np.mean(tips,  axis=0)
    base_model = np.mean(bases, axis=0)

    direction  = tip_model - base_model
    norm       = np.linalg.norm(direction)
    dir_unit   = direction / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])

    return FusedPose(
        tip_model  = tip_model,
        base_model = base_model,
        dir_model  = dir_unit,
        n_cameras  = len(tips),
    )
