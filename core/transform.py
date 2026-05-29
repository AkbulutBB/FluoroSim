"""
core/transform.py — Rigid-body transform utilities.

CameraModelTransform encapsulates the 4×4 homogeneous transform that maps
a point from camera space into model/board space.  It is produced by the
PlatformBoardTracker on every frame and consumed by PoseFusion and XRayOverlay.
"""

import numpy as np


def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4×4 homogeneous transform from a 3×3 rotation and 3-vector translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = t.flatten()
    return T


def _invert_T(T: np.ndarray) -> np.ndarray:
    """Invert a rigid-body 4×4 transform without a general matrix inversion."""
    R = T[:3, :3]
    t = T[:3,  3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3,  3] = -(R.T @ t)
    return T_inv


def apply_T(T: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Apply a 4×4 rigid transform to a 3-vector."""
    ph = np.append(pt.flatten(), 1.0)
    return (T @ ph)[:3]


class CameraModelTransform:
    """
    Rigid transform from camera space to model space for one camera.

    Produced each frame by PlatformBoardTracker and passed to PoseFusion.
    T maps:  pt_model = T @ pt_cam  (homogeneous coordinates).
    """

    def __init__(self, R_cam_to_model: np.ndarray, t_cam_to_model: np.ndarray):
        self._T = _make_T(
            R_cam_to_model.astype(np.float64),
            t_cam_to_model.astype(np.float64),
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def point_cam_to_model(self, pt_cam: np.ndarray) -> np.ndarray:
        """Transform a single 3-D point from camera space to model space."""
        return apply_T(self._T, pt_cam)

    @property
    def T(self) -> np.ndarray:
        """The raw 4×4 transform matrix (copy)."""
        return self._T.copy()
