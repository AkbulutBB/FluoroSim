"""
core/projection.py — Projects probe pose from model space into X-ray pixel space.

The X-ray is modelled as a pinhole camera (the C-arm IS a camera).
The 3×4 projection matrix P is computed once during the OR visit from
N ≥ 6 radiopaque fiducial correspondences:
    3-D model coordinate ↔ 2-D pixel in the stored X-ray image.

At runtime, any 3-D point in model space is projected via:
    [u·w, v·w, w]ᵀ  =  P  ×  [X, Y, Z, 1]ᵀ
    (u, v) = pixel in X-ray image
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from core.calibration import CameraModelTransform
from core.tracker import ProbeDetection
from config import (
    ROD_TIP_IN_CUBE, ROD_BASE_IN_CUBE,
    OVERLAY_COLOR_TIP, OVERLAY_COLOR_SHAFT,
    OVERLAY_THICKNESS, OVERLAY_TIP_RADIUS, OVERLAY_SHAFT_EXTEND,
)


# ─── DLT — Direct Linear Transform ────────────────────────────────────────────

def compute_projection_matrix(
    obj_pts: np.ndarray,   # (N, 3) 3-D model coordinates
    img_pts: np.ndarray,   # (N, 2) pixel positions in X-ray image
) -> np.ndarray:           # (3, 4) projection matrix P
    """
    Compute the 3×4 projective matrix via the DLT algorithm.
    Requires N ≥ 6 non-degenerate correspondences.
    """
    N = len(obj_pts)
    if N < 6:
        raise ValueError(f"Need at least 6 correspondences, got {N}.")

    A = []
    for (X, Y, Z), (u, v) in zip(obj_pts, img_pts):
        A.append([-X, -Y, -Z, -1,  0,  0,  0,  0, u*X, u*Y, u*Z, u])
        A.append([ 0,  0,  0,  0, -X, -Y, -Z, -1, v*X, v*Y, v*Z, v])

    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    # Normalise so that the last row has unit norm
    P /= np.linalg.norm(P[2, :3])
    return P


def project_point(P: np.ndarray, pt_model: np.ndarray) -> np.ndarray:
    """Project a single 3-D model point to 2-D pixel via matrix P."""
    ph  = np.append(pt_model.flatten(), 1.0)
    uvw = P @ ph
    return uvw[:2] / uvw[2]


def reprojection_error(
    P: np.ndarray,
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
) -> float:
    """Mean reprojection error in pixels for validation."""
    errors = []
    for pt3d, pt2d in zip(obj_pts, img_pts):
        proj = project_point(P, pt3d)
        errors.append(np.linalg.norm(proj - pt2d))
    return float(np.mean(errors))


# ─── Overlay renderer ─────────────────────────────────────────────────────────

class XRayOverlay:
    """
    Renders the probe trajectory onto a stored X-ray image.

    Accepts a probe detection in camera space, the camera-to-model transform,
    and the X-ray projection matrix.  Returns an annotated copy of the X-ray.
    """

    def __init__(self, xray_image: np.ndarray, P: np.ndarray):
        self._xray = xray_image.copy()
        self._P    = P

    @property
    def blank(self) -> np.ndarray:
        """Return the unmodified X-ray."""
        return self._xray.copy()

    def render(
        self,
        detection: Optional[ProbeDetection],
        cam_model_xfm: CameraModelTransform,
        extend_mm: float = OVERLAY_SHAFT_EXTEND,
    ) -> np.ndarray:
        """
        Return the X-ray image with probe overlay, or the plain X-ray if
        no detection is provided.
        """
        out = self._xray.copy()

        if detection is None:
            cv2.putText(
                out, "Probe not detected", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 2,
            )
            return out

        # 1. Transform rod geometry from camera space to model space
        tip_model  = cam_model_xfm.point_cam_to_model(detection.rod_tip_cam)
        base_model = cam_model_xfm.point_cam_to_model(detection.rod_base_cam)

        # 2. Extend shaft line in model space for visual clarity
        direction  = tip_model - base_model
        norm       = np.linalg.norm(direction)
        if norm < 1e-6:
            return out
        unit = direction / norm
        shaft_end = tip_model + unit * extend_mm

        # 3. Project all points to X-ray pixel space
        px_tip  = project_point(self._P, tip_model).astype(int)
        px_base = project_point(self._P, base_model).astype(int)
        px_end  = project_point(self._P, shaft_end).astype(int)

        # 4. Draw: shaft from base through tip and onward, dot at tip
        cv2.line(out, tuple(px_base), tuple(px_end), OVERLAY_COLOR_SHAFT, OVERLAY_THICKNESS)
        cv2.circle(out, tuple(px_tip), OVERLAY_TIP_RADIUS, OVERLAY_COLOR_TIP, -1)
        cv2.circle(out, tuple(px_tip), OVERLAY_TIP_RADIUS + 3, OVERLAY_COLOR_TIP, 2)

        # Depth annotation (distance from base to tip in mm)
        depth = float(np.linalg.norm(tip_model - base_model))
        cv2.putText(
            out, f"Depth: {depth:.0f} mm",
            (20, out.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, OVERLAY_COLOR_TIP, 2,
        )
        return out


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_projection_matrix(model_id: str, view: str, P: np.ndarray, models_dir: str):
    path = Path(models_dir) / model_id
    path.mkdir(parents=True, exist_ok=True)
    np.save(str(path / f"P_{view}.npy"), P)


def load_projection_matrix(model_id: str, view: str, models_dir: str) -> Optional[np.ndarray]:
    p = Path(models_dir) / model_id / f"P_{view}.npy"
    if not p.exists():
        return None
    return np.load(str(p))
