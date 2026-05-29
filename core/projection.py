"""
core/projection.py — X-ray projection matrix and overlay rendering.

The C-arm fluoroscope is modelled as a pinhole camera.  During the one-time
OR visit, N ≥ 6 radiopaque fiducials with known 3-D model-space coordinates
are clicked in the stored X-ray images.  The Direct Linear Transform (DLT)
computes a 3×4 projection matrix P from these correspondences.

At runtime, the fused probe pose (model space) is projected through P to
obtain pixel positions on the stored X-ray, and the trajectory is drawn.

No camera hardware is involved at runtime — only the pre-acquired X-ray
images and the pre-computed projection matrices are used.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from core.pose_fusion import FusedPose
from config import (
    OVERLAY_COLOR_TIP, OVERLAY_COLOR_SHAFT,
    OVERLAY_THICKNESS, OVERLAY_TIP_RADIUS, OVERLAY_SHAFT_EXTEND,
    MODELS_DIR,
)


# ── Direct Linear Transform ────────────────────────────────────────────────────

def compute_projection_matrix(
    obj_pts: np.ndarray,   # (N, 3)  3-D model coordinates
    img_pts: np.ndarray,   # (N, 2)  pixel positions in X-ray image
) -> np.ndarray:           # (3, 4)  projection matrix P
    """
    Compute a 3×4 projective matrix P via the DLT algorithm.
    Requires N ≥ 6 non-degenerate, non-coplanar correspondences.
    """
    N = len(obj_pts)
    if N < 6:
        raise ValueError(f"DLT requires at least 6 correspondences (got {N}).")

    A = []
    for (X, Y, Z), (u, v) in zip(obj_pts, img_pts):
        A.append([-X, -Y, -Z, -1,  0,  0,  0,  0, u*X, u*Y, u*Z,  u])
        A.append([ 0,  0,  0,  0, -X, -Y, -Z, -1, v*X, v*Y, v*Z,  v])

    A  = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    P  = Vt[-1].reshape(3, 4)
    P /= np.linalg.norm(P[2, :3])   # normalise so homogeneous scale is consistent
    return P


def project_point(P: np.ndarray, pt_model: np.ndarray) -> np.ndarray:
    """Project a single 3-D model-space point to a 2-D pixel via P."""
    ph  = np.append(pt_model.flatten(), 1.0)
    uvw = P @ ph
    return uvw[:2] / uvw[2]


def reprojection_error(
    P:       np.ndarray,
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
) -> float:
    """Mean reprojection error in pixels — used to validate OR setup quality."""
    errors = [np.linalg.norm(project_point(P, p3) - p2)
              for p3, p2 in zip(obj_pts, img_pts)]
    return float(np.mean(errors))


# ── X-ray overlay renderer ─────────────────────────────────────────────────────

class XRayOverlay:
    """
    Renders the fused probe trajectory onto a stored X-ray image.

    Accepts a FusedPose (model space) and the pre-computed projection matrix
    for this X-ray view.  Returns an annotated copy of the stored X-ray.
    The live camera feed is never shown — only the X-ray with overlay.
    """

    def __init__(self, xray_image: np.ndarray, P: np.ndarray):
        self._xray = xray_image.copy()
        self._P    = P

    @property
    def blank(self) -> np.ndarray:
        """Return the unmodified X-ray (no overlay)."""
        return self._xray.copy()

    def render(self, fused: Optional[FusedPose]) -> np.ndarray:
        """
        Return the X-ray with the probe trajectory overlay drawn on it.

        If fused is None, returns the plain X-ray with a 'not detected' notice.
        """
        out = self._xray.copy()

        if fused is None:
            cv2.putText(
                out, "Probe not detected", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2,
            )
            return out

        # Extend the shaft line past the tip for visual clarity
        shaft_end = fused.tip_model + fused.dir_model * OVERLAY_SHAFT_EXTEND

        # Project model-space points to X-ray pixel positions
        px_base = project_point(self._P, fused.base_model).astype(int)
        px_tip  = project_point(self._P, fused.tip_model ).astype(int)
        px_end  = project_point(self._P, shaft_end       ).astype(int)

        # Shaft line: from rod base, through tip, and slightly beyond
        cv2.line(out, tuple(px_base), tuple(px_end), OVERLAY_COLOR_SHAFT, OVERLAY_THICKNESS)

        # Tip marker: filled circle with ring
        cv2.circle(out, tuple(px_tip), OVERLAY_TIP_RADIUS,     OVERLAY_COLOR_TIP, -1)
        cv2.circle(out, tuple(px_tip), OVERLAY_TIP_RADIUS + 3, OVERLAY_COLOR_TIP,  2)

        # Status annotation at bottom of image
        depth_text = (
            f"{fused.confidence_label}   "
            f"depth: {fused.insertion_depth_mm:.0f} mm"
        )
        cv2.putText(
            out, depth_text,
            (20, out.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, OVERLAY_COLOR_TIP, 2,
        )
        return out


# ── Persistence ────────────────────────────────────────────────────────────────

def save_projection_matrix(model_id: str, view: str, P: np.ndarray):
    path = Path(MODELS_DIR) / model_id
    path.mkdir(parents=True, exist_ok=True)
    np.save(str(path / f"P_{view}.npy"), P)


def load_projection_matrix(model_id: str, view: str) -> Optional[np.ndarray]:
    p = Path(MODELS_DIR) / model_id / f"P_{view}.npy"
    if not p.exists():
        return None
    return np.load(str(p))
