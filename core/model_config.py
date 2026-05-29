"""
core/model_config.py — Spine model package loading and persistence.

A ModelPackage bundles everything associated with a particular 3D-printed
spine model: the stored X-ray images, the projection matrices computed
during the OR visit, and descriptive metadata.

There are no calibration slots in this version — camera-to-model
registration is handled entirely by the platform CharucoBoard.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from config import MODELS_DIR
from core.projection import load_projection_matrix


@dataclass
class ModelPackage:
    """All data associated with one 3D-printed spine model."""

    model_id:    str
    name:        str
    description: str

    # X-ray images (loaded from PNG)
    xray_ap:  Optional[np.ndarray] = field(default=None, repr=False)
    xray_lat: Optional[np.ndarray] = field(default=None, repr=False)

    # 3×4 projection matrices (computed during OR setup)
    P_ap:  Optional[np.ndarray] = field(default=None, repr=False)
    P_lat: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def has_xrays(self) -> bool:
        return self.xray_ap is not None and self.xray_lat is not None

    @property
    def has_projection(self) -> bool:
        return self.P_ap is not None and self.P_lat is not None

    @property
    def is_ready(self) -> bool:
        """True when the model is fully configured for training."""
        return self.has_xrays and self.has_projection


# ── Load / save ────────────────────────────────────────────────────────────────

def load_model(model_id: str) -> ModelPackage:
    """
    Load a model package from disk.

    Raises FileNotFoundError if the model directory or config JSON is absent.
    X-rays and projection matrices are loaded if present; absence is not an error
    (the user may not have completed OR Setup yet).
    """
    model_dir = Path(MODELS_DIR) / model_id
    cfg_path  = model_dir / "model_config.json"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    pkg = ModelPackage(
        model_id    = model_id,
        name        = cfg.get("name",        model_id),
        description = cfg.get("description", ""),
    )

    # X-ray images
    for view in ("ap", "lat"):
        p = model_dir / f"xray_{view}.png"
        if p.exists():
            img = cv2.imread(str(p))
            setattr(pkg, f"xray_{view}", img)

    # Projection matrices
    pkg.P_ap  = load_projection_matrix(model_id, "ap")
    pkg.P_lat = load_projection_matrix(model_id, "lat")

    return pkg


def list_models() -> list[str]:
    """Return a list of model IDs found in the models directory."""
    base = Path(MODELS_DIR)
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "model_config.json").exists()
    )


def create_model(model_id: str, name: str, description: str = "") -> ModelPackage:
    """Create a new model package directory with a minimal config JSON."""
    model_dir = Path(MODELS_DIR) / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    cfg = {"name": name, "description": description}
    with open(model_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    return ModelPackage(model_id=model_id, name=name, description=description)


def save_xray(model_id: str, view: str, image: np.ndarray):
    """Save an X-ray image (AP or LAT) for the given model."""
    path = Path(MODELS_DIR) / model_id / f"xray_{view}.png"
    cv2.imwrite(str(path), image)


def save_fiducials(model_id: str, obj_pts: np.ndarray, img_pts_ap: np.ndarray, img_pts_lat: np.ndarray):
    """Persist fiducial correspondence points for later inspection or recomputation."""
    path = Path(MODELS_DIR) / model_id / "fiducials.npz"
    np.savez(str(path), obj_pts=obj_pts, img_pts_ap=img_pts_ap, img_pts_lat=img_pts_lat)


def load_fiducials(model_id: str) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    p = Path(MODELS_DIR) / model_id / "fiducials.npz"
    if not p.exists():
        return None
    data = np.load(str(p))
    return data["obj_pts"], data["img_pts_ap"], data["img_pts_lat"]
