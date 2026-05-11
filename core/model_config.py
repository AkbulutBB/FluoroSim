"""
core/model_config.py — Loads and validates a spine model package.

A model package is a directory under data/models/<id>/ containing:
  model_config.json  — slots, metadata
  xray_ap.png        — AP fluoroscopy image
  xray_lat.png       — LAT fluoroscopy image
  P_ap.npy           — AP projection matrix (optional; set up via OR mode)
  P_lat.npy          — LAT projection matrix (optional)
  fiducials.json     — 3D↔pixel correspondences used to build P matrices
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from core.calibration import SlotDefinition
from config import MODELS_DIR


@dataclass
class ModelPackage:
    model_id    : str
    name        : str
    directory   : Path
    slots       : list[SlotDefinition]
    xray_ap     : Optional[np.ndarray]  = field(default=None, repr=False)
    xray_lat    : Optional[np.ndarray]  = field(default=None, repr=False)
    P_ap        : Optional[np.ndarray]  = field(default=None, repr=False)
    P_lat       : Optional[np.ndarray]  = field(default=None, repr=False)

    @property
    def has_projection(self) -> bool:
        return self.P_ap is not None and self.P_lat is not None

    @property
    def has_xrays(self) -> bool:
        return self.xray_ap is not None and self.xray_lat is not None


def load_model(model_id: str) -> ModelPackage:
    """Load a model package from disk. Raises FileNotFoundError if missing."""
    model_dir = Path(MODELS_DIR) / model_id

    cfg_path = model_dir / "model_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {cfg_path}")

    with open(cfg_path) as f:
        cfg = json.load(f)

    slots = [SlotDefinition.from_dict(s) for s in cfg["slots"]]

    def _load_img(fname: str) -> Optional[np.ndarray]:
        p = model_dir / fname
        return cv2.imread(str(p)) if p.exists() else None

    def _load_npy(fname: str) -> Optional[np.ndarray]:
        p = model_dir / fname
        return np.load(str(p)) if p.exists() else None

    return ModelPackage(
        model_id  = model_id,
        name      = cfg.get("name", model_id),
        directory = model_dir,
        slots     = slots,
        xray_ap   = _load_img("xray_ap.png"),
        xray_lat  = _load_img("xray_lat.png"),
        P_ap      = _load_npy("P_ap.npy"),
        P_lat     = _load_npy("P_lat.npy"),
    )


def list_models() -> list[str]:
    """Return IDs of all model packages present on disk."""
    base = Path(MODELS_DIR)
    if not base.exists():
        return []
    return [
        d.name for d in sorted(base.iterdir())
        if d.is_dir() and (d / "model_config.json").exists()
    ]


def save_fiducials(model_id: str, obj_pts: np.ndarray, img_pts_ap: np.ndarray, img_pts_lat: np.ndarray):
    """Persist OR-visit fiducial correspondences to disk."""
    path = Path(MODELS_DIR) / model_id
    path.mkdir(parents=True, exist_ok=True)
    data = {
        "obj_pts":     obj_pts.tolist(),
        "img_pts_ap":  img_pts_ap.tolist(),
        "img_pts_lat": img_pts_lat.tolist(),
    }
    with open(path / "fiducials.json", "w") as f:
        json.dump(data, f, indent=2)


def load_fiducials(model_id: str) -> Optional[dict]:
    p = Path(MODELS_DIR) / model_id / "fiducials.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return {
        "obj_pts":     np.array(d["obj_pts"],     np.float64),
        "img_pts_ap":  np.array(d["img_pts_ap"],  np.float64),
        "img_pts_lat": np.array(d["img_pts_lat"], np.float64),
    }
