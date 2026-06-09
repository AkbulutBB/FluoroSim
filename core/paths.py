"""
core/paths.py  —  Where FluoroSim keeps its files
==================================================

Everything the app saves (camera lens calibrations, model registrations,
verification settings) lives under a single top-level ``data/`` folder next to
the code, so it is easy to find, back up, or wipe.  Folders are created on
demand.

    data/
      cameras/        camera_ap.json, camera_lat.json     (lens intrinsics)
      models/         <model-name>.json                    (X-ray registrations)
      verification.json                                    (calibration-hole settings)
"""

from __future__ import annotations
import os

# Project root = the folder that contains this package's parent (…/FluoroSim/).
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(_THIS_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, "data")
CAMERA_DIR = os.path.join(DATA_DIR, "cameras")
MODEL_DIR  = os.path.join(DATA_DIR, "models")
VERIFICATION_FILE = os.path.join(DATA_DIR, "verification.json")


def ensure_dirs() -> None:
    for d in (DATA_DIR, CAMERA_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)


def camera_file(role: str) -> str:
    return os.path.join(CAMERA_DIR, f"camera_{role}.json")


def model_file(name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip().replace(" ", "_")
    return os.path.join(MODEL_DIR, f"{safe}.json")


def list_models():
    ensure_dirs()
    out = []
    for fn in sorted(os.listdir(MODEL_DIR)):
        if fn.endswith(".json"):
            out.append(os.path.join(MODEL_DIR, fn))
    return out
