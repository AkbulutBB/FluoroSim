"""
core/model_store.py  —  X-ray model registration
=================================================

A "model registration" ties one physical spine model to two stored fluoroscopy
images (AP + lateral) through a shared set of steel-bearing fiducials:

  * fiducials_model : the known 3-D coordinates of the bearings (model space)
  * per view (ap/lat): the X-ray image, the clicked pixel of each bearing, and
    the resulting DLT projection matrix P
  * calib_hole_model: the known coordinate of the verification hole

Once both views have a P matrix, the registration is complete and the live
probe tip can be projected onto both X-rays.

Saved as JSON under data/models/<name>.json; the X-ray images are copied next
to it so a model folder is self-contained.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os
import shutil

import numpy as np

import config
from core import paths
from core.dlt import ProjectionMatrix


@dataclass
class ViewData:
    image_path: str = ""                 # absolute path to the X-ray image
    clicks:     List[List[float]] = field(default_factory=list)   # (N,2) pixel
    P:          Optional[ProjectionMatrix] = None
    reproj_px:  Optional[float] = None


@dataclass
class ModelRegistration:
    name:            str = "untitled"
    fiducials_model: List[List[float]] = field(default_factory=list)   # (N,3) mm
    ap:              ViewData = field(default_factory=ViewData)
    lat:             ViewData = field(default_factory=ViewData)
    calib_hole_model: List[float] = field(default_factory=lambda: config.CALIB_HOLE_MODEL_MM.tolist())

    # ---- helpers -----------------------------------------------------------
    def view(self, role: str) -> ViewData:
        return self.ap if role == "ap" else self.lat

    @property
    def n_fiducials(self) -> int:
        return len(self.fiducials_model)

    def compute_view(self, role: str) -> float:
        """Fit the DLT for one view from its clicks; returns reprojection error (px)."""
        v = self.view(role)
        obj = np.asarray(self.fiducials_model, dtype=np.float64)
        img = np.asarray(v.clicks, dtype=np.float64)
        if len(obj) != len(img):
            raise ValueError("Number of fiducials and clicks must match.")
        v.P = ProjectionMatrix.from_correspondences(obj, img)
        v.reproj_px = v.P.reprojection_error(obj, img)
        return v.reproj_px

    @property
    def is_complete(self) -> bool:
        return self.ap.P is not None and self.lat.P is not None

    # ---- persistence -------------------------------------------------------
    def save(self) -> str:
        paths.ensure_dirs()
        fn = paths.model_file(self.name)
        base = os.path.splitext(os.path.basename(fn))[0]

        def stash_image(v: ViewData, role: str) -> str:
            if v.image_path and os.path.isfile(v.image_path):
                ext = os.path.splitext(v.image_path)[1] or ".png"
                dest = os.path.join(paths.MODEL_DIR, f"{base}_{role}{ext}")
                if os.path.abspath(v.image_path) != os.path.abspath(dest):
                    shutil.copyfile(v.image_path, dest)
                return os.path.basename(dest)
            return ""

        def view_dict(v: ViewData, role: str) -> Dict:
            return {
                "image": stash_image(v, role),
                "clicks": [list(map(float, c)) for c in v.clicks],
                "P": v.P.tolist() if v.P is not None else None,
                "reproj_px": v.reproj_px,
            }

        with open(fn, "w") as f:
            json.dump({
                "name": self.name,
                "fiducials_model": [list(map(float, p)) for p in self.fiducials_model],
                "calib_hole_model": list(map(float, self.calib_hole_model)),
                "ap":  view_dict(self.ap, "ap"),
                "lat": view_dict(self.lat, "lat"),
            }, f, indent=2)
        return fn

    @classmethod
    def load(cls, json_path: str) -> "ModelRegistration":
        with open(json_path) as f:
            d = json.load(f)

        def load_view(vd: Dict) -> ViewData:
            img = vd.get("image", "")
            img_abs = os.path.join(paths.MODEL_DIR, img) if img else ""
            P = ProjectionMatrix.fromlist(vd["P"]) if vd.get("P") else None
            return ViewData(image_path=img_abs,
                            clicks=[list(map(float, c)) for c in vd.get("clicks", [])],
                            P=P, reproj_px=vd.get("reproj_px"))

        return cls(
            name=d.get("name", "untitled"),
            fiducials_model=[list(map(float, p)) for p in d.get("fiducials_model", [])],
            ap=load_view(d.get("ap", {})),
            lat=load_view(d.get("lat", {})),
            calib_hole_model=list(map(float, d.get("calib_hole_model",
                                                   config.CALIB_HOLE_MODEL_MM.tolist()))),
        )
