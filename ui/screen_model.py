"""
ui/screen_model.py  —  Model registration
==========================================

Tie the spine model to its AP and lateral X-rays through the steel-bearing
fiducials:

  1. Enter the known 3-D model coordinates of the bearings (one "X Y Z" per
     line — you can paste these straight from a spreadsheet).
  2. Load the AP X-ray and click the bearings in the SAME order.
  3. Load the lateral X-ray and click them in the SAME order.
  4. Compute & save — this fits a DLT projection matrix for each view and
     reports the reprojection error (your registration quality gauge).

Clicking order must match the coordinate order; right-click removes the last
point.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np

import config
from core import paths
from core.model_store import ModelRegistration
from ui.app import Screen
from ui import widgets as W


class ModelScreen(Screen):
    def __init__(self, master, app):
        super().__init__(master, style="App.TFrame")
        self.app = app
        self.session = app.session
        self._images = {"ap": None, "lat": None}     # loaded BGR arrays
        self._image_paths = {"ap": "", "lat": ""}

        self._build_header()

        body = ttk.Frame(self, style="App.TFrame")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # left column: name + fiducials + actions
        left = ttk.Frame(body, style="App.TFrame")
        left.pack(side="left", fill="y", padx=(0, 14))

        ttk.Label(left, text="Model name", style="Muted.TLabel").pack(anchor="w")
        self.name_var = tk.StringVar(value="model_01")
        ttk.Entry(left, textvariable=self.name_var, width=26).pack(anchor="w", pady=(0, 12))

        ttk.Label(left, text="Fiducial coordinates  (X Y Z per line, mm)",
                  style="Muted.TLabel").pack(anchor="w")
        self.fid_text = tk.Text(left, width=26, height=12, bg=W.PANEL, fg=W.FG,
                                insertbackground=W.FG, relief="flat")
        self.fid_text.pack(anchor="w", pady=(2, 4))
        self.fid_count = ttk.Label(left, text="0 fiducials", style="Muted.TLabel")
        self.fid_count.pack(anchor="w")
        self.fid_text.bind("<KeyRelease>", lambda e: self._update_counts())

        ttk.Separator(left).pack(fill="x", pady=12)
        ttk.Button(left, text="Compute & save", style="Accent.TButton",
                   command=self._compute_save).pack(anchor="w", fill="x")
        ttk.Button(left, text="Load existing model", command=self._load_model).pack(anchor="w", fill="x", pady=6)
        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Button(left, text="Check registration", command=self._check_registration).pack(anchor="w", fill="x")
        ttk.Button(left, text="Clear check", command=self._clear_check).pack(anchor="w", fill="x", pady=6)
        ttk.Label(left, text="blue \u25cb = your clicks   red \u00d7 = model prediction.\n"
                             "Good registration: every \u00d7 sits on its \u25cb.",
                  style="Muted.TLabel", wraplength=240, justify="left").pack(anchor="w")
        self.status = ttk.Label(left, text="", style="Muted.TLabel", wraplength=240, justify="left")
        self.status.pack(anchor="w", pady=10)

        # right column: AP / LAT image tabs
        self.nb = ttk.Notebook(body)
        self.nb.pack(side="left", fill="both", expand=True)
        self.canvas = {}
        self.click_lbl = {}
        for r in config.CAMERA_ROLES:
            tab = ttk.Frame(self.nb, style="App.TFrame")
            self.nb.add(tab, text=f"  {config.XRAY_LABEL[r]} X-ray  ")
            top = ttk.Frame(tab, style="App.TFrame")
            top.pack(fill="x", pady=6)
            ttk.Button(top, text="Load X-ray", command=lambda rr=r: self._load_image(rr)).pack(side="left")
            ttk.Button(top, text="Clear points", command=lambda rr=r: self._clear(rr)).pack(side="left", padx=6)
            lbl = ttk.Label(top, text="0 points", style="Muted.TLabel")
            lbl.pack(side="left", padx=6)
            self.click_lbl[r] = lbl
            cv = W.ClickableImage(tab, max_w=620, max_h=560,
                                  on_change=lambda rr=r: self._update_counts())
            cv.pack(fill="both", expand=True)
            self.canvas[r] = cv

    def _build_header(self):
        bar = ttk.Frame(self, style="App.TFrame")
        bar.pack(fill="x", padx=16, pady=12)
        ttk.Button(bar, text="\u2190 Home", command=lambda: self.app.show("home")).pack(side="left")
        ttk.Label(bar, text="Model registration", style="H2.TLabel").pack(side="left", padx=14)

    # ── data helpers ──────────────────────────────────────────────────────────
    def _parse_fiducials(self) -> np.ndarray:
        rows = []
        for line in self.fid_text.get("1.0", "end").strip().splitlines():
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.asarray(rows, dtype=float)

    def _update_counts(self):
        try:
            n_fid = len(self._parse_fiducials())
        except ValueError:
            n_fid = -1
        self.fid_count.configure(text=(f"{n_fid} fiducials" if n_fid >= 0
                                       else "invalid coordinate line"))
        for r in config.CAMERA_ROLES:
            self.click_lbl[r].configure(text=f"{len(self.canvas[r].points)} points")

    def _load_image(self, role: str):
        path = filedialog.askopenfilename(
            title=f"Load {config.ROLE_LABEL[role]} X-ray",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All", "*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showwarning("Load", "Could not read that image.")
            return
        self._images[role] = img
        self._image_paths[role] = path
        self.canvas[role].set_image(img)

    def _clear(self, role: str):
        self.canvas[role].clear_points()

    def _clear_check(self):
        for r in config.CAMERA_ROLES:
            self.canvas[r].set_predicted([])
        self.status.configure(text="")

    def _check_registration(self):
        """Fit a DLT from the current clicks and overlay where it predicts each
        fiducial (red x).  Every x on its blue circle = good registration."""
        from core.dlt import ProjectionMatrix
        try:
            fids = self._parse_fiducials()
        except ValueError:
            messagebox.showwarning("Fiducials", "One of the coordinate lines is not numeric.")
            return
        n = len(fids)
        if n < 6:
            messagebox.showwarning("Fiducials", "Enter at least 6 fiducial coordinates.")
            return
        msgs = []
        for r in config.CAMERA_ROLES:
            clicks = self.canvas[r].points
            if len(clicks) != n:
                msgs.append(f"{config.XRAY_LABEL[r]}: {len(clicks)} clicks vs {n} fiducials - skipped")
                self.canvas[r].set_predicted([])
                continue
            try:
                P = ProjectionMatrix.from_correspondences(fids, np.asarray(clicks, float))
            except Exception as exc:  # noqa: BLE001
                msgs.append(f"{config.XRAY_LABEL[r]}: DLT failed ({exc})")
                continue
            self.canvas[r].set_predicted(P.project(fids).tolist())
            err = P.reprojection_error(fids, np.asarray(clicks, float))
            flag = "" if err <= config.DLT_REPROJ_WARN_PX else "  (high)"
            msgs.append(f"{config.XRAY_LABEL[r]}: mean error {err:.2f} px{flag}")
        self.status.configure(text="Registration check:\n" + "\n".join(msgs))

    # ── compute / persist ──────────────────────────────────────────────────────
    def _compute_save(self):
        try:
            fids = self._parse_fiducials()
        except ValueError:
            messagebox.showwarning("Fiducials", "One of the coordinate lines is not numeric.")
            return
        n = len(fids)
        if n < 6:
            messagebox.showwarning("Fiducials", "Enter at least 6 fiducial coordinates.")
            return

        reg = ModelRegistration(name=self.name_var.get().strip() or "model_01")
        reg.fiducials_model = fids.tolist()
        msgs = []
        for r in config.CAMERA_ROLES:
            clicks = self.canvas[r].points
            if len(clicks) != n:
                messagebox.showwarning(
                    "Clicks",
                    f"{config.ROLE_LABEL[r]}: {len(clicks)} points clicked but {n} fiducials defined. "
                    "They must match (same order).")
                return
            reg.view(r).image_path = self._image_paths[r]
            reg.view(r).clicks = [list(c) for c in clicks]
            try:
                err = reg.compute_view(r)
            except Exception as exc:  # noqa: BLE001
                messagebox.showwarning("DLT", f"{config.ROLE_LABEL[r]}: {exc}")
                return
            flag = "" if err <= config.DLT_REPROJ_WARN_PX else "  (high - check clicks/coords)"
            msgs.append(f"{r.upper()} reprojection: {err:.2f} px{flag}")

        # carry the verification hole from settings if present
        path = reg.save()
        self.session.model = reg
        self.status.configure(text="Saved to data/models.\n" + "\n".join(msgs))

    def _load_model(self):
        path = filedialog.askopenfilename(
            title="Load model registration", initialdir=paths.MODEL_DIR,
            filetypes=[("FluoroSim model", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            reg = ModelRegistration.load(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showwarning("Load", f"Could not load: {exc}")
            return
        self.session.model = reg
        self.name_var.set(reg.name)
        self.fid_text.delete("1.0", "end")
        self.fid_text.insert("1.0", "\n".join(f"{x:.2f} {y:.2f} {z:.2f}"
                                              for x, y, z in reg.fiducials_model))
        for r in config.CAMERA_ROLES:
            v = reg.view(r)
            img = cv2.imread(v.image_path) if v.image_path else None
            if img is not None:
                self._images[r] = img
                self._image_paths[r] = v.image_path
                self.canvas[r].set_image(img)
                self.canvas[r].set_points(v.clicks)
        self._update_counts()
        self.status.configure(text=f"Loaded '{reg.name}'. "
                                   + ("Complete." if reg.is_complete else "Incomplete - recompute."))

    def on_show(self):
        if self.session.model is not None and not self.fid_text.get("1.0", "end").strip():
            pass
        self._update_counts()
