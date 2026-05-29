"""
ui/or_setup.py — One-time OR visit: compute AP and LAT projection matrices.

This view is used once per spine model, in a room with fluoroscopy access.

Workflow
────────
1.  Select (or confirm) the target model.
2.  Load the AP X-ray image (PNG exported from the C-arm).
3.  Load the LAT X-ray image.
4.  For each image, click ≥ 6 radiopaque fiducial markers that are embedded
    in the 3D-printed model at precisely known positions.
5.  Enter the 3D model-space coordinates for each clicked point.
    IMPORTANT: coordinates are in mm from the bottom-left corner of the
    CharucoBoard (board origin = model origin), measured along:
    +X = rightward, +Y = upward, +Z = toward cameras (out of board face).
6.  Click Compute.  The software runs the DLT algorithm and shows the
    reprojection error.  Anything under 2 px is excellent.
7.  Click Save.  Projection matrices are written to disk.
    This is the only fluoroscopy ever required for this model.
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path

from ui.widgets import (
    DarkFrame, primary_btn, success_btn,
    info_label, section_label,
    BG, BG2, BG3, FG, FG_MUTED, FG_SUCCESS, FG_ERR, FG_WARN,
    FONT_BODY, FONT_LABEL, FONT_TITLE, FONT_MONO,
)
from core.projection  import compute_projection_matrix, reprojection_error, save_projection_matrix
from core.model_config import save_xray, save_fiducials, load_fiducials, list_models, load_model
from config import MODELS_DIR

CANVAS_W = 560
CANVAS_H = 460
MIN_FIDUCIALS = 6


# ── Fiducial canvas ────────────────────────────────────────────────────────────

class FiducialCanvas(tk.Frame):
    """
    Displays an X-ray image and collects fiducial click positions.
    Left-click to add a point, right-click to undo the last point.
    """

    def __init__(self, parent, view_label: str, **kwargs):
        super().__init__(parent, bg=BG2, **kwargs)
        self._label     = view_label
        self._points:   list[tuple[float, float]] = []
        self._img_orig: np.ndarray | None = None
        self._scale     = 1.0
        self._tk_img    = None

        tk.Label(self, text=view_label, font=FONT_TITLE,
                 fg=FG, bg=BG2).pack(pady=4)

        self._canvas = tk.Canvas(self, width=CANVAS_W, height=CANVAS_H,
                                 bg="#111", cursor="crosshair")
        self._canvas.pack()
        self._canvas.bind("<Button-1>",  self._on_click)
        self._canvas.bind("<Button-3>",  self._on_right_click)

        self._count_var = tk.StringVar(value="0 points")
        tk.Label(self, textvariable=self._count_var,
                 font=FONT_LABEL, fg=FG_MUTED, bg=BG2).pack(pady=2)
        info_label(self, "Left-click = add point  •  Right-click = undo last",
                   color=FG_MUTED).pack()

    def load_image(self, path: str):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load: {path}")
        self._img_orig = img
        self._points   = []
        self._redraw()

    def _redraw(self):
        if self._img_orig is None:
            return
        h, w = self._img_orig.shape[:2]
        self._scale = min(CANVAS_W / w, CANVAS_H / h)
        dw = int(w * self._scale)
        dh = int(h * self._scale)

        rgb = cv2.cvtColor(self._img_orig, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((dw, dh), Image.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(image=pil)

        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

        for i, (px, py) in enumerate(self._points):
            cx, cy = int(px * self._scale), int(py * self._scale)
            r = 8
            self._canvas.create_line(cx-r, cy, cx+r, cy, fill="#ff4444", width=2)
            self._canvas.create_line(cx, cy-r, cx, cy+r, fill="#ff4444", width=2)
            self._canvas.create_oval(cx-3, cy-3, cx+3, cy+3, outline="#ffff00")
            self._canvas.create_text(cx+12, cy-10, text=str(i+1),
                                     fill="#ffff00", font=FONT_MONO)
        self._count_var.set(f"{len(self._points)} points marked")

    def _on_click(self, event):
        if self._img_orig is None:
            return
        self._points.append((event.x / self._scale, event.y / self._scale))
        self._redraw()

    def _on_right_click(self, event):
        if self._points:
            self._points.pop()
            self._redraw()

    @property
    def points(self) -> list[tuple[float, float]]:
        return list(self._points)

    @property
    def has_image(self) -> bool:
        return self._img_orig is not None

    @property
    def image(self) -> np.ndarray | None:
        return self._img_orig


# ── OR Setup view ──────────────────────────────────────────────────────────────

class ORSetupView(DarkFrame):

    def __init__(self, parent, app, state, **kwargs):
        super().__init__(parent, **kwargs)
        self._app   = app
        self._state = state
        self._build()

    # ── Build ────────────────────────────────────────────────────────────────

    def _build(self):
        hdr = tk.Frame(self, bg=BG2, pady=8)
        hdr.pack(fill=tk.X)
        section_label(hdr, "OR Setup — One-Time Projection Calibration").pack(pady=4)
        info_label(
            hdr,
            "Embed ≥ 6 radiopaque markers (metal BBs or barium spheres) at precisely "
            "known positions in the 3D-printed spine model.  Take AP and lateral "
            "fluoroscopy shots.  Load each image, click each visible marker, enter "
            "its 3D coordinates (mm from the CharucoBoard bottom-left corner), then "
            "Compute and Save.  This is the only fluoroscopy ever required.",
            color=FG_MUTED,
        ).pack(padx=20, pady=(0, 4))

        # Model selector
        sel_row = tk.Frame(self, bg=BG)
        sel_row.pack(fill=tk.X, padx=20, pady=4)
        tk.Label(sel_row, text="Model:", font=FONT_LABEL,
                 fg=FG_MUTED, bg=BG).pack(side=tk.LEFT)
        self._model_var = tk.StringVar(value="")
        models = list_models()
        self._model_combo = tk.OptionMenu(sel_row, self._model_var,
                                          *models if models else ["(none)"])
        self._model_combo.configure(bg=BG3, fg=FG, font=FONT_BODY,
                                    activebackground=BG3, relief=tk.FLAT)
        self._model_combo.pack(side=tk.LEFT, padx=8)
        if models:
            self._model_var.set(models[0])

        # X-ray canvases
        canvas_row = tk.Frame(self, bg=BG)
        canvas_row.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        self._canvas_ap  = FiducialCanvas(canvas_row, "AP  (anteroposterior)")
        self._canvas_ap.pack(side=tk.LEFT, padx=8, fill=tk.BOTH, expand=True)
        primary_btn(canvas_row, "Load AP X-ray",
                    command=lambda: self._load_xray("ap"), width=14
                    ).pack(side=tk.LEFT, padx=4, anchor=tk.N, pady=60)

        self._canvas_lat = FiducialCanvas(canvas_row, "Lateral")
        self._canvas_lat.pack(side=tk.LEFT, padx=8, fill=tk.BOTH, expand=True)
        primary_btn(canvas_row, "Load LAT X-ray",
                    command=lambda: self._load_xray("lat"), width=14
                    ).pack(side=tk.LEFT, padx=4, anchor=tk.N, pady=60)

        # 3D coordinate entry
        coord_frame = tk.Frame(self, bg=BG2, padx=14, pady=8)
        coord_frame.pack(fill=tk.X, padx=20, pady=4)
        tk.Label(coord_frame,
                 text="3-D coordinates of clicked fiducials  (X, Y, Z in mm from CharucoBoard origin — one per line):",
                 font=FONT_LABEL, fg=FG_MUTED, bg=BG2).pack(anchor=tk.W)
        self._coord_text = tk.Text(coord_frame, height=5, width=60,
                                   bg=BG3, fg=FG, font=FONT_MONO,
                                   relief=tk.FLAT, insertbackground=FG)
        self._coord_text.pack(pady=4)
        info_label(coord_frame,
                   "Example:  0.0, 15.0, 0.0\n"
                   "One line per fiducial, same order as click order in both images.",
                   color=FG_MUTED).pack(anchor=tk.W)

        # Result / status
        self._result_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._result_var,
                 font=FONT_LABEL, fg=FG_WARN, bg=BG).pack()

        # Bottom buttons
        btns = tk.Frame(self, bg=BG)
        btns.pack(pady=8)
        primary_btn(btns, "↩ Back",       command=self._back,    width=12).pack(side=tk.LEFT, padx=6)
        primary_btn(btns, "Compute",      command=self._compute, width=12).pack(side=tk.LEFT, padx=6)
        self._save_btn = success_btn(btns, "Save & Return →",
                                     command=self._save, width=18)
        self._save_btn.configure(state=tk.DISABLED)
        self._save_btn.pack(side=tk.LEFT, padx=6)

        self._P_ap  = None
        self._P_lat = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_show(self, **kwargs):
        # Refresh model list
        models = list_models()
        menu = self._model_combo["menu"]
        menu.delete(0, tk.END)
        for m in models:
            menu.add_command(label=m, command=lambda v=m: self._model_var.set(v))
        if self._state.model:
            self._model_var.set(self._state.model.model_id)
        elif models:
            self._model_var.set(models[0])
        self._result_var.set("")
        self._save_btn.configure(state=tk.DISABLED)
        self._P_ap = self._P_lat = None

    # ── Actions ──────────────────────────────────────────────────────────────

    def _load_xray(self, view: str):
        path = filedialog.askopenfilename(
            title=f"Select {view.upper()} X-ray image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            canvas = self._canvas_ap if view == "ap" else self._canvas_lat
            canvas.load_image(path)
        except ValueError as e:
            messagebox.showerror("Load Error", str(e))

    def _parse_coords(self) -> np.ndarray | None:
        raw = self._coord_text.get("1.0", tk.END).strip()
        if not raw:
            messagebox.showerror("Coordinates", "Enter 3D coordinates for the fiducials.")
            return None
        pts = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parts = [float(v.strip()) for v in line.replace(";", ",").split(",")]
                if len(parts) != 3:
                    raise ValueError
                pts.append(parts)
            except ValueError:
                messagebox.showerror("Coordinates",
                                     f"Could not parse: '{line}'\n"
                                     "Each line must be:  X, Y, Z")
                return None
        return np.array(pts, dtype=np.float64)

    def _compute(self):
        obj_pts = self._parse_coords()
        if obj_pts is None:
            return

        pts_ap  = np.array(self._canvas_ap.points,  dtype=np.float64)
        pts_lat = np.array(self._canvas_lat.points, dtype=np.float64)

        n = len(obj_pts)
        if len(pts_ap) != n or len(pts_lat) != n:
            messagebox.showerror(
                "Mismatch",
                f"Number of 3D coordinates ({n}) must match "
                f"AP clicks ({len(pts_ap)}) and LAT clicks ({len(pts_lat)})."
            )
            return

        if n < MIN_FIDUCIALS:
            messagebox.showerror("Insufficient Fiducials",
                                 f"At least {MIN_FIDUCIALS} fiducials are required (got {n}).")
            return

        try:
            P_ap  = compute_projection_matrix(obj_pts, pts_ap)
            P_lat = compute_projection_matrix(obj_pts, pts_lat)
        except Exception as e:
            messagebox.showerror("DLT Error", str(e))
            return

        err_ap  = reprojection_error(P_ap,  obj_pts, pts_ap)
        err_lat = reprojection_error(P_lat, obj_pts, pts_lat)

        quality = lambda e: "excellent" if e < 1.5 else "good" if e < 3.0 else "poor — consider adding more fiducials"
        self._result_var.set(
            f"AP reprojection: {err_ap:.2f} px ({quality(err_ap)})   "
            f"LAT reprojection: {err_lat:.2f} px ({quality(err_lat)})"
        )

        self._P_ap  = P_ap
        self._P_lat = P_lat
        self._save_btn.configure(state=tk.NORMAL)

        self._obj_pts_cache  = obj_pts
        self._pts_ap_cache   = pts_ap
        self._pts_lat_cache  = pts_lat

    def _save(self):
        model_id = self._model_var.get()
        if not model_id or model_id == "(none)":
            messagebox.showwarning("No Model", "Select a model first.")
            return

        save_projection_matrix(model_id, "ap",  self._P_ap)
        save_projection_matrix(model_id, "lat", self._P_lat)

        # Save X-ray images alongside the matrices
        if self._canvas_ap.image is not None:
            save_xray(model_id, "ap",  self._canvas_ap.image)
        if self._canvas_lat.image is not None:
            save_xray(model_id, "lat", self._canvas_lat.image)

        # Persist fiducial correspondences for future reference
        save_fiducials(model_id,
                       self._obj_pts_cache,
                       self._pts_ap_cache,
                       self._pts_lat_cache)

        messagebox.showinfo("Saved",
                            f"Projection matrices saved for model '{model_id}'.\n"
                            "This model is now ready for training.")

        # Reload the model in state so navigation picks up the new data
        try:
            self._state.model = load_model(model_id)
            self._state.build_overlays()
        except Exception:
            pass

        self._back()

    def _back(self):
        self._app.go_to_navigation()
