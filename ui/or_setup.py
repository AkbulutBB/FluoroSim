"""
ui/or_setup.py — One-time OR visit setup.

This view is used ONCE in the operating room (or lab with fluoroscopy access)
to establish the relationship between 3D model coordinates and X-ray pixel
positions.

Workflow:
  1. Select (or create) a model package.
  2. Load AP and LAT X-ray images.
  3. For each image, click N ≥ 6 radiopaque fiducial markers whose 3D
     coordinates are known from the model design (embedded metal spheres).
  4. Enter the 3D model coordinates for each clicked point.
  5. Click Compute — the software calculates P_ap and P_lat.
  6. The matrices are saved to the model package for all future sessions.

After this step, no fluoroscopy is ever required again.
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
import json
import shutil

from ui.widgets import (
    DarkFrame, primary_btn, success_btn, section_label, info_label,
    BG, BG2, BG3, FG, FG_MUTED, FG_SUCCESS, FG_ERR, FG_WARN,
    FONT_BODY, FONT_LABEL, FONT_TITLE, FONT_MONO,
)
from core.projection import compute_projection_matrix, reprojection_error, save_projection_matrix
from core.model_config import save_fiducials, load_fiducials, list_models
from config import MODELS_DIR


CANVAS_W = 600
CANVAS_H = 700
MIN_FIDUCIALS = 6


class FiducialCanvas(tk.Frame):
    """
    Displays an X-ray image and allows the user to click fiducial marker positions.
    Shows numbered crosshairs at each clicked position.
    """

    def __init__(self, parent, label: str, canvas_h: int = 480, **kwargs):
        super().__init__(parent, bg=BG2, **kwargs)
        self._label = label
        self._points: list[tuple[float, float]] = []  # pixel coords in original image
        self._img_orig: np.ndarray | None = None
        self._scale: float = 1.0

        tk.Label(self, text=label, font=FONT_TITLE, fg=FG, bg=BG2).pack()
        self._canvas = tk.Canvas(self, width=CANVAS_W, height=canvas_h,
                                  bg="#111", cursor="crosshair")
        self._canvas.pack()
        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<Button-3>", self._on_right_click)

        self._tk_img = None
        self._count_var = tk.StringVar(value="0 points marked")
        tk.Label(self, textvariable=self._count_var, font=FONT_LABEL,
                 fg=FG_MUTED, bg=BG2).pack(pady=2)
        info_label(self, "Left-click to add • Right-click to undo last", FG_MUTED).pack()

    def load_image(self, path: str):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load image: {path}")
        self._img_orig = img
        self._points   = []
        self._redraw()

    def _redraw(self):
        if self._img_orig is None:
            return
        h, w = self._img_orig.shape[:2]
        self._scale = min(CANVAS_W / w, CANVAS_H / h)
        disp_w = int(w * self._scale)
        disp_h = int(h * self._scale)

        rgb = cv2.cvtColor(self._img_orig, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((disp_w, disp_h), Image.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(image=pil)

        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

        # Draw crosshairs
        for i, (px, py) in enumerate(self._points):
            cx = int(px * self._scale)
            cy = int(py * self._scale)
            r  = 8
            self._canvas.create_line(cx-r, cy, cx+r, cy, fill="#ff4444", width=2)
            self._canvas.create_line(cx, cy-r, cx, cy+r, fill="#ff4444", width=2)
            self._canvas.create_oval(cx-3, cy-3, cx+3, cy+3, outline="#ffff00")
            self._canvas.create_text(cx+12, cy-10, text=str(i+1),
                                      fill="#ffff00", font=FONT_MONO)

        self._count_var.set(f"{len(self._points)} points marked")

    def _on_click(self, event):
        if self._img_orig is None:
            return
        px = event.x / self._scale
        py = event.y / self._scale
        self._points.append((px, py))
        self._redraw()

    def _on_right_click(self, event):
        if self._points:
            self._points.pop()
            self._redraw()

    @property
    def points(self) -> list[tuple[float, float]]:
        return list(self._points)

    @property
    def image_path(self) -> str | None:
        return None  # stored externally


class ORSetupView(DarkFrame):

    def __init__(self, parent, app, state, **kwargs):
        super().__init__(parent, **kwargs)
        self._app   = app
        self._state = state
        self._model_id: str | None = None
        self._ap_path:  str | None = None
        self._lat_path: str | None = None

        self._build()

    # ── Build ───────────────────────────────────────────────────────────────

    def _build(self):
        hdr = tk.Frame(self, bg=BG2)
        hdr.pack(fill=tk.X, pady=(0, 6))
        section_label(hdr, "OR Setup — One-Time Projection Calibration").pack(pady=10)
        info_label(
            hdr,
            "Use this tool once in the OR with fluoroscopy access. "
            "Place ≥6 radiopaque markers at known 3D positions on the spine model, "
            "take AP and LAT shots, then load them here and click each marker. "
            "After computing, no further fluoroscopy is needed.",
            color=FG_MUTED,
        ).pack(pady=(0, 6))

        # Top controls
        ctrl = tk.Frame(self, bg=BG)
        ctrl.pack(fill=tk.X, padx=12, pady=4)

        tk.Label(ctrl, text="Model:", font=FONT_BODY, fg=FG, bg=BG).pack(side=tk.LEFT)
        self._model_var = tk.StringVar(value="— select —")
        self._model_menu = tk.OptionMenu(ctrl, self._model_var, "— select —",
                                          command=self._on_model_select)
        self._model_menu.configure(bg=BG3, fg=FG, font=FONT_LABEL, relief=tk.FLAT)
        self._model_menu.pack(side=tk.LEFT, padx=6)

        primary_btn(ctrl, "New model…", command=self._new_model, width=12).pack(side=tk.LEFT, padx=4)

        tk.Label(ctrl, text="  Load X-rays:", font=FONT_BODY, fg=FG, bg=BG).pack(side=tk.LEFT, padx=8)
        primary_btn(ctrl, "AP image", command=lambda: self._load_xray("ap"), width=10).pack(side=tk.LEFT, padx=3)
        primary_btn(ctrl, "LAT image", command=lambda: self._load_xray("lat"), width=10).pack(side=tk.LEFT, padx=3)

        # Main area: canvases left, coordinates panel right
        main_row = tk.Frame(self, bg=BG)
        main_row.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Left: two X-ray canvases
        canvas_row = tk.Frame(main_row, bg=BG)
        canvas_row.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._canvas_ap  = FiducialCanvas(canvas_row, "AP (Craniocaudal)", canvas_h=480)
        self._canvas_ap.pack(side=tk.LEFT, padx=6, fill=tk.BOTH, expand=True)

        self._canvas_lat = FiducialCanvas(canvas_row, "Lateral", canvas_h=480)
        self._canvas_lat.pack(side=tk.LEFT, padx=6, fill=tk.BOTH, expand=True)

        # Right: coordinates + buttons panel
        self._right_panel = tk.Frame(main_row, bg=BG2, padx=12, pady=10, width=280)
        self._right_panel.pack(side=tk.LEFT, fill=tk.Y, padx=6)
        self._right_panel.pack_propagate(False)

        tk.Label(self._right_panel, text="3D coordinates",
                 font=("Segoe UI", 10, "bold"), fg=FG, bg=BG2).pack(anchor=tk.W)
        tk.Label(self._right_panel, text="One fiducial per line: X Y Z (mm)",
                 font=FONT_LABEL, fg=FG_MUTED, bg=BG2, wraplength=250).pack(anchor=tk.W, pady=(2,4))

        self._coords_text = tk.Text(self._right_panel, height=14, width=26,
                                     bg=BG3, fg=FG, font=FONT_MONO,
                                     insertbackground=FG, relief=tk.FLAT)
        self._coords_text.pack(fill=tk.X, pady=4)

        tk.Label(self._right_panel,
                 text="Example (6 fiducials):\n0 0 0\n60 0 0\n30 40 0\n0 0 30\n60 0 30\n30 40 30",
                 font=FONT_MONO, fg=FG_MUTED, bg=BG2, justify=tk.LEFT).pack(anchor=tk.W, pady=4)

        tk.Label(self._right_panel,
                 text="Points must be clicked\nin the SAME ORDER\nin both X-ray images.",
                 font=FONT_LABEL, fg=FG_WARN, bg=BG2, wraplength=250, justify=tk.LEFT).pack(anchor=tk.W, pady=6)

        primary_btn(self._right_panel, "Validate", command=self._validate, width=20).pack(pady=4)
        self._compute_btn = success_btn(self._right_panel, "Compute & Save →",
                                         command=self._compute, width=20)
        self._compute_btn.configure(state=tk.DISABLED)
        self._compute_btn.pack(pady=4)

        self._result_var = tk.StringVar(value="")
        tk.Label(self._right_panel, textvariable=self._result_var, font=FONT_LABEL,
                 fg=FG_SUCCESS, bg=BG2, wraplength=250, justify=tk.LEFT).pack(pady=6)

        self._refresh_model_menu()

    def on_show(self, **kwargs):
        self._refresh_model_menu()
        # If a model is already selected, restore its saved fiducials
        if self._model_id:
            self._restore_fiducials(self._model_id)

    def _on_model_select(self, value):
        self._model_var.set(value)
        self._model_id = value if value != "— select —" else None
        if self._model_id:
            self._restore_fiducials(self._model_id)

    def _restore_fiducials(self, model_id: str):
        """Load previously saved fiducial data back into the UI."""
        from core.model_config import load_fiducials
        data = load_fiducials(model_id)
        if data is None:
            return

        # Restore 3D coordinates text box
        self._coords_text.delete("1.0", tk.END)
        for row in data["obj_pts"]:
            self._coords_text.insert(tk.END, f"{row[0]:.2f} {row[1]:.2f} {row[2]:.2f}\n")

        # Restore canvas click points (requires X-ray images to be loaded)
        ap_path  = Path(MODELS_DIR) / model_id / "xray_ap.png"
        lat_path = Path(MODELS_DIR) / model_id / "xray_lat.png"

        if ap_path.exists() and self._canvas_ap._img_orig is None:
            try:
                self._canvas_ap.load_image(str(ap_path))
                self._ap_path = str(ap_path)
            except Exception:
                pass

        if lat_path.exists() and self._canvas_lat._img_orig is None:
            try:
                self._canvas_lat.load_image(str(lat_path))
                self._lat_path = str(lat_path)
            except Exception:
                pass

        # Restore click positions
        if "img_pts_ap" in data and self._canvas_ap._img_orig is not None:
            self._canvas_ap._points = [tuple(p) for p in data["img_pts_ap"].tolist()]
            self._canvas_ap._redraw()

        if "img_pts_lat" in data and self._canvas_lat._img_orig is not None:
            self._canvas_lat._points = [tuple(p) for p in data["img_pts_lat"].tolist()]
            self._canvas_lat._redraw()

        self._result_var.set(f"Loaded {len(data['obj_pts'])} saved fiducials.")

    def _go_to_nav(self):
        if self._model_id:
            self._app.go_to_navigation_direct(self._model_id)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _refresh_model_menu(self):
        models = list_models()
        menu   = self._model_menu["menu"]
        menu.delete(0, tk.END)
        menu.add_command(label="— select —", command=lambda: self._model_var.set("— select —"))
        for m in models:
            menu.add_command(label=m, command=lambda v=m: self._on_model_select(v))

    def _new_model(self):
        win = tk.Toplevel(self)
        win.title("New Model")
        win.configure(bg=BG)
        win.geometry("400x200")
        tk.Label(win, text="Model ID (no spaces):", font=FONT_BODY, fg=FG, bg=BG).pack(pady=10)
        id_var  = tk.StringVar()
        name_var = tk.StringVar()
        tk.Entry(win, textvariable=id_var, font=FONT_BODY, bg=BG3, fg=FG,
                 insertbackground=FG, width=24).pack()
        tk.Label(win, text="Display name:", font=FONT_BODY, fg=FG, bg=BG).pack(pady=6)
        tk.Entry(win, textvariable=name_var, font=FONT_BODY, bg=BG3, fg=FG,
                 insertbackground=FG, width=24).pack()

        def _create():
            mid  = id_var.get().strip().replace(" ", "_")
            name = name_var.get().strip() or mid
            if not mid:
                return
            model_dir = Path(MODELS_DIR) / mid
            model_dir.mkdir(parents=True, exist_ok=True)
            cfg = {
                "name": name,
                "slots": [
                    {"label": "Slot 1",
                     "cube_center_model": [0.0, 0.0, 20.0],
                     "cube_R_model": [1,0,0, 0,0,1, 0,-1,0]},
                    {"label": "Slot 2",
                     "cube_center_model": [60.0, 0.0, 20.0],
                     "cube_R_model": [1,0,0, 0,0,1, 0,-1,0]},
                ],
            }
            import json
            with open(model_dir / "model_config.json", "w") as f:
                json.dump(cfg, f, indent=2)
            win.destroy()
            self._refresh_model_menu()
            self._model_var.set(mid)
            self._model_id = mid

        success_btn(win, "Create", command=_create, width=12).pack(pady=12)

    def _load_xray(self, view: str):
        path = filedialog.askopenfilename(
            title=f"Load {view.upper()} X-ray",
            filetypes=[("Image files", "*.png *.jpg *.bmp *.tiff"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            canvas = self._canvas_ap if view == "ap" else self._canvas_lat
            canvas.load_image(path)
            if view == "ap":
                self._ap_path = path
            else:
                self._lat_path = path
            # Copy to model directory if model selected
            if self._model_id:
                dst = Path(MODELS_DIR) / self._model_id / f"xray_{view}.png"
                img = cv2.imread(path)
                cv2.imwrite(str(dst), img)
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def _parse_coords(self) -> np.ndarray | None:
        raw = self._coords_text.get("1.0", tk.END).strip()
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        pts = []
        for ln in lines:
            parts = ln.split()
            if len(parts) != 3:
                return None
            try:
                pts.append([float(p) for p in parts])
            except ValueError:
                return None
        return np.array(pts, np.float64) if pts else None

    def _validate(self):
        pts_ap  = self._canvas_ap.points
        pts_lat = self._canvas_lat.points
        obj_pts = self._parse_coords()

        issues = []
        if len(pts_ap) < MIN_FIDUCIALS:
            issues.append(f"AP: {len(pts_ap)} points marked (need ≥{MIN_FIDUCIALS})")
        if len(pts_lat) < MIN_FIDUCIALS:
            issues.append(f"LAT: {len(pts_lat)} points marked (need ≥{MIN_FIDUCIALS})")
        if obj_pts is None:
            issues.append("3D coordinates: invalid format")
        elif len(obj_pts) != len(pts_ap) or len(obj_pts) != len(pts_lat):
            issues.append(
                f"Point count mismatch: AP={len(pts_ap)}, LAT={len(pts_lat)}, "
                f"3D coords={len(obj_pts) if obj_pts is not None else 0}"
            )
        if not self._model_id:
            issues.append("No model selected")

        if issues:
            messagebox.showwarning("Validation", "\n".join(issues))
            self._compute_btn.configure(state=tk.DISABLED)
        else:
            self._result_var.set("Validation passed. Ready to compute.")
            self._compute_btn.configure(state=tk.NORMAL)

    def _compute(self):
        pts_ap  = np.array(self._canvas_ap.points,  np.float64)
        pts_lat = np.array(self._canvas_lat.points, np.float64)
        obj_pts = self._parse_coords()

        try:
            P_ap  = compute_projection_matrix(obj_pts, pts_ap)
            P_lat = compute_projection_matrix(obj_pts, pts_lat)
        except Exception as exc:
            messagebox.showerror("Computation Error", str(exc))
            return

        err_ap  = reprojection_error(P_ap,  obj_pts, pts_ap)
        err_lat = reprojection_error(P_lat, obj_pts, pts_lat)

        save_projection_matrix(self._model_id, "ap",  P_ap,  MODELS_DIR)
        save_projection_matrix(self._model_id, "lat", P_lat, MODELS_DIR)
        save_fiducials(self._model_id, obj_pts, pts_ap, pts_lat)

        self._result_var.set(
            f"Projection matrices saved. "
            f"Mean reprojection error — AP: {err_ap:.2f} px, LAT: {err_lat:.2f} px. "
            f"({'Excellent' if max(err_ap, err_lat) < 3 else 'Acceptable' if max(err_ap, err_lat) < 6 else 'High — recheck fiducials'})"
        )
        messagebox.showinfo(
            "OR Setup Complete",
            f"Projection matrices saved to model '{self._model_id}'.\n"
            f"AP reprojection error:  {err_ap:.2f} px\n"
            f"LAT reprojection error: {err_lat:.2f} px\n\n"
            "This model is now ready for radiation-free training sessions.",
        )
        # Show a direct navigation button
        if not hasattr(self, '_nav_btn'):
            self._nav_btn = success_btn(
                self._right_panel,
                "▶  Start Training Session",
                command=self._go_to_nav,
                width=22,
            )
            self._nav_btn.pack(pady=8)
