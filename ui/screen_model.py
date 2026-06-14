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
from collections import deque

import cv2
import numpy as np

import config
from core import paths, markers
from core.model_store import ModelRegistration
from core.camera_io import CameraStream
from core.tracking import (BoardTracker, ProbeTracker, CameraView,
                           solve_probe_multiview, draw_board_pose, draw_probe_in_view)
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

        ttk.Label(left, text="Fiducial coordinates in the board frame\n"
                             "(X Y Z per line, mm)",
                  style="Muted.TLabel", justify="left").pack(anchor="w")
        self.fid_text = tk.Text(left, width=26, height=12, bg=W.PANEL, fg=W.FG,
                                insertbackground=W.FG, relief="flat")
        self.fid_text.pack(anchor="w", pady=(2, 4))
        self.fid_count = ttk.Label(left, text="0 fiducials", style="Muted.TLabel")
        self.fid_count.pack(anchor="w")
        self.fid_text.bind("<KeyRelease>", lambda e: self._update_counts())

        ttk.Separator(left).pack(fill="x", pady=12)
        ttk.Button(left, text="Compute & save", style="Accent.TButton",
                   command=self._compute_save).pack(anchor="w", fill="x")
        ttk.Button(left, text="Align CAD \u2192 probe frame (best fit)",
                   command=self._open_align).pack(anchor="w", fill="x", pady=(6, 0))
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

    def _open_align(self):
        if not self.session.cameras_calibrated:
            messagebox.showwarning("Align", "Calibrate both cameras first (Cameras screen).")
            return
        try:
            cad = self._parse_fiducials()
        except ValueError:
            messagebox.showwarning("Align", "A fiducial line is not numeric.")
            return
        if len(cad) < 3:
            messagebox.showwarning("Align", "Paste your CAD fiducials first (at least 3).")
            return
        DigitizeDialog(self, self.session, len(cad), self._align_done)

    def _align_done(self, captured):
        """Best-fit the CAD fiducials (exact geometry) onto the probe-measured
        points (correct frame) with a rigid Kabsch transform, then rewrite the
        fiducial box in the tracker frame."""
        cad = self._parse_fiducials()
        cap = np.asarray(captured, float)
        n = min(len(cad), len(cap))
        if n < 3:
            return
        A, B = cad[:n], cap[:n]               # A = CAD, B = tracker-frame captures
        cA, cB = A.mean(0), B.mean(0)
        H = (A - cA).T @ (B - cB)
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        t = cB - R @ cA
        aligned_all = (R @ cad.T).T + t       # apply to ALL CAD fiducials
        resid = np.linalg.norm((R @ A.T).T + t - B, axis=1)
        self.fid_text.delete("1.0", "end")
        self.fid_text.insert("1.0",
                             "\n".join(f"{p[0]:.2f} {p[1]:.2f} {p[2]:.2f}" for p in aligned_all))
        self._update_counts()
        self.status.configure(
            text=f"Aligned {len(cad)} CAD fiducials to the probe frame using {n} touches.\n"
                 f"fit error: mean {resid.mean():.1f} mm, max {resid.max():.1f} mm.\n"
                 f"Now click the bearings on AP + lateral (same order) and Compute & save.")

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

        reg.save()
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


class DigitizeDialog(tk.Toplevel):
    """Touch each bearing with the probe tip; the two-camera tracker tip is
    averaged over several steady frames and recorded. These points define the
    true tracker frame that the CAD fiducials are then best-fit onto."""

    def __init__(self, parent, session, expected, on_done):
        super().__init__(parent)
        self.title("Align fiducials to probe")
        self.configure(bg=W.BG)
        self.session = session
        self.expected = expected
        self.on_done = on_done
        self.streams = {r: None for r in config.CAMERA_ROLES}
        self._board = BoardTracker()
        self._probe = ProbeTracker()
        self._probe_aruco = markers.make_aruco_detector()
        self._buf = deque(maxlen=60)
        self._captured = []
        self._tick_id = None

        wrap = ttk.Frame(self, style="App.TFrame"); wrap.pack(fill="both", expand=True, padx=14, pady=12)
        ttk.Label(wrap, text=f"Touch bearings 1\u2026{expected} with the probe tip, in the SAME "
                             f"order as your fiducial list.", style="Muted.TLabel").pack(anchor="w")
        feeds = ttk.Frame(wrap, style="App.TFrame"); feeds.pack(pady=(6, 0))
        self.feed = {}
        for r in config.CAMERA_ROLES:
            col = ttk.Frame(feeds, style="App.TFrame"); col.pack(side="left", padx=6)
            ttk.Label(col, text=config.ROLE_LABEL[r], style="Muted.TLabel").pack()
            self.feed[r] = W.VideoPanel(col, max_w=360, max_h=270); self.feed[r].pack()

        self.readout = ttk.Label(wrap, text="", style="Muted.TLabel", justify="left")
        self.readout.pack(anchor="w", pady=(10, 6))

        row = ttk.Frame(wrap, style="App.TFrame"); row.pack(fill="x")
        self.btn_cap = ttk.Button(row, text="Capture point", style="Accent.TButton", command=self._capture)
        self.btn_cap.pack(side="left")
        ttk.Button(row, text="Remove last", command=self._remove_last).pack(side="left", padx=6)
        ttk.Button(row, text="Done", command=self._done).pack(side="right")
        ttk.Button(row, text="Cancel", command=self._cancel).pack(side="right", padx=6)

        ttk.Label(wrap, text="Captured (tracker-frame mm):", style="Muted.TLabel").pack(anchor="w", pady=(10, 0))
        self.listbox = tk.Listbox(wrap, width=48, height=8, bg=W.PANEL, fg=W.FG, relief="flat", highlightthickness=0)
        self.listbox.pack(anchor="w", pady=(2, 0))

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.transient(parent)
        self._open_streams()
        self._tick()

    def _open_streams(self):
        for r in config.CAMERA_ROLES:
            idx = self.session.device_index.get(r)
            if idx is not None:
                try:
                    self.streams[r] = CameraStream(idx).start()
                except Exception:
                    self.streams[r] = None

    def _close(self):
        if self._tick_id is not None:
            self.after_cancel(self._tick_id); self._tick_id = None
        for r in config.CAMERA_ROLES:
            if self.streams.get(r) is not None:
                self.streams[r].release(); self.streams[r] = None
        self.destroy()

    def _tick(self):
        views, boards, frames = [], {}, {}
        for r in config.CAMERA_ROLES:
            stream = self.streams.get(r); intr = self.session.intrinsics.get(r)
            frame = stream.read() if stream else None
            frames[r] = (frame, intr)
            if frame is None or intr is None:
                continue
            board = self._board.estimate(frame, intr.mtx, intr.dist)
            det = self._probe.detect(frame); boards[r] = board
            if board is not None and det is not None:
                R_mc = cv2.Rodrigues(board.rvec)[0]
                views.append(CameraView(R_mc, board.tvec.flatten(), intr.mtx, intr.dist,
                                        det[0], det[1], det[2]))
        mv = solve_probe_multiview(views) if views else None

        for r in config.CAMERA_ROLES:
            frame, intr = frames[r]
            if frame is None or intr is None:
                self.feed[r].show(frame, "no signal / not calibrated"); continue
            disp = frame.copy()
            pc, pids, _ = self._probe_aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if pids is not None:
                cv2.aruco.drawDetectedMarkers(disp, pc, pids, (40, 160, 255))
            if boards.get(r) is not None:
                draw_board_pose(disp, boards[r], intr.mtx, intr.dist)
                if mv is not None:
                    draw_probe_in_view(disp, mv, boards[r], intr.mtx, intr.dist)
            self.feed[r].show(disp)

        good = (mv is not None and mv.n_cameras == 2 and mv.reproj_px <= config.DIGITIZE_REJECT_PX)
        if good:
            self._buf.append(mv.tip_model.copy())
        elif mv is None:
            self._buf.clear()

        spread = None
        if len(self._buf) >= 3:
            arr = np.array(self._buf); spread = float(np.mean(np.linalg.norm(arr - arr.mean(0), axis=1)))
        ready = len(self._buf) >= config.DIGITIZE_MIN_SAMPLES
        if mv is not None:
            t = mv.tip_model
            sp = f"spread \u00b1{spread:.1f}mm" if spread is not None else "spread --"
            msg = (f"point {len(self._captured)+1}/{self.expected}   "
                   f"tip [{t[0]:.0f},{t[1]:.0f},{t[2]:.0f}]   {mv.n_cameras}cam  "
                   f"reproj {mv.reproj_px:.1f}px   {sp}   frames {min(len(self._buf),config.DIGITIZE_MIN_SAMPLES)}/{config.DIGITIZE_MIN_SAMPLES}")
            if mv.n_cameras < 2:
                msg += "   (need BOTH cameras)"
            elif ready and spread is not None and spread <= config.DIGITIZE_STABLE_MM:
                msg += "   \u2714 steady"
            self.readout.configure(text=msg)
            self.btn_cap.state(["!disabled"] if ready else ["disabled"])
        else:
            self.readout.configure(text="probe not detected - bring the cube into both views")
            self.btn_cap.state(["disabled"])
        self._tick_id = self.after(50, self._tick)

    def _capture(self):
        if len(self._buf) < config.DIGITIZE_MIN_SAMPLES:
            return
        pts = np.array(self._buf); mean = pts.mean(0)
        jit = float(np.mean(np.linalg.norm(pts - mean, axis=1)))
        self._captured.append(mean.tolist())
        self.listbox.insert("end", f"{len(self._captured):>2}:  [{mean[0]:6.1f},{mean[1]:6.1f},{mean[2]:6.1f}]   \u00b1{jit:.1f}mm")
        self._buf.clear()

    def _remove_last(self):
        if self._captured:
            self._captured.pop(); self.listbox.delete("end")

    def _done(self):
        cb, pts = self.on_done, list(self._captured)
        self._close()
        cb(pts)

    def _cancel(self):
        self._close()
