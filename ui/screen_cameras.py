"""
ui/screen_cameras.py  —  Camera calibration & verification
===========================================================

Two jobs, in two tabs:

  Calibrate : a one-time lens calibration per webcam.  Show the ChArUco board
              from several angles, capture views, solve, save.  Independent of
              where the camera sits.

  Verify    : the accuracy gate.  Put the probe tip in the platform's
              calibration hole (a point whose model-space coordinate you know)
              and read back the measured tip position.  Reports both the error
              vs the known hole AND the disagreement between the two cameras
              (a calibration-free quality signal).  This is also what you re-run
              whenever the cameras get bumped or repositioned.
"""

from __future__ import annotations
import json
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import cv2

import config
from core import paths
from core.camera_io import CameraStream, list_available_cameras
from core.intrinsics import CharucoCalibrator
from core.tracking import (BoardTracker, ProbeTracker, CameraView,
                           solve_probe_multiview, single_view_reproj,
                           draw_board_pose, draw_probe_in_view)
from core import markers
from ui.app import Screen
from ui import widgets as W


class CamerasScreen(Screen):
    def __init__(self, master, app):
        super().__init__(master, style="App.TFrame")
        self.app = app
        self.session = app.session

        self.streams: dict = {r: None for r in config.CAMERA_ROLES}
        self._tick_id = None
        self._calibrators: dict = {r: CharucoCalibrator() for r in config.CAMERA_ROLES}
        self._board_tracker = BoardTracker()
        self._probe_tracker = ProbeTracker()
        self._probe_aruco = markers.make_aruco_detector()
        self._calib_role = config.CAMERA_ROLES[0]
        self._available = []

        self._build_header()
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._build_calibrate_tab()
        self._build_verify_tab()
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self._on_tab_change())

        self._load_hole()

    # ── header ───────────────────────────────────────────────────────────────
    def _build_header(self):
        bar = ttk.Frame(self, style="App.TFrame")
        bar.pack(fill="x", padx=16, pady=12)
        ttk.Button(bar, text="\u2190 Home", command=lambda: self.app.show("home")).pack(side="left")
        ttk.Label(bar, text="Cameras", style="H2.TLabel").pack(side="left", padx=14)

    # ── calibrate tab ─────────────────────────────────────────────────────────
    def _build_calibrate_tab(self):
        tab = ttk.Frame(self.nb, style="App.TFrame")
        self.nb.add(tab, text="  Calibrate  ")

        left = ttk.Frame(tab, style="App.TFrame")
        left.pack(side="left", fill="y", padx=14, pady=14)

        ttk.Label(left, text="Camera", style="Muted.TLabel").pack(anchor="w")
        self.role_var = tk.StringVar(value=self._calib_role)
        for r in config.CAMERA_ROLES:
            ttk.Radiobutton(left, text=config.ROLE_LABEL[r], value=r, variable=self.role_var,
                            command=self._on_role_change).pack(anchor="w", pady=2)

        ttk.Label(left, text="Device", style="Muted.TLabel").pack(anchor="w", pady=(12, 0))
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(left, textvariable=self.device_var, state="readonly", width=22)
        self.device_combo.pack(anchor="w")
        self.device_combo.bind("<<ComboboxSelected>>", lambda e: self._open_calib_stream())
        ttk.Button(left, text="Rescan devices", command=self._scan_devices).pack(anchor="w", pady=6)

        ttk.Separator(left).pack(fill="x", pady=12)
        self.capture_lbl = ttk.Label(left, text="0 views captured", style="Muted.TLabel")
        self.capture_lbl.pack(anchor="w")
        ttk.Button(left, text="Capture view", command=self._capture_view).pack(anchor="w", pady=6, fill="x")
        ttk.Button(left, text="Reset views", command=self._reset_views).pack(anchor="w", fill="x")
        ttk.Button(left, text="Calibrate & save", style="Accent.TButton",
                   command=self._calibrate).pack(anchor="w", pady=(14, 0), fill="x")
        self.calib_status = ttk.Label(left, text="", style="Muted.TLabel", wraplength=240)
        self.calib_status.pack(anchor="w", pady=10)

        self.calib_video = W.VideoPanel(tab, max_w=640, max_h=480, style="Panel.TLabel")
        self.calib_video.pack(side="left", fill="both", expand=True, padx=14, pady=14)

    def _on_role_change(self):
        self._calib_role = self.role_var.get()
        self._refresh_calib_status()
        self._open_calib_stream()

    def _scan_devices(self):
        self._available = list_available_cameras()
        vals = [str(i) for i in self._available]
        self.device_combo.configure(values=vals)
        if vals and not self.device_var.get():
            self.device_var.set(vals[0])
            self._open_calib_stream()

    def _open_calib_stream(self):
        self._close_all_streams()
        try:
            idx = int(self.device_var.get())
        except (ValueError, tk.TclError):
            return
        self.streams[self._calib_role] = CameraStream(idx).start()
        self.session.device_index[self._calib_role] = idx

    def _capture_view(self):
        stream = self.streams.get(self._calib_role)
        if stream is None:
            return
        frame = stream.read()
        if frame is None:
            return
        ok, n = self._calibrators[self._calib_role].try_add_view(frame)
        cnt = self._calibrators[self._calib_role].n_views
        if ok:
            self.capture_lbl.configure(text=f"{cnt} views captured  (last: {n} corners)")
        else:
            self.capture_lbl.configure(text=f"{cnt} views  (rejected: only {n} corners - reposition board)")

    def _reset_views(self):
        self._calibrators[self._calib_role].reset()
        self.capture_lbl.configure(text="0 views captured")

    def _calibrate(self):
        role = self._calib_role
        try:
            intr = self._calibrators[role].calibrate()
        except Exception as exc:  # noqa: BLE001 - surface any solver error to the user
            messagebox.showwarning("Calibration", str(exc))
            return
        intr.save(role)
        self.session.intrinsics[role] = intr
        quality = "good" if intr.rms <= config.GOOD_CALIB_RMS else "usable, but high"
        self.calib_status.configure(
            text=f"Saved. RMS = {intr.rms:.2f} px ({quality}), {intr.n_views} views.")
        self._refresh_calib_status()

    def _refresh_calib_status(self):
        intr = self.session.intrinsics.get(self._calib_role)
        if intr is not None:
            self.calib_status.configure(
                text=f"Current calibration: RMS {intr.rms:.2f} px, {intr.n_views} views.")
        else:
            self.calib_status.configure(text="No calibration saved for this camera yet.")

    # ── verify tab ────────────────────────────────────────────────────────────
    def _build_verify_tab(self):
        tab = ttk.Frame(self.nb, style="App.TFrame")
        self.nb.add(tab, text="  Verify  ")

        videos = ttk.Frame(tab, style="App.TFrame")
        videos.pack(side="top", fill="both", expand=True, padx=14, pady=14)
        self.verify_video = {}
        for c, r in enumerate(config.CAMERA_ROLES):
            col = ttk.Frame(videos, style="App.TFrame")
            col.grid(row=0, column=c, padx=10, sticky="nsew")
            ttk.Label(col, text=config.ROLE_LABEL[r], style="Muted.TLabel").pack()
            vp = W.VideoPanel(col, max_w=480, max_h=360, style="Panel.TLabel")
            vp.pack()
            self.verify_video[r] = vp
        videos.columnconfigure(0, weight=1)
        videos.columnconfigure(1, weight=1)

        panel = ttk.Frame(tab, style="App.TFrame")
        panel.pack(fill="x", padx=14, pady=(0, 14))

        ttk.Label(panel, text="Calibration-hole coordinate (model mm):",
                  style="Muted.TLabel").grid(row=0, column=0, columnspan=6, sticky="w")
        self.hole_vars = []
        for i, axis in enumerate("XYZ"):
            ttk.Label(panel, text=axis, style="Muted.TLabel").grid(row=1, column=2 * i, padx=(6, 2))
            v = tk.StringVar()
            ttk.Entry(panel, textvariable=v, width=8).grid(row=1, column=2 * i + 1, padx=(0, 8))
            self.hole_vars.append(v)
        ttk.Button(panel, text="Save hole", command=self._save_hole).grid(row=1, column=6, padx=6)

        self.verify_readout = ttk.Label(panel, text="Place the probe tip in the calibration hole.",
                                        style="Muted.TLabel", justify="left")
        self.verify_readout.grid(row=2, column=0, columnspan=7, sticky="w", pady=(12, 0))
        self.btn_mark = ttk.Button(panel, text="Mark cameras verified",
                                   style="Accent.TButton", command=self._mark_verified)
        self.btn_mark.grid(row=3, column=0, columnspan=2, pady=(12, 0), sticky="w")
        self.btn_mark.state(["disabled"])
        self._last_error = None

    def _load_hole(self):
        hole = config.CALIB_HOLE_MODEL_MM
        try:
            with open(paths.VERIFICATION_FILE) as f:
                hole = np.asarray(json.load(f).get("calib_hole_model", hole), dtype=float)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        for v, val in zip(self.hole_vars, np.asarray(hole, dtype=float)):
            v.set(f"{val:.1f}")

    def _save_hole(self):
        try:
            hole = [float(v.get()) for v in self.hole_vars]
        except ValueError:
            messagebox.showwarning("Hole", "Enter three numeric coordinates.")
            return
        paths.ensure_dirs()
        with open(paths.VERIFICATION_FILE, "w") as f:
            json.dump({"calib_hole_model": hole}, f, indent=2)
        messagebox.showinfo("Hole", "Calibration-hole coordinate saved.")

    def _hole_xyz(self) -> np.ndarray:
        try:
            return np.array([float(v.get()) for v in self.hole_vars], dtype=float)
        except ValueError:
            return config.CALIB_HOLE_MODEL_MM

    def _open_verify_streams(self):
        self._close_all_streams()
        for r in config.CAMERA_ROLES:
            idx = self.session.device_index.get(r)
            if idx is None and self._available:
                idx = self._available[0]
            if idx is not None:
                self.streams[r] = CameraStream(idx).start()

    def _mark_verified(self):
        if self._last_error is not None and self._last_error <= config.TIP_ERROR_TOLERANCE_MM:
            self.session.last_hole_error_mm = self._last_error
            messagebox.showinfo("Verified",
                                 f"Cameras marked verified (tip error {self._last_error:.1f} mm).")

    # ── frame tick ─────────────────────────────────────────────────────────────
    def _on_tab_change(self):
        idx = self.nb.index(self.nb.select())
        if idx == 0:
            self._open_calib_stream()
        else:
            if not self.session.cameras_calibrated:
                self.verify_readout.configure(
                    text="Calibrate both cameras first (Calibrate tab).")
            self._open_verify_streams()

    def _tick(self):
        which = self.nb.index(self.nb.select())
        if which == 0:
            stream = self.streams.get(self._calib_role)
            frame = stream.read() if stream else None
            self.calib_video.show(frame)
        else:
            self._tick_verify()
        self._tick_id = self.after(40, self._tick)

    def _tick_verify(self):
        views, view_roles, boards, frames = [], [], {}, {}
        faces = {r: 0 for r in config.CAMERA_ROLES}
        for r in config.CAMERA_ROLES:
            stream = self.streams.get(r)
            intr = self.session.intrinsics.get(r)
            frame = stream.read() if stream else None
            frames[r] = (frame, intr)
            if frame is None or intr is None:
                continue
            board = self._board_tracker.estimate(frame, intr.mtx, intr.dist)
            det = self._probe_tracker.detect(frame)
            boards[r] = board
            if det is not None:
                faces[r] = det[2]
            if board is not None and det is not None:
                R_mc = cv2.Rodrigues(board.rvec)[0]
                views.append(CameraView(R_mc, board.tvec.flatten(), intr.mtx, intr.dist,
                                        det[0], det[1], det[2]))
                view_roles.append(r)

        mv = solve_probe_multiview(views) if views else None

        for r in config.CAMERA_ROLES:
            frame, intr = frames[r]
            if frame is None or intr is None:
                self.verify_video[r].show(frame, "no signal / not calibrated")
                continue
            disp = frame.copy()
            pc, pids, _ = self._probe_aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if pids is not None:
                cv2.aruco.drawDetectedMarkers(disp, pc, pids, (40, 160, 255))
            if boards.get(r) is not None:
                draw_board_pose(disp, boards[r], intr.mtx, intr.dist)
                if mv is not None:
                    draw_probe_in_view(disp, mv, boards[r], intr.mtx, intr.dist)
            self.verify_video[r].show(disp)

        def fmt(v):
            return f"[{v[0]:.0f}, {v[1]:.0f}, {v[2]:.0f}]"

        role_view = {vr: v for vr, v in zip(view_roles, views)}
        lines = []
        for r in config.CAMERA_ROLES:
            short = config.ROLE_LABEL[r].split(' (')[0]
            b = 'board OK' if boards.get(r) is not None else 'no board'
            fit = ""
            v = role_view.get(r)
            if v is not None and v.n_faces >= 2:
                e = single_view_reproj(v)
                if e is not None:
                    fit = f", fit {e:.1f}px"
            lines.append(f"{short}: {b}, {faces[r]}f cube{fit}")

        if mv is not None:
            hole = self._hole_xyz()
            err = float(np.linalg.norm(mv.tip_model - hole))
            self._last_error = err if mv.n_cameras == 2 else None
            pc = "  ".join(f"{config.ROLE_LABEL[vr].split(' (')[0]} {e:.1f}px"
                           for vr, e in zip(view_roles, mv.per_cam_reproj))
            lines.append(f"tip {fmt(mv.tip_model)}   joint reproj {mv.reproj_px:.1f}px   ({pc})")
            lines.append(f"known hole {fmt(hole)}   tip error {err:.0f} mm "
                         f"(target \u2264 {config.TIP_ERROR_TOLERANCE_MM:.0f})")
            ready = (mv.n_cameras == 2 and err <= config.TIP_ERROR_TOLERANCE_MM
                     and mv.reproj_px <= 4.0)
            self.btn_mark.state(["!disabled"] if ready else ["disabled"])
            self.verify_readout.configure(text="\n".join(lines))
        else:
            self._last_error = None
            self.verify_readout.configure(
                text="Place the probe tip in the calibration hole, with both cameras "
                     "seeing the board and the probe cube.")
            self.btn_mark.state(["disabled"])

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def _close_all_streams(self):
        for r in config.CAMERA_ROLES:
            if self.streams.get(r) is not None:
                self.streams[r].release()
                self.streams[r] = None

    def on_show(self):
        self._scan_devices()
        self._refresh_calib_status()
        self._on_tab_change()
        if self._tick_id is None:
            self._tick()

    def on_hide(self):
        if self._tick_id is not None:
            self.after_cancel(self._tick_id)
            self._tick_id = None
        self._close_all_streams()
