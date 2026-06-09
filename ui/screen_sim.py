"""
ui/screen_sim.py  —  Simulated fluoroscopy
===========================================

Every frame:

  * each camera finds the board (-> camera->model transform) and the probe cube
    (-> rod tip/base in camera space);
  * the rod endpoints are carried into model space and the cameras fused;
  * the fused tip and trajectory are projected through each X-ray's saved DLT
    matrix and drawn over the stored AP and lateral images.

IMPORTANT decoupling: the two camera feeds (bottom) are only there to fix the
probe in 3-D.  The two big panels are the X-ray VIEWS (true AP, true lateral),
each projected through its own matrix.  Camera angle and X-ray view are
independent -- a 45 deg camera still feeds a true-lateral overlay.

The live feeds are annotated: green boxes = ChArUco (board) markers, orange
boxes = probe-cube markers.  If you see no orange box on the cube, the probe is
not being detected (lighting / focus / distance / angle), which is why no
overlay appears -- fix that first.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

import config
from core import markers
from core.camera_io import CameraStream
from core.tracking import BoardTracker, ProbeTracker, fuse_model_points
from ui.app import Screen
from ui import widgets as W


class SimulationScreen(Screen):
    def __init__(self, master, app):
        super().__init__(master, style="App.TFrame")
        self.app = app
        self.session = app.session
        self.streams = {r: None for r in config.CAMERA_ROLES}
        self._tick_id = None
        self._board_tracker = BoardTracker()
        self._probe_tracker = ProbeTracker()
        # lightweight detectors used only to draw boxes on the feeds
        self._board_aruco = cv2.aruco.ArucoDetector(markers.CHARUCO_DICTIONARY)
        self._probe_aruco = markers.make_aruco_detector()
        self._xray = {r: None for r in config.CAMERA_ROLES}

        self._build_header()

        # main output: the two X-ray views with overlay
        out = ttk.Frame(self, style="App.TFrame")
        out.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self.xray_panel = {}
        for c, r in enumerate(config.CAMERA_ROLES):
            col = ttk.Frame(out, style="App.TFrame")
            col.grid(row=0, column=c, padx=10, sticky="nsew")
            ttk.Label(col, text=f"{config.XRAY_LABEL[r]} X-ray  \u2014  simulated fluoroscopy",
                      style="Muted.TLabel").pack()
            p = W.VideoPanel(col, max_w=520, max_h=440, style="Panel.TLabel")
            p.pack(fill="both", expand=True)
            self.xray_panel[r] = p
        out.columnconfigure(0, weight=1)
        out.columnconfigure(1, weight=1)

        # footer: small live feeds + status
        foot = ttk.Frame(self, style="App.TFrame")
        foot.pack(fill="x", padx=16, pady=(0, 12))
        self.feed = {}
        for c, r in enumerate(config.CAMERA_ROLES):
            col = ttk.Frame(foot, style="App.TFrame")
            col.grid(row=0, column=c, padx=6)
            ttk.Label(col, text=config.ROLE_LABEL[r], style="Muted.TLabel").pack()
            f = W.VideoPanel(col, max_w=220, max_h=165, style="Panel.TLabel")
            f.pack()
            self.feed[r] = f
        self.status = ttk.Label(foot, text="", style="Muted.TLabel", justify="left")
        self.status.grid(row=0, column=2, padx=18, sticky="w")

    def _build_header(self):
        bar = ttk.Frame(self, style="App.TFrame")
        bar.pack(fill="x", padx=16, pady=12)
        ttk.Button(bar, text="\u2190 Home", command=lambda: self.app.show("home")).pack(side="left")
        ttk.Label(bar, text="Simulation", style="H2.TLabel").pack(side="left", padx=14)

    # ── feed annotation (diagnostic) ─────────────────────────────────────────
    def _annotate(self, frame):
        """Draw detected board (green) and probe (orange) markers; return counts."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        n_board = n_probe = 0
        bc, bids, _ = self._board_aruco.detectMarkers(gray)
        if bids is not None:
            cv2.aruco.drawDetectedMarkers(frame, bc, bids, (60, 200, 60))
            n_board = len(bids)
        pc, pids, _ = self._probe_aruco.detectMarkers(gray)
        if pids is not None:
            cv2.aruco.drawDetectedMarkers(frame, pc, pids, (40, 160, 255))
            n_probe = len(pids)
        return n_board, n_probe

    # ── overlay drawing ──────────────────────────────────────────────────────
    @staticmethod
    def _safe_project(P, pt, shape):
        p = P.project(pt)
        if not np.all(np.isfinite(p)):
            return None
        h, w = shape[:2]
        # reject wildly off-image projections (symptom of a bad DLT / coords)
        if not (-2 * w <= p[0] <= 3 * w and -2 * h <= p[1] <= 3 * h):
            return None
        return tuple(p.astype(int))

    def _draw_overlay(self, xray_bgr, P, base_model, tip_model):
        img = xray_bgr.copy()
        pb = self._safe_project(P, base_model, img.shape)
        pt = self._safe_project(P, tip_model, img.shape)
        if pb is None or pt is None:
            return img, False
        cv2.line(img, pb, pt, (60, 220, 60), 3, cv2.LINE_AA)
        cv2.circle(img, pb, 7, (40, 200, 255), 2, cv2.LINE_AA)      # entry (yellow)
        cv2.circle(img, pt, 8, (60, 60, 240), -1, cv2.LINE_AA)      # tip (red)
        cv2.circle(img, pt, 8, (255, 255, 255), 1, cv2.LINE_AA)
        return img, True

    # ── tick ─────────────────────────────────────────────────────────────────
    def _tick(self):
        model = self.session.model
        tips, bases = [], []
        board_seen = {r: False for r in config.CAMERA_ROLES}
        probe_used = {r: 0 for r in config.CAMERA_ROLES}
        probe_markers = {r: 0 for r in config.CAMERA_ROLES}

        for r in config.CAMERA_ROLES:
            stream = self.streams.get(r)
            intr = self.session.intrinsics.get(r)
            frame = stream.read() if stream else None
            if frame is None or intr is None:
                self.feed[r].show(frame, "no signal / not calibrated")
                continue
            board = self._board_tracker.estimate(frame, intr.mtx, intr.dist)
            probe = self._probe_tracker.estimate(frame, intr.mtx, intr.dist)
            board_seen[r] = board is not None
            if board is not None and probe is not None:
                probe_used[r] = probe.n_faces
                tips.append(board.transform.apply(probe.rod_tip_cam))
                bases.append(board.transform.apply(probe.rod_base_cam))
            _, probe_markers[r] = self._annotate(frame)   # draw boxes for diagnosis
            self.feed[r].show(frame)

        fused = fuse_model_points(tips, bases) if tips else None
        off_image = False

        for r in config.CAMERA_ROLES:
            base_img = self._xray.get(r)
            P = model.view(r).P if model else None
            if base_img is None or P is None:
                self.xray_panel[r].show(None, "no X-ray / not registered")
                continue
            if fused is not None:
                shown, ok = self._draw_overlay(base_img, P, fused.base_model, fused.tip_model)
                off_image = off_image or (not ok)
            else:
                shown = base_img
            self.xray_panel[r].show(shown)

        self._update_status(board_seen, probe_used, probe_markers, fused, off_image)
        self._tick_id = self.after(50, self._tick)

    def _update_status(self, board_seen, probe_used, probe_markers, fused, off_image):
        lines = []
        for r in config.CAMERA_ROLES:
            b = "board OK" if board_seen[r] else "no board"
            if probe_used[r]:
                f = f"probe {probe_used[r]}f"
            elif probe_markers[r]:
                f = f"cube seen ({probe_markers[r]} mk) but not solved"
            else:
                f = "no cube"
            lines.append(f"{config.ROLE_LABEL[r]}: {b}, {f}")
        if fused is not None:
            t = fused.tip_model
            lines.append(f"tip(model): [{t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f}] mm "
                         f"({fused.n_cameras} cam)")
            if fused.agreement_mm is not None:
                tag = "" if fused.agreement_mm <= config.CAMERA_AGREEMENT_TOL_MM else "  \u26a0 high"
                lines.append(f"agreement: {fused.agreement_mm:.1f} mm{tag}")
            if off_image:
                lines.append("\u26a0 tip projects off-image - check fiducial coordinates")
        else:
            lines.append("probe not solved in any view (need board + cube together)")
        self.status.configure(text="\n".join(lines))

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def on_show(self):
        model = self.session.model
        for r in config.CAMERA_ROLES:
            self._xray[r] = cv2.imread(model.view(r).image_path) \
                if (model and model.view(r).image_path) else None
            idx = self.session.device_index.get(r)
            if idx is not None:
                self.streams[r] = CameraStream(idx).start()
        if self._tick_id is None:
            self._tick()

    def on_hide(self):
        if self._tick_id is not None:
            self.after_cancel(self._tick_id)
            self._tick_id = None
        for r in config.CAMERA_ROLES:
            if self.streams.get(r) is not None:
                self.streams[r].release()
                self.streams[r] = None
