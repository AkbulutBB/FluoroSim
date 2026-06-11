"""
ui/screen_sim.py  —  Simulated fluoroscopy
===========================================

Every frame: each camera finds the board (-> camera->model transform) and the
probe cube (-> rod tip/base in camera space); the rod endpoints are carried into
model space, the cameras fused, and the fused tip/trajectory projected through
each X-ray's saved DLT matrix and drawn over the stored images.

Decoupling: the bottom feeds are TRACKING cameras (any angle that gives a good
solve); the big panels are the X-ray VIEWS (true AP, true lateral), each with
its own matrix.  Camera angle and X-ray view are independent.

Feeds are annotated for diagnosis: green = board markers, orange = probe-cube
markers, RGB axes = solved poses, red line+dot = the rod the tracker computed.
If the red rod doesn't sit on the physical K-wire, the probe pose is wrong.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

import config
from core import markers
from core.camera_io import CameraStream
from core.tracking import (BoardTracker, ProbeTracker, CameraView,
                           solve_probe_multiview, draw_board_pose, draw_probe_in_view)
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
        self._board_aruco = cv2.aruco.ArucoDetector(markers.CHARUCO_DICTIONARY)
        self._probe_aruco = markers.make_aruco_detector()
        self._xray = {r: None for r in config.CAMERA_ROLES}
        self._sm_tip = None     # smoothed model-space tip/base (reduces flicker)
        self._sm_base = None

        self._build_header()

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

    # ── feed annotation ───────────────────────────────────────────────────────
    def _annotate(self, frame):
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
        if not (-2 * w <= p[0] <= 3 * w and -2 * h <= p[1] <= 3 * h):
            return None
        return tuple(p.astype(int))

    def _draw_overlay(self, xray_bgr, P, base_model, tip_model):
        img = xray_bgr.copy()
        pb = self._safe_project(P, base_model, img.shape)   # cube/back end
        pt = self._safe_project(P, tip_model, img.shape)    # pointy/working end
        if pb is None or pt is None:
            return img, False
        th = config.OVERLAY_THICKNESS_PX
        tr = config.OVERLAY_TIP_RADIUS_PX
        pb = np.array(pb, float); pt = np.array(pt, float)
        # shaft with dark outline for contrast on the grey X-ray
        cv2.line(img, tuple(pb.astype(int)), tuple(pt.astype(int)), (0, 0, 0), th + 4, cv2.LINE_AA)
        cv2.line(img, tuple(pb.astype(int)), tuple(pt.astype(int)), (60, 220, 60), th, cv2.LINE_AA)
        # CUBE end: open square (the handle/back of the probe)
        s = tr + 1
        cv2.rectangle(img, (int(pb[0]-s), int(pb[1]-s)), (int(pb[0]+s), int(pb[1]+s)), (0, 0, 0), 3)
        cv2.rectangle(img, (int(pb[0]-s), int(pb[1]-s)), (int(pb[0]+s), int(pb[1]+s)), (60, 220, 60), 1)
        # TIP end: a clean filled arrowhead pointing the way the probe travels
        d = pt - pb; n = np.linalg.norm(d)
        if n > 1e-3:
            u = d / n
            perp = np.array([-u[1], u[0]])
            L = tr * 2.6           # arrowhead length
            wdt = tr * 1.5         # half-width
            apex  = pt + u * tr
            left  = pt - u * (L - tr) + perp * wdt
            right = pt - u * (L - tr) - perp * wdt
            tri = np.array([apex, left, right], np.int32)
            cv2.fillConvexPoly(img, tri, (60, 60, 240), cv2.LINE_AA)   # red tip
            cv2.polylines(img, [tri], True, (0, 0, 0), 1, cv2.LINE_AA)  # outline
        return img, True

    # ── tick ─────────────────────────────────────────────────────────────────
    def _tick(self):
        model = self.session.model
        views, boards = [], {}
        board_seen = {r: False for r in config.CAMERA_ROLES}
        probe_faces = {r: 0 for r in config.CAMERA_ROLES}
        probe_markers = {r: 0 for r in config.CAMERA_ROLES}
        frames = {}

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
            board_seen[r] = board is not None
            if det is not None:
                probe_faces[r] = det[2]
            if board is not None and det is not None:
                R_mc = cv2.Rodrigues(board.rvec)[0]
                views.append(CameraView(R_mc, board.tvec.flatten(), intr.mtx, intr.dist,
                                        det[0], det[1], det[2]))

        mv = solve_probe_multiview(views) if views else None

        # temporal smoothing + outlier rejection to calm the flicker
        if mv is not None:
            if self._sm_tip is None:
                self._sm_tip, self._sm_base = mv.tip_model.copy(), mv.base_model.copy()
            else:
                jump = float(np.linalg.norm(mv.tip_model - self._sm_tip))
                if mv.reproj_px <= config.SMOOTH_TRUST_PX or jump <= config.SMOOTH_MAX_JUMP_MM:
                    a = config.SMOOTH_ALPHA
                    self._sm_tip  = a * self._sm_tip  + (1 - a) * mv.tip_model
                    self._sm_base = a * self._sm_base + (1 - a) * mv.base_model
                # else: treat as an outlier (likely a pose flip) and hold steady

        # draw feeds with overlays
        for r in config.CAMERA_ROLES:
            frame, intr = frames[r]
            if frame is None or intr is None:
                self.feed[r].show(frame, "no signal / not calibrated")
                continue
            _, probe_markers[r] = self._annotate(frame)
            if boards.get(r) is not None:
                draw_board_pose(frame, boards[r], intr.mtx, intr.dist)
                if mv is not None:
                    draw_probe_in_view(frame, mv, boards[r], intr.mtx, intr.dist)
            self.feed[r].show(frame)

        off_image = False
        for r in config.CAMERA_ROLES:
            base_img = self._xray.get(r)
            P = model.view(r).P if model else None
            if base_img is None or P is None:
                self.xray_panel[r].show(None, "no X-ray / not registered")
                continue
            if self._sm_tip is not None:
                shown, ok = self._draw_overlay(base_img, P, self._sm_base, self._sm_tip)
                off_image = off_image or (not ok)
            else:
                shown = base_img
            self.xray_panel[r].show(shown)

        self._update_status(board_seen, probe_faces, probe_markers, mv, off_image)
        self._tick_id = self.after(50, self._tick)

    def _update_status(self, board_seen, probe_faces, probe_markers, mv, off_image):
        lines = []
        for r in config.CAMERA_ROLES:
            b = "board OK" if board_seen[r] else "no board"
            if probe_faces[r]:
                f = f"cube {probe_faces[r]}f"
            elif probe_markers[r]:
                f = f"cube seen ({probe_markers[r]} mk)"
            else:
                f = "no cube"
            lines.append(f"{config.ROLE_LABEL[r]}: {b}, {f}")
        if mv is not None:
            t = mv.tip_model
            lines.append(f"tip(model): [{t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f}] mm "
                         f"({mv.n_cameras} cam, {mv.n_faces}f)")
            tag = "" if mv.reproj_px <= 3.0 else "  \u26a0 high"
            lines.append(f"solve reprojection: {mv.reproj_px:.1f} px{tag}")
            if off_image:
                lines.append("\u26a0 tip projects off-image - check fiducial frame")
        else:
            lines.append("probe not solved (need board + cube in a view)")
        self.status.configure(text="\n".join(lines))

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def on_show(self):
        model = self.session.model
        self._sm_tip = self._sm_base = None
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
