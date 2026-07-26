"""
ui/app.py — FluoroSim main application window.

Layout
------
┌──────────────────────────────────────────────────────────────┐
│  Toolbar:  [Start] [Stop] [Render BG] [Snapshot] [Status]     │
├───────────────────────────┬──────────────────────────────────┤
│  Camera A preview         │   AP synthetic X-ray + overlay    │
│  Camera B preview         │   LAT synthetic X-ray + overlay   │
└───────────────────────────┴──────────────────────────────────┘

Live loop (5 Hz):
  grab frames → detect board + probe (both cams) → fuse to world pose →
  overlay probe on cached AP/LAT backgrounds → display.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

import config as cfg
from core.camera import CameraStream, load_intrinsics
from core.markers import BoardTracker, ProbeTracker
from core.navigation import NavigationEngine
from core.xray_sim import XRaySimulator, GVXR_AVAILABLE

logger = logging.getLogger(__name__)


class FluoroSimApp:
    def __init__(self, root: tk.Tk):
        self._root = root
        root.title(f"{cfg.APP_TITLE}  v{cfg.APP_VERSION}")
        root.configure(bg="#1e1e1e")

        # ── Backend state ─────────────────────────────────────────────
        self._cam_a: Optional[CameraStream] = None
        self._cam_b: Optional[CameraStream] = None
        self._mtx_a = self._dist_a = None
        self._mtx_b = self._dist_b = None

        self._board_tracker = BoardTracker()
        self._probe_tracker = ProbeTracker()
        # BOARD_TO_SPINE, not BOARD_TO_WORLD: the gVXR render frame is
        # spine-local (see config.py's derived-transforms section). FusedPose's
        # .tip_world / .base_world are therefore points in the RENDER frame,
        # which is exactly what render_snapshot_with_probe() expects.
        self._nav  = NavigationEngine(cfg.BOARD_TO_SPINE)
        self._sim  = XRaySimulator(cfg)

        self._ap_bg:  Optional[np.ndarray] = None
        self._lat_bg: Optional[np.ndarray] = None
        self._running = False
        self._sim_ready = False

        self._build_ui()
        self._load_calibration()

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self):
        # Toolbar
        bar = tk.Frame(self._root, bg="#2d2d2d")
        bar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        self._btn_start = ttk.Button(bar, text="Start", command=self._on_start)
        self._btn_start.pack(side=tk.LEFT, padx=3)
        self._btn_stop  = ttk.Button(bar, text="Stop", command=self._on_stop, state=tk.DISABLED)
        self._btn_stop.pack(side=tk.LEFT, padx=3)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self._btn_render = ttk.Button(bar, text="Render X-ray Background",
                                      command=self._on_render_bg)
        self._btn_render.pack(side=tk.LEFT, padx=3)
        self._btn_snap = ttk.Button(bar, text="Photorealistic Snapshot",
                                    command=self._on_snapshot)
        self._btn_snap.pack(side=tk.LEFT, padx=3)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self._btn_align = ttk.Button(bar, text="Align Views to Probe",
                                     command=self._on_align_views)
        self._btn_align.pack(side=tk.LEFT, padx=3)

        self._status = tk.StringVar(value="Ready.")
        tk.Label(bar, textvariable=self._status, bg="#2d2d2d", fg="#ddd",
                 font=("Segoe UI", 10)).pack(side=tk.RIGHT, padx=10)

        # Main split
        main = tk.Frame(self._root, bg="#1e1e1e")
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left column: camera previews
        left = tk.Frame(main, bg="#1e1e1e")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        tk.Label(left, text="Camera A", bg="#1e1e1e", fg="#aaa").pack()
        self._lbl_cam_a = tk.Label(left, bg="black")
        self._lbl_cam_a.pack(pady=(0, 8))
        tk.Label(left, text="Camera B", bg="#1e1e1e", fg="#aaa").pack()
        self._lbl_cam_b = tk.Label(left, bg="black")
        self._lbl_cam_b.pack()

        # Right column: X-ray panels
        right = tk.Frame(main, bg="#1e1e1e")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(right, text="AP (Anteroposterior)", bg="#1e1e1e", fg="#aaa").pack()
        self._lbl_ap = tk.Label(right, bg="black")
        self._lbl_ap.pack(pady=(0, 8))
        tk.Label(right, text="Lateral", bg="#1e1e1e", fg="#aaa").pack()
        self._lbl_lat = tk.Label(right, bg="black")
        self._lbl_lat.pack()

        # Show placeholder X-ray panels at startup
        placeholder = np.full((cfg.XRAY_PANEL_H, cfg.XRAY_PANEL_W), 28, np.uint8)
        cv2.putText(placeholder, "Render background to begin",
                    (60, cfg.XRAY_PANEL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 120, 2)
        self._show(self._lbl_ap,  placeholder)
        self._show(self._lbl_lat, placeholder)

        if not GVXR_AVAILABLE:
            self._status.set("gVXR not installed — X-ray simulation disabled.")
            self._btn_render.configure(state=tk.DISABLED)
            self._btn_snap.configure(state=tk.DISABLED)

    # ── Calibration load ──────────────────────────────────────────────

    def _load_calibration(self):
        a = load_intrinsics(cfg.INTRINSICS_PATH_0)
        b = load_intrinsics(cfg.INTRINSICS_PATH_1)
        if a is None or b is None:
            self._status.set("Missing camera intrinsics — run tools/calibrate_intrinsics.py")
            return
        self._mtx_a, self._dist_a = a
        self._mtx_b, self._dist_b = b
        logger.info("Loaded intrinsics for both cameras.")

    # ── Toolbar actions ───────────────────────────────────────────────

    def _on_start(self):
        if self._mtx_a is None or self._mtx_b is None:
            messagebox.showwarning(
                "Calibration required",
                "Camera intrinsics not found.\n\n"
                "Run  tools/calibrate_intrinsics.py  for both cameras first.",
            )
            return
        self._cam_a = CameraStream(cfg.CAMERA_IDS[0], cfg.CAMERA_WIDTH,
                                   cfg.CAMERA_HEIGHT, cfg.CAMERA_FPS)
        self._cam_b = CameraStream(cfg.CAMERA_IDS[1], cfg.CAMERA_WIDTH,
                                   cfg.CAMERA_HEIGHT, cfg.CAMERA_FPS)
        if not self._cam_a.start() or not self._cam_b.start():
            messagebox.showerror("Camera error", "Could not open both cameras.")
            self._on_stop()
            return
        self._running = True
        self._btn_start.configure(state=tk.DISABLED)
        self._btn_stop.configure(state=tk.NORMAL)
        self._status.set("Tracking…")
        self._loop()

    def _on_stop(self):
        self._running = False
        for cam in (self._cam_a, self._cam_b):
            if cam is not None:
                cam.stop()
        self._cam_a = self._cam_b = None
        self._btn_start.configure(state=tk.NORMAL)
        self._btn_stop.configure(state=tk.DISABLED)
        self._status.set("Stopped.")

    def _on_render_bg(self):
        if not GVXR_AVAILABLE:
            return
        self._status.set("Loading synthetic X-ray background…")
        self._root.update_idletasks()
        if not self._sim_ready:
            if not self._sim.initialise():
                messagebox.showerror("gVXR error", "Failed to initialise X-ray simulator.")
                self._status.set("X-ray init failed.")
                return
            self._sim_ready = True

        # Cache-first: if the launcher already prepared this configuration the
        # images load from disk instantly and no GPU render happens at all.
        self._ap_bg, self._lat_bg = self._sim.load_cached_background()
        if self._ap_bg is None:
            self._status.set("Rendering synthetic X-ray background (first time)…")
            self._root.update_idletasks()
            self._ap_bg, self._lat_bg = self._sim.render_background()

        if self._ap_bg is None:
            self._status.set("Render failed.")
            return
        self._show(self._lbl_ap,  self._ap_bg)
        self._show(self._lbl_lat, self._lat_bg)
        self._status.set("Background ready. Start tracking to overlay the probe.")

    def _on_align_views(self):
        """Lock AP + lateral to the current probe trajectory (defines the
        reference insertion axis: probe → dot in AP, full line in lateral)."""
        if not (self._sim_ready and self._running):
            messagebox.showinfo(
                "Align views",
                "Start tracking and render the background first, then hold the "
                "probe along the ideal trajectory and click again.",
            )
            return
        pose = self._last_fused
        if pose is None or not pose.valid:
            messagebox.showinfo("Align views", "Probe not currently visible.")
            return
        self._status.set("Aligning AP/lateral to probe axis…")
        self._root.update_idletasks()
        self._sim.set_views_from_probe_axis(pose.tip_world, pose.base_world)
        self._ap_bg, self._lat_bg = self._sim.render_background()
        if self._ap_bg is not None:
            self._show(self._lbl_ap,  self._ap_bg)
            self._show(self._lbl_lat, self._lat_bg)
        self._status.set("Views aligned. AP looks down the probe axis; "
                         "lateral shows the full trajectory.")

    def _on_snapshot(self):
        """Render photorealistic AP+LAT with the K-wire as an actual cylinder."""
        if not (self._sim_ready and self._running):
            messagebox.showinfo(
                "Snapshot",
                "Start tracking and render the background first.",
            )
            return
        pose = self._last_fused
        if pose is None or not pose.valid:
            messagebox.showinfo("Snapshot", "Probe not currently visible.")
            return
        self._status.set("Rendering photorealistic snapshot…")
        self._root.update_idletasks()
        ap, lat = self._sim.render_snapshot_with_probe(pose.tip_world, pose.base_world)
        if ap is not None:
            self._show(self._lbl_ap,  ap)
            self._show(self._lbl_lat, lat)
        # Restore fast-overlay background afterwards
        self._sim.invalidate_background()
        self._ap_bg, self._lat_bg = self._sim.render_background()
        self._status.set("Snapshot complete.")

    # ── Live loop ─────────────────────────────────────────────────────

    _last_fused = None

    def _loop(self):
        if not self._running:
            return

        frame_a = self._cam_a.read() if self._cam_a else None
        frame_b = self._cam_b.read() if self._cam_b else None

        board_a = probe_a = board_b = probe_b = None

        if frame_a is not None:
            board_a = self._board_tracker.estimate(frame_a, self._mtx_a, self._dist_a)
            probe_a = self._probe_tracker.estimate(frame_a, self._mtx_a, self._dist_a)
            disp_a  = self._board_tracker.annotate(frame_a, board_a, self._mtx_a, self._dist_a)
            disp_a  = self._probe_tracker.annotate(disp_a, probe_a, self._mtx_a, self._dist_a)
            self._show(self._lbl_cam_a, disp_a, cfg.PREVIEW_W, cfg.PREVIEW_H)

        if frame_b is not None:
            board_b = self._board_tracker.estimate(frame_b, self._mtx_b, self._dist_b)
            probe_b = self._probe_tracker.estimate(frame_b, self._mtx_b, self._dist_b)
            disp_b  = self._board_tracker.annotate(frame_b, board_b, self._mtx_b, self._dist_b)
            disp_b  = self._probe_tracker.annotate(disp_b, probe_b, self._mtx_b, self._dist_b)
            self._show(self._lbl_cam_b, disp_b, cfg.PREVIEW_W, cfg.PREVIEW_H)

        # Fuse and overlay
        fused = self._nav.fuse(board_a, probe_a, board_b, probe_b)
        self._last_fused = fused

        if fused.valid and self._ap_bg is not None:
            ap_out  = self._sim.overlay_probe(
                self._ap_bg,  fused.tip_world, fused.base_world, view="AP",
                extend_mm=cfg.OVERLAY_EXTEND_MM,
            )
            lat_out = self._sim.overlay_probe(
                self._lat_bg, fused.tip_world, fused.base_world, view="LAT",
                extend_mm=cfg.OVERLAY_EXTEND_MM,
            )
            self._show(self._lbl_ap,  ap_out)
            self._show(self._lbl_lat, lat_out)
            self._status.set(
                f"Tracking — {fused.n_cameras} cam(s), "
                f"reproj {fused.mean_rms:.1f}px, "
                f"tip ({fused.tip_world[0]:.0f}, {fused.tip_world[1]:.0f}, "
                f"{fused.tip_world[2]:.0f}) mm"
            )
        elif self._ap_bg is not None:
            self._status.set("Tracking — probe not visible to both board + probe cameras.")

        self._root.after(cfg.UPDATE_INTERVAL_MS, self._loop)

    # ── Display helpers ───────────────────────────────────────────────

    def _show(self, label: tk.Label, img: np.ndarray,
              w: int = cfg.XRAY_PANEL_W, h: int = cfg.XRAY_PANEL_H):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        label.configure(image=photo)
        label.image = photo   # keep reference

    # ── Shutdown ──────────────────────────────────────────────────────

    def on_close(self):
        self._on_stop()
        if self._sim_ready:
            self._sim.shutdown()
        # True process exit: now it is safe to tear down the process-wide
        # gVXR OpenGL context (it cannot be recreated afterwards).
        try:
            XRaySimulator.destroy_context()
        except Exception:
            pass
        self._root.destroy()
