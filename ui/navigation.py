"""
ui/navigation.py — Simulated fluoroscopy training view.

This is the primary training screen.  On every tick:

  1. Grab frames from both cameras.
  2. PlatformBoardTracker detects the CharucoBoard on each frame →
     camera-to-model transform (xfm1, xfm2).
  3. ArucoTracker detects the probe cube on each frame →
     probe pose in camera space (det1, det2).
  4. fuse_poses() combines both camera estimates into a single model-space
     FusedPose.
  5. XRayOverlay.render() projects the fused pose onto the stored AP and
     LAT X-ray images.
  6. The annotated X-rays are displayed side-by-side.

The live camera feed is never shown.  The trainee sees only the stored
X-ray images with the trajectory overlay — matching the fluoroscopy
experience as closely as possible.
"""

import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import cv2
import numpy as np

from ui.widgets import (
    DarkFrame, primary_btn, success_btn,
    BG, BG2, FG, FG_MUTED, FG_SUCCESS, FG_ERR, FG_WARN,
    FONT_BODY, FONT_LABEL, FONT_TITLE, frame_to_tk,
)
from core.pose_fusion import fuse_poses
from config import XRAY_DISPLAY_W, XRAY_DISPLAY_H, NAV_UPDATE_MS, NAV_REALTIME


class NavigationView(DarkFrame):

    def __init__(self, parent, app, state, **kwargs):
        super().__init__(parent, **kwargs)
        self._app   = app
        self._state = state

        self._realtime   = NAV_REALTIME
        self._after_id   = None
        self._last_ap    = None
        self._last_lat   = None
        self._tk_img_ap  = None
        self._tk_img_lat = None

        self._build()

    # ── Build ────────────────────────────────────────────────────────────────

    def _build(self):
        # ── Top toolbar ──────────────────────────────────────────────────────
        top = tk.Frame(self, bg=BG2, pady=6)
        top.pack(fill=tk.X)

        tk.Label(top, text="Simulated Fluoroscopy",
                 font=FONT_TITLE, fg=FG, bg=BG2).pack(side=tk.LEFT, padx=14)

        # Right-side controls
        primary_btn(top, "↩ Models",
                    command=self._app.proceed_after_model_select, width=10
                    ).pack(side=tk.RIGHT, padx=6)
        primary_btn(top, "OR Setup",
                    command=self._app.go_to_or_setup, width=10
                    ).pack(side=tk.RIGHT, padx=6)
        success_btn(top, "💾 Save",
                    command=self._save, width=10
                    ).pack(side=tk.RIGHT, padx=6)
        self._capture_btn = primary_btn(top, "📷 Capture",
                                        command=self._capture, width=10)
        self._capture_btn.pack(side=tk.RIGHT, padx=6)

        # Real-time toggle
        self._mode_var = tk.BooleanVar(value=self._realtime)
        tk.Checkbutton(
            top, text="Live",
            variable=self._mode_var,
            font=FONT_LABEL, fg=FG_MUTED, bg=BG2,
            selectcolor=BG2, activebackground=BG2,
            command=self._toggle_mode,
        ).pack(side=tk.RIGHT, padx=10)

        # ── Status row ────────────────────────────────────────────────────────
        status_row = tk.Frame(self, bg=BG)
        status_row.pack(fill=tk.X, padx=12, pady=2)

        self._board_var  = tk.StringVar(value="Board: —")
        self._probe_var  = tk.StringVar(value="Probe: —")
        self._status_var = tk.StringVar(value="")

        tk.Label(status_row, textvariable=self._board_var,
                 font=FONT_LABEL, fg=FG_MUTED, bg=BG).pack(side=tk.LEFT, padx=10)
        tk.Label(status_row, textvariable=self._probe_var,
                 font=FONT_LABEL, fg=FG_MUTED, bg=BG).pack(side=tk.LEFT, padx=10)
        tk.Label(status_row, textvariable=self._status_var,
                 font=FONT_LABEL, fg=FG_WARN, bg=BG).pack(side=tk.RIGHT, padx=10)

        # ── X-ray display panels ──────────────────────────────────────────────
        display = tk.Frame(self, bg=BG)
        display.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        ap_col = tk.Frame(display, bg=BG2, padx=6, pady=6)
        ap_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8)
        tk.Label(ap_col, text="AP  (anteroposterior)",
                 font=FONT_BODY, fg=FG_MUTED, bg=BG2).pack()
        self._ap_lbl = tk.Label(ap_col, bg="#000")
        self._ap_lbl.pack(pady=4, fill=tk.BOTH, expand=True)

        lat_col = tk.Frame(display, bg=BG2, padx=6, pady=6)
        lat_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8)
        tk.Label(lat_col, text="Lateral",
                 font=FONT_BODY, fg=FG_MUTED, bg=BG2).pack()
        self._lat_lbl = tk.Label(lat_col, bg="#000")
        self._lat_lbl.pack(pady=4, fill=tk.BOTH, expand=True)

        # Model name strip
        self._model_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._model_var,
                 font=FONT_LABEL, fg=FG_MUTED, bg=BG2, pady=3
                 ).pack(fill=tk.X, side=tk.BOTTOM)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_show(self, **kwargs):
        model = self._state.model
        if model is None:
            messagebox.showwarning("No Model", "No model selected.")
            return

        self._model_var.set(f"Model: {model.name}")

        if not model.has_xrays:
            messagebox.showwarning("No X-rays",
                                   "This model has no X-ray images.  Run OR Setup first.")
        if not model.has_projection:
            messagebox.showwarning("No Projection",
                                   "Projection matrices missing.  Run OR Setup first.")

        self._show_blank_xrays()

        if self._realtime:
            self._start_realtime()
        self._update_capture_btn()

    def on_hide(self):
        self._stop_realtime()

    # ── Mode ─────────────────────────────────────────────────────────────────

    def _toggle_mode(self):
        self._realtime = self._mode_var.get()
        if self._realtime:
            self._start_realtime()
        else:
            self._stop_realtime()
        self._update_capture_btn()

    def _update_capture_btn(self):
        self._capture_btn.configure(
            state=tk.DISABLED if self._realtime else tk.NORMAL
        )

    # ── Real-time loop ────────────────────────────────────────────────────────

    def _start_realtime(self):
        self._status_var.set("Live")
        self._tick()

    def _stop_realtime(self):
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self._status_var.set("")

    def _tick(self):
        self._render_overlay()
        self._after_id = self.after(NAV_UPDATE_MS, self._tick)

    def _capture(self):
        self._status_var.set("Captured.")
        self._render_overlay()

    # ── Core render ───────────────────────────────────────────────────────────

    def _render_overlay(self):
        if self._state.overlay_ap is None or self._state.overlay_lat is None:
            return

        # ── Grab frames ──────────────────────────────────────────────────────
        frame1 = self._state.cap1.get_frame() if self._state.cap1 else None
        frame2 = self._state.cap2.get_frame() if self._state.cap2 else None

        # ── Board tracking → camera-to-model transforms ───────────────────────
        xfm1 = (self._state.board_tracker.estimate_pose(
                    frame1, self._state.mtx1, self._state.dist1)
                if frame1 is not None and self._state.mtx1 is not None else None)

        xfm2 = (self._state.board_tracker.estimate_pose(
                    frame2, self._state.mtx2, self._state.dist2)
                if frame2 is not None and self._state.mtx2 is not None else None)

        board_seen = sum(x is not None for x in (xfm1, xfm2))
        self._board_var.set(
            f"Board: {board_seen}/2 cameras" if board_seen
            else "Board: not visible"
        )

        # ── Probe detection ───────────────────────────────────────────────────
        det1 = (self._state.probe_tracker.detect(
                    frame1, self._state.mtx1, self._state.dist1)
                if frame1 is not None and self._state.mtx1 is not None else None)

        det2 = (self._state.probe_tracker.detect(
                    frame2, self._state.mtx2, self._state.dist2)
                if frame2 is not None and self._state.mtx2 is not None else None)

        # ── Pose fusion ───────────────────────────────────────────────────────
        fused = fuse_poses(det1, xfm1, det2, xfm2)

        if fused is not None:
            self._probe_var.set(
                f"Probe: {fused.confidence_label}  "
                f"| depth {fused.insertion_depth_mm:.0f} mm"
            )
        else:
            self._probe_var.set("Probe: not detected")

        # ── Render onto X-rays ────────────────────────────────────────────────
        img_ap  = self._state.overlay_ap.render(fused)
        img_lat = self._state.overlay_lat.render(fused)

        self._last_ap  = img_ap
        self._last_lat = img_lat
        self._display(img_ap,  self._ap_lbl,  "ap")
        self._display(img_lat, self._lat_lbl, "lat")

    def _display(self, img: np.ndarray, label: tk.Label, tag: str):
        try:
            tk_img = frame_to_tk(img, XRAY_DISPLAY_W, XRAY_DISPLAY_H)
            label.configure(image=tk_img)
            label.image = tk_img
            if tag == "ap":
                self._tk_img_ap  = tk_img
            else:
                self._tk_img_lat = tk_img
        except Exception:
            pass

    def _show_blank_xrays(self):
        model = self._state.model
        if model and model.xray_ap is not None:
            self._display(model.xray_ap,  self._ap_lbl,  "ap")
        if model and model.xray_lat is not None:
            self._display(model.xray_lat, self._lat_lbl, "lat")

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save(self):
        if self._last_ap is None:
            messagebox.showinfo("Save", "Nothing to save yet — capture first.")
            return
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        ap  = f"fluorosim_ap_{ts}.png"
        lat = f"fluorosim_lat_{ts}.png"
        cv2.imwrite(ap,  self._last_ap)
        cv2.imwrite(lat, self._last_lat)
        messagebox.showinfo("Saved", f"Images saved:\n  {ap}\n  {lat}")
