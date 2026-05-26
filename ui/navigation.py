"""
ui/navigation.py — Live simulated fluoroscopy navigation.

Displays the pre-acquired AP and LAT X-ray images with the probe
trajectory overlaid in real time.

Two modes:
  Real-time — overlay refreshes every NAV_UPDATE_MS milliseconds.
  Snapshot  — user presses Capture to take a single overlay image.

v1.1: ToolPanel integrated in right sidebar — allows switching between
      Standard Probe (calibration) and Custom Tool (chisel/awl) with
      live tip-distance editing.
"""

import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime

from ui.widgets import (
    DarkFrame, primary_btn, success_btn,
    BG, BG2, BG3, ACCENT, FG, FG_MUTED, FG_SUCCESS, FG_ERR, FG_WARN,
    FONT_BODY, FONT_LABEL, FONT_TITLE, frame_to_tk,
)
from ui.tool_panel import ToolPanel
from core.projection import XRayOverlay
import config
from config import (
    XRAY_DISPLAY_W, XRAY_DISPLAY_H,
    NAV_UPDATE_MS, NAV_REALTIME,
)


class NavigationView(DarkFrame):

    def __init__(self, parent, app, state, **kwargs):
        super().__init__(parent, **kwargs)
        self._app   = app
        self._state = state

        self._realtime    = NAV_REALTIME
        self._after_id    = None
        self._tk_img_ap   = None
        self._tk_img_lat  = None
        self._last_ap:  np.ndarray | None = None
        self._last_lat: np.ndarray | None = None

        self._build()

    # ── Build ───────────────────────────────────────────────────────────────

    def _build(self):
        # ── Top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=BG2, pady=6)
        top.pack(fill=tk.X)

        tk.Label(
            top, text="Simulated Fluoroscopy Navigation",
            font=FONT_TITLE, fg=FG, bg=BG2,
        ).pack(side=tk.LEFT, padx=14)

        # Recalibrate
        primary_btn(
            top, "↩  Recalibrate",
            command=self._recalibrate, width=14,
        ).pack(side=tk.RIGHT, padx=6)

        # Save
        primary_btn(
            top, "💾  Save image",
            command=self._save, width=14,
        ).pack(side=tk.RIGHT, padx=6)

        # Capture (snapshot mode only)
        self._capture_btn = primary_btn(
            top, "📷  Capture",
            command=self._capture, width=12,
        )
        self._capture_btn.pack(side=tk.RIGHT, padx=6)

        # Real-time toggle
        self._mode_var = tk.BooleanVar(value=self._realtime)
        tk.Checkbutton(
            top, text="Real-time mode",
            variable=self._mode_var,
            font=FONT_LABEL, fg=FG_MUTED, bg=BG2,
            selectcolor=BG3, activebackground=BG2,
            command=self._toggle_mode,
        ).pack(side=tk.RIGHT, padx=10)

        # ── Status row ───────────────────────────────────────────────────────
        status_row = tk.Frame(self, bg=BG)
        status_row.pack(fill=tk.X, padx=12, pady=4)

        self._status_var = tk.StringVar(value="Ready.")
        tk.Label(
            status_row, textvariable=self._status_var,
            font=FONT_LABEL, fg=FG_MUTED, bg=BG,
        ).pack(side=tk.LEFT)

        self._detect_var = tk.StringVar(value="Probe: not detected")
        self._detect_lbl = tk.Label(
            status_row, textvariable=self._detect_var,
            font=FONT_LABEL, fg=FG_MUTED, bg=BG,
        )
        self._detect_lbl.pack(side=tk.RIGHT, padx=10)

        # ── Main content: X-ray panels + right sidebar ───────────────────────
        content = tk.Frame(self, bg=BG)
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # AP panel
        ap_col = tk.Frame(content, bg=BG2, padx=6, pady=6)
        ap_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        tk.Label(
            ap_col, text="AP (Craniocaudal) View",
            font=FONT_BODY, fg=FG_MUTED, bg=BG2,
        ).pack()
        self._ap_lbl = tk.Label(ap_col, bg="#000",
                                 width=XRAY_DISPLAY_W, height=XRAY_DISPLAY_H)
        self._ap_lbl.pack(pady=4)

        # LAT panel
        lat_col = tk.Frame(content, bg=BG2, padx=6, pady=6)
        lat_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        tk.Label(
            lat_col, text="Lateral View",
            font=FONT_BODY, fg=FG_MUTED, bg=BG2,
        ).pack()
        self._lat_lbl = tk.Label(lat_col, bg="#000",
                                  width=XRAY_DISPLAY_W, height=XRAY_DISPLAY_H)
        self._lat_lbl.pack(pady=4)

        # ── Right sidebar ────────────────────────────────────────────────────
        sidebar = tk.Frame(content, bg=BG2, width=220)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 0))
        sidebar.pack_propagate(False)   # keep fixed width

        # Tool panel
        self._tool_panel = ToolPanel(
            sidebar,
            on_change=self._on_tool_changed,
        )
        self._tool_panel.pack(fill=tk.X, padx=4, pady=(8, 4))

        # Separator
        tk.Frame(sidebar, bg=BG3, height=1).pack(fill=tk.X, padx=8, pady=6)

        # Depth readout
        tk.Label(
            sidebar, text="INSERTION DEPTH",
            font=("Segoe UI", 8, "bold"), fg=FG_MUTED, bg=BG2,
        ).pack(anchor=tk.W, padx=8, pady=(4, 0))

        self._depth_var = tk.StringVar(value="— mm")
        tk.Label(
            sidebar, textvariable=self._depth_var,
            font=("Segoe UI", 20, "bold"), fg=FG_SUCCESS, bg=BG2,
        ).pack(anchor=tk.W, padx=8, pady=(2, 8))

        # Separator
        tk.Frame(sidebar, bg=BG3, height=1).pack(fill=tk.X, padx=8, pady=6)

        # Calibration reminder note
        tk.Label(
            sidebar,
            text="⚠  Always calibrate with\nStandard Probe before\nswitching to Custom Tool.",
            font=("Segoe UI", 8), fg=FG_WARN, bg=BG2,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=8, pady=4)

        # ── Bottom strip ─────────────────────────────────────────────────────
        bottom = tk.Frame(self, bg=BG2, pady=4)
        bottom.pack(fill=tk.X, side=tk.BOTTOM)

        self._bottom_info_var = tk.StringVar(value="")
        tk.Label(
            bottom, textvariable=self._bottom_info_var,
            font=FONT_LABEL, fg=FG_MUTED, bg=BG2,
        ).pack()

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def on_show(self, **kwargs):
        model = self._state.model

        if not model.has_xrays:
            messagebox.showwarning(
                "No X-rays",
                "This model has no X-ray images. Run OR Setup first.",
            )
            return

        if not model.has_projection:
            messagebox.showwarning(
                "No Projection Data",
                "Projection matrices are missing. Run OR Setup to compute them.\n"
                "X-rays will be displayed without tracking.",
            )

        # Ensure navigation starts with standard probe active —
        # the custom tool is only valid after the user deliberately selects it.
        self._tool_panel.set_tool("standard_probe")
        self._status_var.set("Ready — Standard Probe active.")
        self._show_blank_xrays()

        if self._realtime:
            self._start_realtime()
        else:
            self._update_capture_btn_visibility()

    def on_hide(self):
        self._stop_realtime()

    # ── Tool change callback ─────────────────────────────────────────────────

    def _on_tool_changed(self, tool_key: str):
        """
        Called by ToolPanel whenever the active tool or tip distance changes.
        Updates the status bar and bottom info strip.
        """
        tool = config.get_active_tool()
        self._status_var.set(
            f"Active tool: {tool.name}  |  tip {tool.tip_distance_mm:.0f} mm from cube face"
        )
        self._bottom_info_var.set(
            f"{tool.name}  —  tip offset {tool.tip_distance_mm:.0f} mm"
        )

    # ── Mode management ─────────────────────────────────────────────────────

    def _toggle_mode(self):
        self._realtime = self._mode_var.get()
        if self._realtime:
            self._stop_realtime()
            self._start_realtime()
        else:
            self._stop_realtime()
        self._update_capture_btn_visibility()

    def _update_capture_btn_visibility(self):
        state = tk.DISABLED if self._realtime else tk.NORMAL
        self._capture_btn.configure(state=state)

    # ── Real-time loop ───────────────────────────────────────────────────────

    def _start_realtime(self):
        self._status_var.set("Real-time mode active.")
        self._tick()

    def _stop_realtime(self):
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _tick(self):
        self._render_overlay()
        self._after_id = self.after(NAV_UPDATE_MS, self._tick)

    # ── Snapshot ─────────────────────────────────────────────────────────────

    def _capture(self):
        self._status_var.set("Capturing…")
        self._render_overlay()
        self._status_var.set("Captured.")

    # ── Core render ──────────────────────────────────────────────────────────

    def _render_overlay(self):
        if self._state.overlay_ap is None or self._state.overlay_lat is None:
            return

        frame_ap  = self._state.cap_ap.get_frame()  if self._state.cap_ap  else None
        frame_lat = self._state.cap_lat.get_frame() if self._state.cap_lat else None

        det_ap  = None
        det_lat = None

        if frame_ap is not None and self._state.mtx_ap is not None:
            det_ap = self._state.tracker.detect(
                frame_ap, self._state.mtx_ap, self._state.dist_ap,
            )
        if frame_lat is not None and self._state.mtx_lat is not None:
            det_lat = self._state.tracker.detect(
                frame_lat, self._state.mtx_lat, self._state.dist_lat,
            )

        # Detection status label
        if det_ap is not None or det_lat is not None:
            self._detect_var.set(
                f"Probe detected  (AP: {'✔' if det_ap else '✘'}  "
                f"LAT: {'✔' if det_lat else '✘'})"
            )
            self._detect_lbl.configure(fg=FG_SUCCESS)
        else:
            self._detect_var.set("Probe: not detected")
            self._detect_lbl.configure(fg=FG_ERR)

        # Depth readout from whichever camera detected the probe
        det_for_depth = det_ap or det_lat
        if det_for_depth is not None and self._state.xfm_ap is not None:
            tip   = self._state.xfm_ap.point_cam_to_model(det_for_depth.rod_tip_cam)
            base  = self._state.xfm_ap.point_cam_to_model(det_for_depth.rod_base_cam)
            depth = float(np.linalg.norm(tip - base))
            self._depth_var.set(f"{depth:.1f} mm")
        else:
            self._depth_var.set("— mm")

        # Render overlays
        img_ap  = self._state.overlay_ap.render(det_ap,  self._state.xfm_ap)
        img_lat = self._state.overlay_lat.render(
            det_lat if det_lat is not None else det_ap,
            self._state.xfm_lat,
        )

        self._last_ap  = img_ap
        self._last_lat = img_lat
        self._display_xray(img_ap,  self._ap_lbl,  "ap")
        self._display_xray(img_lat, self._lat_lbl, "lat")

    def _display_xray(self, img: np.ndarray, label: tk.Label, tag: str):
        try:
            tk_img = frame_to_tk(img, XRAY_DISPLAY_W, XRAY_DISPLAY_H)
            label.configure(image=tk_img)
            label.image = tk_img
            if tag == "ap":
                self._tk_img_ap = tk_img
            else:
                self._tk_img_lat = tk_img
        except Exception:
            pass

    def _show_blank_xrays(self):
        model = self._state.model
        if model.xray_ap is not None:
            self._display_xray(model.xray_ap, self._ap_lbl, "ap")
        if model.xray_lat is not None:
            self._display_xray(model.xray_lat, self._lat_lbl, "lat")

    # ── Utilities ────────────────────────────────────────────────────────────

    def _save(self):
        if self._last_ap is None:
            messagebox.showinfo("Save", "No image to save yet — capture first.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"fluorosim_AP_{ts}.png",  self._last_ap)
        cv2.imwrite(f"fluorosim_LAT_{ts}.png", self._last_lat)
        self._status_var.set(
            f"Saved:  fluorosim_AP_{ts}.png  and  fluorosim_LAT_{ts}.png"
        )

    def _recalibrate(self):
        self._stop_realtime()
        self._app.show_view("model_calib")
