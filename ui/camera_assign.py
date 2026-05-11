"""
ui/camera_assign.py — Step 1: Assign AP and LAT cameras.

Shows live feeds from all detected cameras.
User clicks "Set as AP" / "Set as LAT" under each feed.
Once both are labelled the Next button activates.
"""

import tkinter as tk
from tkinter import messagebox
from typing import Optional
import numpy as np
import cv2

from ui.widgets import (
    DarkFrame, CameraPreview, primary_btn, success_btn,
    section_label, info_label, BG, BG2, FG, FG_MUTED, FG_SUCCESS, FG_ERR,
    FONT_BODY, FONT_LABEL,
)
from core.camera import CameraCapture, list_available_cameras
from config import PREVIEW_W, PREVIEW_H


class CameraAssignView(DarkFrame):

    def __init__(self, parent, app, state, **kwargs):
        super().__init__(parent, **kwargs)
        self._app   = app
        self._state = state

        # Map camera index → CameraCapture (for scanning)
        self._scanners: dict[int, CameraCapture] = {}
        # Which index has been assigned to which role
        self._assigned: dict[str, Optional[int]] = {"ap": None, "lat": None}

        self._previews: dict[int, CameraPreview] = {}

        self._build()

    # ── Build ───────────────────────────────────────────────────────────────

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=BG2)
        hdr.pack(fill=tk.X, pady=(0, 8))
        section_label(hdr, "Step 1 — Assign Camera Views").pack(pady=10)
        info_label(
            hdr,
            "Connect both USB webcams, then assign which camera shows the "
            "AP (craniocaudal) view and which shows the lateral view. "
            "Look at each feed and click the appropriate button.",
            color=FG_MUTED,
        ).pack(pady=(0, 8))

        # Status bar
        self._status_var = tk.StringVar(value="Scanning for cameras…")
        tk.Label(self, textvariable=self._status_var,
                 font=FONT_LABEL, fg=FG_MUTED, bg=BG).pack()

        # Assignment indicators
        ind_row = tk.Frame(self, bg=BG)
        ind_row.pack(pady=4)
        self._ap_var  = tk.StringVar(value="AP View:   not assigned")
        self._lat_var = tk.StringVar(value="LAT View:  not assigned")
        tk.Label(ind_row, textvariable=self._ap_var,
                 font=FONT_BODY, fg=FG_MUTED, bg=BG).pack(side=tk.LEFT, padx=20)
        tk.Label(ind_row, textvariable=self._lat_var,
                 font=FONT_BODY, fg=FG_MUTED, bg=BG).pack(side=tk.LEFT, padx=20)

        # Preview container (filled dynamically)
        self._preview_row = tk.Frame(self, bg=BG)
        self._preview_row.pack(fill=tk.BOTH, expand=True, pady=8)

        # Bottom buttons
        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(pady=10)
        primary_btn(btn_row, "↻ Rescan cameras",
                    command=self._rescan, width=18).pack(side=tk.LEFT, padx=8)
        self._next_btn = success_btn(btn_row, "Next →",
                                     command=self._proceed, width=12)
        self._next_btn.configure(state=tk.DISABLED)
        self._next_btn.pack(side=tk.LEFT, padx=8)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def on_show(self, **kwargs):
        self._rescan()

    def on_hide(self):
        self._stop_scanners()

    # ── Camera scanning ─────────────────────────────────────────────────────

    def _rescan(self):
        self._stop_scanners()
        # Clear previews
        for w in self._preview_row.winfo_children():
            w.destroy()
        self._previews.clear()
        self._assigned = {"ap": None, "lat": None}
        self._refresh_labels()

        indices = list_available_cameras()
        if not indices:
            self._status_var.set("No cameras found. Check USB connections and rescan.")
            return

        self._status_var.set(f"Found {len(indices)} camera(s). Assign roles below.")

        for idx in indices:
            cap = CameraCapture(idx)
            if not cap.start():
                continue
            self._scanners[idx] = cap
            self._add_preview(idx)

    def _add_preview(self, idx: int):
        col = tk.Frame(self._preview_row, bg=BG2, padx=6, pady=6)
        col.pack(side=tk.LEFT, padx=8, fill=tk.BOTH, expand=True)

        prev = CameraPreview(col, label=f"Camera {idx}", width=320, height=240)
        prev.pack()
        prev.start(source_fn=self._scanners[idx].get_frame)
        self._previews[idx] = prev

        # Assignment buttons
        btn_row = tk.Frame(col, bg=BG2)
        btn_row.pack(fill=tk.X, pady=4)

        primary_btn(btn_row, "Set as AP",
                    command=lambda i=idx: self._assign(i, "ap"), width=10).pack(side=tk.LEFT, padx=3)
        primary_btn(btn_row, "Set as LAT",
                    command=lambda i=idx: self._assign(i, "lat"), width=10).pack(side=tk.LEFT, padx=3)
        primary_btn(btn_row, "Set as Both",
                    command=lambda i=idx: self._assign_both(i), width=10).pack(side=tk.LEFT, padx=3)

    def _assign(self, cam_idx: int, role: str):
        # Simply assign this camera to the role — do NOT un-assign from other roles
        self._assigned[role] = cam_idx
        print(f"[DEBUG] Assigned camera {cam_idx} as {role.upper()}")
        self._refresh_labels()

    def _assign_both(self, cam_idx: int):
        self._assigned["ap"]  = cam_idx
        self._assigned["lat"] = cam_idx
        print(f"[DEBUG] Assigned camera {cam_idx} as BOTH AP and LAT")
        self._refresh_labels()

    def _refresh_labels(self):
        ap_idx  = self._assigned["ap"]
        lat_idx = self._assigned["lat"]

        self._ap_var.set(
            f"AP View:   Camera {ap_idx}" if ap_idx is not None else "AP View:   not assigned"
        )
        self._lat_var.set(
            f"LAT View:  Camera {lat_idx}" if lat_idx is not None else "LAT View:  not assigned"
        )

        both_assigned = ap_idx is not None and lat_idx is not None
        self._next_btn.configure(
            state=tk.NORMAL if both_assigned else tk.DISABLED
        )
        # Warn if same camera used for both (allowed but suboptimal)
        if both_assigned and ap_idx == lat_idx:
            self._status_var.set("⚠ Same camera assigned to both views — acceptable for testing, use two cameras for training.")

        # Colour highlight on previews
        for idx, prev in self._previews.items():
            if idx == ap_idx:
                prev.set_label(f"Camera {idx}  ← AP")
                prev.set_status("Assigned as AP", FG_SUCCESS)
            elif idx == lat_idx:
                prev.set_label(f"Camera {idx}  ← LAT")
                prev.set_status("Assigned as LAT", FG_SUCCESS)
            else:
                prev.set_label(f"Camera {idx}")
                prev.set_status("Unassigned")

    def _proceed(self):
        ap_idx  = self._assigned["ap"]
        lat_idx = self._assigned["lat"]
        if ap_idx is None or lat_idx is None or ap_idx == lat_idx:
            messagebox.showwarning("Assignment", "Please assign two different cameras.")
            return
        # Pass indices to state
        self._state.cam_ap_idx  = ap_idx
        self._state.cam_lat_idx = lat_idx
        # Stop scanners — app will open dedicated captures
        self._stop_scanners()
        for prev in self._previews.values():
            prev.stop()
        self._app.proceed_after_camera_assign()

    def _stop_scanners(self):
        for cap in self._scanners.values():
            cap.stop()
        self._scanners.clear()
