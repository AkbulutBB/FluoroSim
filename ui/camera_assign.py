"""
ui/camera_assign.py — Step 1: Assign the two cranial cameras.

Shows a live preview from every detected USB camera.
The user labels one as "Cranial (straight down)" and one as "Oblique (45°)".
Once both are assigned the Next button becomes active.
"""

import tkinter as tk
from tkinter import messagebox
from typing import Optional

from ui.widgets import (
    DarkFrame, CameraPreview, primary_btn, success_btn,
    info_label, section_label,
    BG, BG2, FG, FG_MUTED, FG_SUCCESS, FG_ERR, FONT_BODY, FONT_LABEL,
)
from core.camera import CameraCapture, list_available_cameras
from config import PREVIEW_W, PREVIEW_H


class CameraAssignView(DarkFrame):

    def __init__(self, parent, app, state, **kwargs):
        super().__init__(parent, **kwargs)
        self._app   = app
        self._state = state

        self._scanners: dict[int, CameraCapture] = {}
        self._assigned: dict[str, Optional[int]] = {"cam1": None, "cam2": None}
        self._previews: dict[int, CameraPreview] = {}

        self._build()

    # ── Build ────────────────────────────────────────────────────────────────

    def _build(self):
        hdr = tk.Frame(self, bg=BG2, pady=10)
        hdr.pack(fill=tk.X)

        section_label(hdr, "Step 1 — Assign Cameras").pack(pady=6)
        info_label(
            hdr,
            "Connect both USB webcams.  Assign Camera 1 to the cranial camera "
            "(mounted straight down) and Camera 2 to the oblique camera (45°).  "
            "Both cameras should be mounted on the same cranial frame above the platform.",
            color=FG_MUTED,
        ).pack(padx=20, pady=(0, 6))

        self._status_var = tk.StringVar(value="Scanning for cameras…")
        tk.Label(self, textvariable=self._status_var,
                 font=FONT_LABEL, fg=FG_MUTED, bg=BG).pack(pady=2)

        # Assignment indicators
        ind = tk.Frame(self, bg=BG)
        ind.pack(pady=4)
        self._cam1_var = tk.StringVar(value="Camera 1 (cranial):   not assigned")
        self._cam2_var = tk.StringVar(value="Camera 2 (oblique):   not assigned")
        tk.Label(ind, textvariable=self._cam1_var,
                 font=FONT_BODY, fg=FG_MUTED, bg=BG).pack(side=tk.LEFT, padx=20)
        tk.Label(ind, textvariable=self._cam2_var,
                 font=FONT_BODY, fg=FG_MUTED, bg=BG).pack(side=tk.LEFT, padx=20)

        # Preview row (populated dynamically)
        self._preview_row = tk.Frame(self, bg=BG)
        self._preview_row.pack(fill=tk.BOTH, expand=True, pady=8)

        # Bottom buttons
        btns = tk.Frame(self, bg=BG)
        btns.pack(pady=12)
        primary_btn(btns, "↻ Rescan",
                    command=self._rescan, width=16).pack(side=tk.LEFT, padx=8)
        self._next_btn = success_btn(btns, "Next →",
                                     command=self._proceed, width=12)
        self._next_btn.configure(state=tk.DISABLED)
        self._next_btn.pack(side=tk.LEFT, padx=8)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_show(self, **kwargs):
        self._rescan()

    def on_hide(self):
        self._stop_scanners()

    # ── Camera scanning ──────────────────────────────────────────────────────

    def _rescan(self):
        self._stop_scanners()
        for w in self._preview_row.winfo_children():
            w.destroy()
        self._previews.clear()
        self._assigned = {"cam1": None, "cam2": None}
        self._refresh_labels()

        indices = list_available_cameras()
        if not indices:
            self._status_var.set("No cameras found — check USB connections and rescan.")
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

        prev = CameraPreview(col, label=f"Camera {idx}",
                             width=320, height=240)
        prev.pack()
        prev.start(source_fn=self._scanners[idx].get_frame)
        self._previews[idx] = prev

        btn_row = tk.Frame(col, bg=BG2)
        btn_row.pack(fill=tk.X, pady=4)

        primary_btn(btn_row, "Cranial ↓",
                    command=lambda i=idx: self._assign(i, "cam1"),
                    width=12).pack(side=tk.LEFT, padx=3)
        primary_btn(btn_row, "Oblique 45°",
                    command=lambda i=idx: self._assign(i, "cam2"),
                    width=12).pack(side=tk.LEFT, padx=3)

    def _assign(self, cam_idx: int, role: str):
        self._assigned[role] = cam_idx
        self._refresh_labels()

    def _refresh_labels(self):
        c1 = self._assigned["cam1"]
        c2 = self._assigned["cam2"]

        self._cam1_var.set(
            f"Camera 1 (cranial):   Camera {c1}" if c1 is not None
            else "Camera 1 (cranial):   not assigned"
        )
        self._cam2_var.set(
            f"Camera 2 (oblique):   Camera {c2}" if c2 is not None
            else "Camera 2 (oblique):   not assigned"
        )

        both = c1 is not None and c2 is not None
        self._next_btn.configure(state=tk.NORMAL if both else tk.DISABLED)

        for idx, prev in self._previews.items():
            if idx == c1 and idx == c2:
                prev.set_label(f"Camera {idx}  ← Cranial + Oblique")
                prev.set_status("Assigned to both roles", FG_ERR)
            elif idx == c1:
                prev.set_label(f"Camera {idx}  ← Cranial ↓")
                prev.set_status("Assigned: cranial", FG_SUCCESS)
            elif idx == c2:
                prev.set_label(f"Camera {idx}  ← Oblique 45°")
                prev.set_status("Assigned: oblique", FG_SUCCESS)
            else:
                prev.set_label(f"Camera {idx}")
                prev.set_status("Unassigned")

    def _proceed(self):
        c1 = self._assigned["cam1"]
        c2 = self._assigned["cam2"]
        if c1 is None or c2 is None:
            messagebox.showwarning("Assignment", "Please assign both cameras before continuing.")
            return
        if c1 == c2:
            if not messagebox.askyesno(
                "Same camera",
                "Both roles are assigned to the same camera.\n"
                "This is only suitable for testing — not for training.\n\nContinue anyway?"
            ):
                return

        self._state.cam1_idx = c1
        self._state.cam2_idx = c2
        self._stop_scanners()
        for prev in self._previews.values():
            prev.stop()
        self._app.proceed_after_camera_assign()

    def _stop_scanners(self):
        for cap in self._scanners.values():
            cap.stop()
        self._scanners.clear()
