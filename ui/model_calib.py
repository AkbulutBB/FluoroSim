"""
ui/model_calib.py — Step 4: Two-slot probe registration.

Guides the user through placing the probe in each calibration slot.
For each slot:
  1. Shows live detection from both cameras.
  2. Highlights when the ArUco cube is detected reliably.
  3. User clicks Confirm to record the detection.

After both slots are confirmed, CalibrationEngine computes T_cam_model
for each camera.
"""

import tkinter as tk
from tkinter import messagebox
import threading
import numpy as np

from ui.widgets import (
    DarkFrame, CameraPreview, primary_btn, success_btn,
    section_label, info_label, StatusLight,
    BG, BG2, BG3, FG, FG_MUTED, FG_SUCCESS, FG_ERR, FG_WARN,
    FONT_BODY, FONT_LABEL, FONT_TITLE,
)
from core.tracker import ArucoTracker, ProbeDetection
from core.calibration import CalibrationEngine, SlotDefinition
from config import PREVIEW_W, PREVIEW_H


DETECT_STABLE_FRAMES = 5  # consecutive detections needed before enabling Confirm


class SlotPanel(tk.Frame):
    """
    Single-slot confirmation widget.
    Shows slot name, detection indicator, and a Confirm button.
    """

    def __init__(self, parent, slot_label: str, on_confirm, **kwargs):
        super().__init__(parent, bg=BG2, padx=12, pady=10, **kwargs)
        self._on_confirm = on_confirm
        self._confirmed  = False

        tk.Label(self, text=slot_label, font=FONT_TITLE, fg=FG, bg=BG2).pack()

        self._status = StatusLight(self, "Detection: waiting")
        self._status.pack(pady=4)

        self._confirm_btn = success_btn(self, "✔  Confirm slot", command=self._do_confirm, width=18)
        self._confirm_btn.configure(state=tk.DISABLED)
        self._confirm_btn.pack(pady=6)

        self._note = tk.Label(self, text="", font=FONT_LABEL, fg=FG_SUCCESS, bg=BG2)
        self._note.pack()

    def update_detection(self, detected: bool, stable: bool):
        if self._confirmed:
            return
        if stable:
            self._status.set("ok", "Detection: stable ✔")
            self._confirm_btn.configure(state=tk.NORMAL)
        elif detected:
            self._status.set("warn", "Detection: unstable — hold still")
            self._confirm_btn.configure(state=tk.DISABLED)
        else:
            self._status.set("pending", "Detection: waiting")
            self._confirm_btn.configure(state=tk.DISABLED)

    def _do_confirm(self):
        self._confirmed = True
        self._confirm_btn.configure(state=tk.DISABLED, text="Confirmed")
        self._note.configure(text="Slot recorded.")
        self._status.set("ok", "Confirmed")
        self._on_confirm()


class ModelCalibView(DarkFrame):

    def __init__(self, parent, app, state, **kwargs):
        super().__init__(parent, **kwargs)
        self._app   = app
        self._state = state

        self._tracker = ArucoTracker()
        self._current_slot = 0

        # Rolling detection buffers for stability check
        self._det_buf_ap : list[bool] = []
        self._det_buf_lat: list[bool] = []

        # Latest detections
        self._latest_det_ap : ProbeDetection | None = None
        self._latest_det_lat: ProbeDetection | None = None

        # Slot panels (created in on_show after model is known)
        self._slot_panels: list[SlotPanel] = []

        self._build()

    # ── Build ───────────────────────────────────────────────────────────────

    def _build(self):
        hdr = tk.Frame(self, bg=BG2)
        hdr.pack(fill=tk.X, pady=(0, 8))
        section_label(hdr, "Step 4 — Model Calibration").pack(pady=10)

        self._instr_var = tk.StringVar(value="")
        tk.Label(hdr, textvariable=self._instr_var,
                 font=FONT_LABEL, fg=FG_MUTED, bg=BG2,
                 wraplength=900).pack(pady=(0, 8))

        # Slot panels row (filled on_show)
        self._slot_row = tk.Frame(self, bg=BG)
        self._slot_row.pack(fill=tk.X, padx=20, pady=8)

        # Camera previews
        prev_row = tk.Frame(self, bg=BG)
        prev_row.pack(fill=tk.BOTH, expand=True, pady=4)

        self._prev_ap  = CameraPreview(prev_row, "AP Camera",  width=560, height=400)
        self._prev_ap.pack(side=tk.LEFT, padx=8, fill=tk.BOTH, expand=True)
        self._prev_lat = CameraPreview(prev_row, "LAT Camera", width=560, height=400)
        self._prev_lat.pack(side=tk.LEFT, padx=8, fill=tk.BOTH, expand=True)

        # Proceed button (hidden until all slots confirmed)
        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(pady=10)
        self._proceed_btn = success_btn(btn_row, "Calculate transforms →",
                                         command=self._calculate, width=24)
        self._proceed_btn.configure(state=tk.DISABLED)
        self._proceed_btn.pack()

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def on_show(self, **kwargs):
        model  = self._state.model
        slots  = model.slots
        n      = len(slots)

        # Build slot panels
        for w in self._slot_row.winfo_children():
            w.destroy()
        self._slot_panels.clear()
        self._confirmed_slots: dict[int, bool] = {i: False for i in range(n)}

        for i, slot_def in enumerate(slots):
            def _make_confirm_cb(idx=i):
                return lambda: self._on_slot_confirmed(idx)

            panel = SlotPanel(self._slot_row, slot_def.label, on_confirm=_make_confirm_cb())
            panel.pack(side=tk.LEFT, padx=10, fill=tk.Y)
            self._slot_panels.append(panel)

        self._current_slot = 0
        self._update_instructions()

        # Clear detection buffers
        self._det_buf_ap  = []
        self._det_buf_lat = []

        self._prev_ap.start(
            source_fn  = self._state.cap_ap.get_frame,
            process_fn = self._process_ap,
        )
        self._prev_lat.start(
            source_fn  = self._state.cap_lat.get_frame,
            process_fn = self._process_lat,
        )

    def on_hide(self):
        self._prev_ap.stop()
        self._prev_lat.stop()

    # ── Processing ──────────────────────────────────────────────────────────

    def _process_ap(self, frame):
        det = self._tracker.detect(frame, self._state.mtx_ap, self._state.dist_ap)
        self._latest_det_ap = det
        self._det_buf_ap.append(det is not None)
        if len(self._det_buf_ap) > DETECT_STABLE_FRAMES:
            self._det_buf_ap.pop(0)
        stable = len(self._det_buf_ap) == DETECT_STABLE_FRAMES and all(self._det_buf_ap)
        self._update_slot_ui(det is not None, stable)
        return self._tracker.annotate(frame, det, self._state.mtx_ap, self._state.dist_ap)

    def _process_lat(self, frame):
        det = self._tracker.detect(frame, self._state.mtx_lat, self._state.dist_lat)
        self._latest_det_lat = det
        self._det_buf_lat.append(det is not None)
        if len(self._det_buf_lat) > DETECT_STABLE_FRAMES:
            self._det_buf_lat.pop(0)
        return self._tracker.annotate(frame, det, self._state.mtx_lat, self._state.dist_lat)

    def _update_slot_ui(self, detected: bool, stable: bool):
        if self._current_slot < len(self._slot_panels):
            # Stability requires both cameras to see the probe
            buf_lat = self._det_buf_lat[-DETECT_STABLE_FRAMES:]
            lat_stable = len(buf_lat) == DETECT_STABLE_FRAMES and all(buf_lat)
            self._slot_panels[self._current_slot].update_detection(detected, stable and lat_stable)

    def _update_instructions(self):
        n = len(self._state.model.slots)
        slot = self._state.model.slots[self._current_slot]
        self._instr_var.set(
            f"Place the probe firmly into  {slot.label}  and hold it still. "
            f"Ensure the ArUco cube faces are clearly visible to both cameras, "
            f"then click Confirm."
        )

    # ── Slot confirmation ───────────────────────────────────────────────────

    def _on_slot_confirmed(self, slot_idx: int):
        det_ap  = self._latest_det_ap
        det_lat = self._latest_det_lat

        if det_ap is None or det_lat is None:
            messagebox.showwarning(
                "Detection Missing",
                "Probe was not visible in one or both cameras at the moment of "
                "confirmation. Re-seat the probe and try again.",
            )
            return

        self._state.cal_engine_ap.record_slot(slot_idx, det_ap)
        self._state.cal_engine_lat.record_slot(slot_idx, det_lat)
        self._confirmed_slots[slot_idx] = True

        # Advance to next slot
        next_idx = slot_idx + 1
        if next_idx < len(self._state.model.slots):
            self._current_slot = next_idx
            self._det_buf_ap  = []
            self._det_buf_lat = []
            self._update_instructions()
        else:
            # All slots confirmed
            self._proceed_btn.configure(state=tk.NORMAL)
            self._instr_var.set(
                "All slots confirmed. Click 'Calculate transforms' to proceed."
            )

    def _calculate(self):
        try:
            xfm_ap  = self._state.cal_engine_ap.compute()
            xfm_lat = self._state.cal_engine_lat.compute()
        except ValueError as exc:
            messagebox.showerror("Calibration Error", str(exc))
            return

        self._state.xfm_ap  = xfm_ap
        self._state.xfm_lat = xfm_lat
        self._app.proceed_after_model_calib()
