"""
ui/intrinsic_calib.py — Step 2: Lens intrinsic calibration.

Guides the user through checkerboard calibration for Camera 1 (cranial)
and then Camera 2 (oblique).  Saved intrinsics are loaded automatically
on subsequent sessions — this step is skipped if both are already saved.
"""

import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np

from ui.widgets import (
    DarkFrame, CameraPreview, primary_btn, success_btn,
    info_label, section_label,
    BG, BG2, FG, FG_MUTED, FG_SUCCESS, FG_ERR, FG_WARN,
    FONT_BODY, FONT_LABEL, FONT_TITLE,
)
from core.camera import IntrinsicCalibrator, save_intrinsics
from config import INTRINSIC_CALIB_FRAMES, PREVIEW_W, PREVIEW_H


_CAMERA_LABELS = {
    1: "Camera 1 — Cranial (straight down)",
    2: "Camera 2 — Oblique (45°)",
}


class IntrinsicCalibView(DarkFrame):

    def __init__(self, parent, app, state, **kwargs):
        super().__init__(parent, **kwargs)
        self._app   = app
        self._state = state

        self._current_cam = 1   # 1 or 2
        self._calibrator: IntrinsicCalibrator = IntrinsicCalibrator()
        self._after_id = None

        self._build()

    # ── Build ────────────────────────────────────────────────────────────────

    def _build(self):
        hdr = tk.Frame(self, bg=BG2, pady=10)
        hdr.pack(fill=tk.X)
        section_label(hdr, "Step 2 — Camera Calibration").pack(pady=6)
        info_label(
            hdr,
            "Hold a printed checkerboard in front of each camera and move it "
            "slowly to different positions and angles.  The system collects "
            f"{INTRINSIC_CALIB_FRAMES} frames automatically.  "
            "Calibration is saved and this step is skipped in future sessions.",
            color=FG_MUTED,
        ).pack(padx=20, pady=(0, 6))

        self._cam_title = tk.StringVar(value="")
        tk.Label(self, textvariable=self._cam_title,
                 font=FONT_TITLE, fg=FG, bg=BG).pack(pady=4)

        self._progress_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._progress_var,
                 font=FONT_BODY, fg=FG_WARN, bg=BG).pack()

        self._preview = CameraPreview(self, width=PREVIEW_W, height=PREVIEW_H)
        self._preview.pack(pady=8)

        btns = tk.Frame(self, bg=BG)
        btns.pack(pady=10)
        primary_btn(btns, "↩ Back", command=self._app.go_home, width=12).pack(side=tk.LEFT, padx=8)
        self._compute_btn = success_btn(btns, "Compute & Save →",
                                        command=self._compute, width=18)
        self._compute_btn.configure(state=tk.DISABLED)
        self._compute_btn.pack(side=tk.LEFT, padx=8)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_show(self, **kwargs):
        # Determine which camera still needs calibration
        if self._state.mtx1 is None:
            self._start_camera(1)
        elif self._state.mtx2 is None:
            self._start_camera(2)
        else:
            self._app.proceed_after_intrinsic_calib()

    def on_hide(self):
        self._stop()

    # ── Camera loop ──────────────────────────────────────────────────────────

    def _start_camera(self, cam_num: int):
        self._current_cam = cam_num
        self._calibrator  = IntrinsicCalibrator()
        self._cam_title.set(_CAMERA_LABELS[cam_num])
        self._progress_var.set(f"0 / {INTRINSIC_CALIB_FRAMES} frames collected")
        self._compute_btn.configure(state=tk.DISABLED)
        self._preview.start(source_fn=self._get_annotated_frame)
        self._tick()

    def _get_cap(self):
        return self._state.cap1 if self._current_cam == 1 else self._state.cap2

    def _get_mtx(self):
        return (self._state.mtx1 if self._current_cam == 1 else self._state.mtx2,
                self._state.dist1 if self._current_cam == 1 else self._state.dist2)

    def _get_annotated_frame(self):
        """Return the most recently annotated frame for the preview widget."""
        return getattr(self, "_last_annotated", None)

    def _tick(self):
        cap = self._get_cap()
        if cap is None:
            self._after_id = self.after(50, self._tick)
            return

        frame = cap.get_frame()
        if frame is not None:
            annotated, accepted = self._calibrator.process_frame(frame)
            self._last_annotated = annotated

            n = self._calibrator.n_frames
            self._progress_var.set(f"{n} / {INTRINSIC_CALIB_FRAMES} frames collected")

            if self._calibrator.is_done:
                self._compute_btn.configure(state=tk.NORMAL)
                self._progress_var.set(
                    f"{INTRINSIC_CALIB_FRAMES} frames collected — click Compute & Save"
                )
                self._stop()
                return

        self._after_id = self.after(50, self._tick)

    def _stop(self):
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self._preview.stop()

    # ── Compute ──────────────────────────────────────────────────────────────

    def _compute(self):
        try:
            mtx, dist, rms = self._calibrator.compute()
        except RuntimeError as e:
            messagebox.showerror("Calibration Error", str(e))
            return

        cam_id = str(self._state.cam1_idx if self._current_cam == 1
                     else self._state.cam2_idx)
        save_intrinsics(cam_id, mtx, dist)

        if self._current_cam == 1:
            self._state.mtx1  = mtx
            self._state.dist1 = dist
        else:
            self._state.mtx2  = mtx
            self._state.dist2 = dist

        messagebox.showinfo(
            "Calibration Saved",
            f"Camera {self._current_cam} calibrated.\n"
            f"Reprojection RMS: {rms:.3f} px\n\n"
            f"{'Good — proceeding to Camera 2.' if self._current_cam == 1 else 'Both cameras calibrated.'}"
        )

        # Advance to next camera or next step
        if self._current_cam == 1 and self._state.mtx2 is None:
            self._start_camera(2)
        else:
            self._app.proceed_after_intrinsic_calib()
