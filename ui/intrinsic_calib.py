"""
ui/intrinsic_calib.py — Step 2: Per-camera intrinsic calibration.

Shows live checkerboard detection for both cameras simultaneously.
Collects INTRINSIC_CALIB_FRAMES diverse views then solves for
the lens intrinsics and saves them to disk.
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np

from ui.widgets import (
    DarkFrame, CameraPreview, primary_btn, success_btn,
    section_label, info_label, StatusLight,
    BG, BG2, FG, FG_MUTED, FG_SUCCESS, FG_WARN, FONT_LABEL, FONT_BODY,
)
from core.camera import IntrinsicCalibrator, save_intrinsics
from config import PREVIEW_W, PREVIEW_H, INTRINSIC_CALIB_FRAMES, CAMERAS_DIR


class IntrinsicCalibView(DarkFrame):

    def __init__(self, parent, app, state, **kwargs):
        super().__init__(parent, **kwargs)
        self._app   = app
        self._state = state

        self._cal_ap  = IntrinsicCalibrator()
        self._cal_lat = IntrinsicCalibrator()

        self._build()

    # ── Build ───────────────────────────────────────────────────────────────

    def _build(self):
        hdr = tk.Frame(self, bg=BG2)
        hdr.pack(fill=tk.X, pady=(0, 8))
        section_label(hdr, "Step 2 — Intrinsic Calibration").pack(pady=10)
        info_label(
            hdr,
            f"This step measures lens distortion for each camera. "
            f"Print or display a 9×6 checkerboard pattern, hold it in front of each camera, "
            f"and tilt/rotate it slowly in different orientations. "
            f"The system auto-collects {INTRINSIC_CALIB_FRAMES} frames, then click Compute. "
            f"Calibration is saved and never needs repeating for these cameras.",
            color=FG_MUTED,
        ).pack(pady=(0, 4))

        # Checkerboard link
        link_row = tk.Frame(hdr, bg=BG2)
        link_row.pack(pady=(0, 6))
        tk.Label(link_row, text="Print checkerboard from:  ", font=FONT_LABEL, fg=FG_MUTED, bg=BG2).pack(side=tk.LEFT)
        link = tk.Label(link_row, text="https://docs.opencv.org/4.x/pattern.png",
                        font=FONT_LABEL, fg="#4fc3f7", bg=BG2, cursor="hand2")
        link.pack(side=tk.LEFT)

        # Skip option
        skip_row = tk.Frame(hdr, bg=BG2)
        skip_row.pack(pady=(0, 8))
        tk.Label(skip_row, text="Testing only — no checkerboard? ",
                 font=FONT_LABEL, fg=FG_WARN, bg=BG2).pack(side=tk.LEFT)
        primary_btn(skip_row, "Skip (use estimated intrinsics)",
                    command=self._skip, width=28).pack(side=tk.LEFT, padx=6)

        # Progress indicators
        prog_row = tk.Frame(self, bg=BG)
        prog_row.pack(pady=4)
        self._status_ap  = StatusLight(prog_row, f"AP:  0/{INTRINSIC_CALIB_FRAMES} frames")
        self._status_ap.pack(side=tk.LEFT, padx=20)
        self._status_lat = StatusLight(prog_row, f"LAT: 0/{INTRINSIC_CALIB_FRAMES} frames")
        self._status_lat.pack(side=tk.LEFT, padx=20)

        # RMS error display
        self._rms_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._rms_var,
                 font=FONT_LABEL, fg=FG_MUTED, bg=BG).pack()

        # Previews
        prev_row = tk.Frame(self, bg=BG)
        prev_row.pack(fill=tk.BOTH, expand=True, pady=8)

        self._prev_ap  = CameraPreview(prev_row, "AP Camera",  width=560, height=380)
        self._prev_ap.pack(side=tk.LEFT, padx=8, fill=tk.BOTH, expand=True)

        self._prev_lat = CameraPreview(prev_row, "LAT Camera", width=560, height=380)
        self._prev_lat.pack(side=tk.LEFT, padx=8, fill=tk.BOTH, expand=True)

        # Buttons
        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(pady=10)
        self._compute_btn = success_btn(btn_row, "Compute intrinsics →",
                                         command=self._compute, width=22)
        self._compute_btn.configure(state=tk.DISABLED)
        self._compute_btn.pack(side=tk.LEFT, padx=8)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def on_show(self, **kwargs):
        # Reset calibrators in case we return to this view
        self._cal_ap  = IntrinsicCalibrator()
        self._cal_lat = IntrinsicCalibrator()

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
        annotated, accepted = self._cal_ap.process_frame(frame)
        n = self._cal_ap.n_frames
        state = "ok" if self._cal_ap.is_done else ("warn" if n > 0 else "pending")
        self._status_ap.set(state, f"AP:  {n}/{INTRINSIC_CALIB_FRAMES} frames")
        self._maybe_enable_compute()
        return annotated

    def _process_lat(self, frame):
        annotated, accepted = self._cal_lat.process_frame(frame)
        n = self._cal_lat.n_frames
        state = "ok" if self._cal_lat.is_done else ("warn" if n > 0 else "pending")
        self._status_lat.set(state, f"LAT: {n}/{INTRINSIC_CALIB_FRAMES} frames")
        self._maybe_enable_compute()
        return annotated

    def _maybe_enable_compute(self):
        if self._cal_ap.is_done and self._cal_lat.is_done:
            self._compute_btn.configure(state=tk.NORMAL)

    def _skip(self):
        """Use estimated intrinsics based on typical webcam parameters.
        Accuracy will be lower but sufficient for testing the full flow."""
        import numpy as np

        def _estimate(cap):
            """Build a reasonable pinhole matrix from the actual frame size."""
            frame = cap.get_frame() if cap else None
            if frame is not None:
                h, w = frame.shape[:2]
            else:
                w, h = 1280, 720
            f  = 0.85 * max(w, h)   # typical focal length estimate
            cx, cy = w / 2.0, h / 2.0
            mtx  = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            dist = np.zeros(5, dtype=np.float64)
            return mtx, dist

        mtx_ap,  dist_ap  = _estimate(self._state.cap_ap)
        mtx_lat, dist_lat = _estimate(self._state.cap_lat)

        self._state.mtx_ap   = mtx_ap
        self._state.dist_ap  = dist_ap
        self._state.mtx_lat  = mtx_lat
        self._state.dist_lat = dist_lat

        # Save so subsequent sessions also skip
        save_intrinsics(str(self._state.cam_ap_idx),  mtx_ap,  dist_ap)
        save_intrinsics(str(self._state.cam_lat_idx), mtx_lat, dist_lat)

        print("[DEBUG] Skipped intrinsic calibration — using estimated values")
        messagebox.showinfo(
            "Estimated intrinsics",
            "Using estimated camera parameters. Suitable for testing.\n"
            "For real training sessions, return here and calibrate with a checkerboard.",
        )
        self._app.proceed_after_intrinsic_calib()

    def _compute(self):
        try:
            mtx_ap,  dist_ap,  rms_ap  = self._cal_ap.compute()
            mtx_lat, dist_lat, rms_lat = self._cal_lat.compute()
        except Exception as exc:
            messagebox.showerror("Calibration Error", str(exc))
            return

        self._state.mtx_ap   = mtx_ap
        self._state.dist_ap  = dist_ap
        self._state.mtx_lat  = mtx_lat
        self._state.dist_lat = dist_lat

        save_intrinsics(str(self._state.cam_ap_idx),  mtx_ap,  dist_ap)
        save_intrinsics(str(self._state.cam_lat_idx), mtx_lat, dist_lat)

        self._rms_var.set(
            f"Calibration complete. RMS reprojection error — AP: {rms_ap:.3f} px, "
            f"LAT: {rms_lat:.3f} px. (< 1.0 px is excellent)"
        )
        self.after(1200, self._app.proceed_after_intrinsic_calib)
