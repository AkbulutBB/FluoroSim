"""
ui/launcher.py — FluoroSim centralized setup & launch screen.

One window. Every setup step is a button with a live status indicator.
When the required steps are green, "Launch Navigation" enables and opens the
tracking window. No command-line flags, no remembering device indices.

Steps
-----
  1. Verify gVXR              → renders a test image
  2. Detect cameras          → scan device indices, assign A / B
  3. Calibrate Camera A      → checkerboard intrinsics → cam0.npz
  4. Calibrate Camera B      → checkerboard intrinsics → cam1.npz
  5. Generate probe sheet    → A4 PDF of the 8 markers
  ───────────────────────────────────────────────────────────
  [ Launch Navigation ]      (enabled when 1–4 are green)
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import config as cfg
from core.camera import load_intrinsics

# Status glyphs
OK    = "✓"
BAD   = "✗"
WAIT  = "…"

COL_OK   = "#37b24d"
COL_BAD  = "#888888"
COL_WARN = "#f08c00"
COL_BG   = "#1e1e1e"
COL_CARD = "#2a2a2a"
COL_TXT  = "#e6e6e6"


class StepRow:
    """One row in the setup checklist: label, status badge, action button."""

    def __init__(self, parent, n: int, title: str, subtitle: str,
                 action_text: str, action_cmd):
        self.frame = tk.Frame(parent, bg=COL_CARD)
        self.frame.pack(fill=tk.X, padx=10, pady=4)

        self.badge = tk.Label(self.frame, text=BAD, width=2, font=("Segoe UI", 16, "bold"),
                              fg=COL_BAD, bg=COL_CARD)
        self.badge.pack(side=tk.LEFT, padx=(10, 8), pady=10)

        textcol = tk.Frame(self.frame, bg=COL_CARD)
        textcol.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(textcol, text=f"{n}.  {title}", anchor="w",
                 font=("Segoe UI", 11, "bold"), fg=COL_TXT, bg=COL_CARD
                 ).pack(fill=tk.X)
        self.sub = tk.Label(textcol, text=subtitle, anchor="w",
                            font=("Segoe UI", 9), fg="#9a9a9a", bg=COL_CARD)
        self.sub.pack(fill=tk.X)

        self.btn = ttk.Button(self.frame, text=action_text, command=action_cmd, width=18)
        self.btn.pack(side=tk.RIGHT, padx=10)

    def set_status(self, state: str, subtitle: Optional[str] = None):
        if state == "ok":
            self.badge.configure(text=OK, fg=COL_OK)
        elif state == "warn":
            self.badge.configure(text=WAIT, fg=COL_WARN)
        else:
            self.badge.configure(text=BAD, fg=COL_BAD)
        if subtitle is not None:
            self.sub.configure(text=subtitle)


class LauncherWindow:
    def __init__(self, root: tk.Tk):
        self._root = root
        root.title(f"{cfg.APP_TITLE} — Setup")
        root.configure(bg=COL_BG)
        root.geometry("680x560")

        self._cameras: list[int] = []
        self._dev_a: Optional[int] = None
        self._dev_b: Optional[int] = None

        self._build()
        self._refresh_all_status()

    # ── Layout ────────────────────────────────────────────────────────

    def _build(self):
        header = tk.Frame(self._root, bg=COL_BG)
        header.pack(fill=tk.X, pady=(14, 6))
        tk.Label(header, text="FluoroSim", font=("Segoe UI", 20, "bold"),
                 fg=COL_TXT, bg=COL_BG).pack()
        tk.Label(header, text="Synthetic fluoroscopy navigation — guided setup",
                 font=("Segoe UI", 10), fg="#9a9a9a", bg=COL_BG).pack()

        body = tk.Frame(self._root, bg=COL_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)

        self._row_gvxr = StepRow(
            body, 1, "Verify X-ray engine (gVXR)",
            "Render a synthetic test image to confirm the GPU context works.",
            "Verify", self._do_verify_gvxr)

        self._row_cams = StepRow(
            body, 2, "Detect cameras",
            "Scan for connected webcams and assign Camera A / Camera B.",
            "Scan", self._do_scan_cameras)

        self._row_calA = StepRow(
            body, 3, "Calibrate Camera A",
            "Checkerboard intrinsic calibration → cam0.npz",
            "Calibrate", lambda: self._do_calibrate("A"))

        self._row_calB = StepRow(
            body, 4, "Calibrate Camera B",
            "Checkerboard intrinsic calibration → cam1.npz",
            "Calibrate", lambda: self._do_calibrate("B"))

        self._row_sheet = StepRow(
            body, 5, "Generate probe marker sheet",
            "Optional — A4 PDF of the 8 cube markers, print at 100% scale.",
            "Save PDF", self._do_probe_sheet)

        # Launch bar
        bar = tk.Frame(self._root, bg=COL_BG)
        bar.pack(fill=tk.X, side=tk.BOTTOM, pady=14)
        self._status_var = tk.StringVar(value="Complete steps 1–4 to enable launch.")
        tk.Label(bar, textvariable=self._status_var, fg="#9a9a9a", bg=COL_BG,
                 font=("Segoe UI", 9)).pack()
        self._launch_btn = ttk.Button(bar, text="▶  Launch Navigation",
                                      command=self._do_launch, state=tk.DISABLED)
        self._launch_btn.pack(pady=6, ipadx=20, ipady=4)

    # ── Status refresh ────────────────────────────────────────────────

    def _refresh_all_status(self):
        # gVXR: we can't know without running it; leave as-is unless verified.
        # Cameras
        if self._dev_a is not None and self._dev_b is not None:
            self._row_cams.set_status(
                "ok", f"Camera A = device {self._dev_a},  Camera B = device {self._dev_b}")
        # Calibrations (check files on disk)
        a_ok = load_intrinsics(cfg.INTRINSICS_PATH_0) is not None
        b_ok = load_intrinsics(cfg.INTRINSICS_PATH_1) is not None
        self._row_calA.set_status("ok" if a_ok else "bad",
            "Calibrated ✓" if a_ok else "Checkerboard intrinsic calibration → cam0.npz")
        self._row_calB.set_status("ok" if b_ok else "bad",
            "Calibrated ✓" if b_ok else "Checkerboard intrinsic calibration → cam1.npz")

        cams_ok = self._dev_a is not None and self._dev_b is not None
        if cams_ok and a_ok and b_ok:
            self._launch_btn.configure(state=tk.NORMAL)
            self._status_var.set("Ready to launch.")
        else:
            self._launch_btn.configure(state=tk.DISABLED)

    # ── Step 1: gVXR ──────────────────────────────────────────────────

    def _do_verify_gvxr(self):
        self._row_gvxr.set_status("warn", "Rendering test image…")
        self._root.update_idletasks()

        def work():
            from core.xray_sim import XRaySimulator, GVXR_AVAILABLE
            if not GVXR_AVAILABLE:
                self._root.after(0, lambda: self._row_gvxr.set_status(
                    "bad", "gVXR not installed — pip install gvxr"))
                return
            sim = XRaySimulator(cfg)
            ok = sim.initialise()
            if ok:
                ap, _ = sim.render_background()
                ok = ap is not None and ap.max() > 0
            sim.shutdown()
            msg = ("X-ray engine OK — GPU context verified."
                   if ok else "gVXR failed to render — see console log.")
            self._root.after(0, lambda: self._row_gvxr.set_status(
                "ok" if ok else "bad", msg))

        threading.Thread(target=work, daemon=True).start()

    # ── Step 2: cameras ───────────────────────────────────────────────

    def _do_scan_cameras(self):
        self._row_cams.set_status("warn", "Scanning device indices…")
        self._root.update_idletasks()

        def work():
            from tools.camera_utils import list_cameras
            found = list_cameras(max_index=6)
            self._root.after(0, lambda: self._on_cameras_found(found))

        threading.Thread(target=work, daemon=True).start()

    def _on_cameras_found(self, found: list[int]):
        self._cameras = found
        if len(found) < 2:
            self._row_cams.set_status(
                "bad", f"Found {len(found)} camera(s) — need 2. Check USB connections.")
            return
        # Open the live-stream assignment dialog
        from ui.camera_assign import CameraAssignDialog
        CameraAssignDialog(self._root, found, self._on_cameras_assigned)

    def _on_cameras_assigned(self, dev_a: int, dev_b: int):
        self._dev_a = dev_a
        self._dev_b = dev_b
        cfg.CAMERA_IDS = [dev_a, dev_b]
        self._refresh_all_status()

    # ── Steps 3–4: calibration ────────────────────────────────────────

    def _do_calibrate(self, which: str):
        if which == "A":
            dev = self._dev_a if self._dev_a is not None else (
                self._cameras[0] if self._cameras else 0)
            out = cfg.INTRINSICS_PATH_0
            row = self._row_calA
        else:
            dev = self._dev_b if self._dev_b is not None else (
                self._cameras[1] if len(self._cameras) > 1 else 1)
            out = cfg.INTRINSICS_PATH_1
            row = self._row_calB

        if self._dev_a is None or self._dev_b is None:
            messagebox.showinfo("Detect cameras first",
                                "Run step 2 (Detect cameras) before calibrating.")
            return

        Path(out).parent.mkdir(parents=True, exist_ok=True)
        row.set_status("warn", f"Calibrating on device {dev}… (OpenCV window)")
        self._root.update_idletasks()

        def work():
            from tools.camera_utils import CalibrationSession
            sess = CalibrationSession(device_index=dev, out_path=out)
            rms = sess.run()
            def done():
                if rms is not None:
                    row.set_status("ok", f"Calibrated ✓  (RMS {rms:.3f} px)")
                else:
                    row.set_status("bad", "Calibration cancelled or too few frames.")
                self._refresh_all_status()
            self._root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    # ── Step 5: probe sheet ───────────────────────────────────────────

    def _do_probe_sheet(self):
        out = filedialog.asksaveasfilename(
            title="Save probe marker sheet",
            defaultextension=".pdf",
            initialfile="probe_sheet.pdf",
            filetypes=[("PDF", "*.pdf"), ("PNG", "*.png")],
        )
        if not out:
            return
        self._row_sheet.set_status("warn", "Generating…")
        self._root.update_idletasks()

        def work():
            import importlib.util, sys
            # Call the generator's functions directly
            sys.argv = ["make_probe_sheet.py", "--out", out, "--dpi", "300"]
            from tools import make_probe_sheet as mps
            try:
                ok = mps.self_check(300)
                sheet = mps.build_sheet(300)
                import cv2
                if out.lower().endswith(".pdf"):
                    from PIL import Image
                    Image.fromarray(sheet).save(out, "PDF", resolution=300)
                else:
                    cv2.imwrite(out, sheet)
                msg = (f"Saved ✓  ({'all markers verified' if ok else 'check failed'})")
                state = "ok"
            except Exception as e:
                msg = f"Failed: {e}"
                state = "bad"
            self._root.after(0, lambda: self._row_sheet.set_status(state, msg))

        threading.Thread(target=work, daemon=True).start()

    # ── Launch ────────────────────────────────────────────────────────

    def _do_launch(self):
        from ui.app import FluoroSimApp
        # Hand off: close launcher, open navigation window
        self._root.destroy()
        nav_root = tk.Tk()
        app = FluoroSimApp(nav_root)
        nav_root.protocol("WM_DELETE_WINDOW", app.on_close)
        nav_root.mainloop()
