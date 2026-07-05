"""
ui/camera_assign.py — Live camera assignment (continuous streams).

Shows a live feed from every detected camera at once, each with
"Set as A" / "Set as B" buttons underneath. You watch the streams, see which
physical camera is which, and click to assign roles. No dropdowns, no
single-frame guessing.

Reconstructed from the previous FluoroSim UI, which used continuous previews.

Design note on Tkinter
-----------------------
CameraPreview inherits from tk.Frame, so it MUST NOT use the attribute name
`_w` (Tkinter reserves it for the widget's Tcl path string — overwriting it
crashes every child widget). We use `_width` / `_height` instead. Every
ImageTk.PhotoImage is created with master=<label> so images bind to the right
Tk root even when Spyder keeps stale roots alive.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

import config as cfg

BG      = "#1e1e1e"
BG2     = "#2a2a2a"
FG      = "#e6e6e6"
FG_MUT  = "#9a9a9a"
FG_OK   = "#37b24d"
ACCENT  = "#378add"


class CameraPreview(tk.Frame):
    """A single continuous camera stream with a role badge and assign buttons."""

    def __init__(self, parent, device_index: int,
                 preview_w: int, preview_h: int,
                 on_assign, **kwargs):
        super().__init__(parent, bg=BG2, **kwargs)
        self.device_index = device_index
        self._width   = preview_w        # NOT _w — Tkinter reserves that
        self._height  = preview_h
        self._on_assign = on_assign

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._role: Optional[str] = None

        # ── Widgets ──────────────────────────────────────────────────────
        tk.Label(self, text=f"Device {device_index}", bg=BG2, fg=FG,
                 font=("Segoe UI", 10, "bold")).pack(pady=(6, 2))

        self._video = tk.Label(self, bg="black", width=self._width, height=self._height)
        self._video.pack(padx=6)

        self._badge = tk.Label(self, text="unassigned", bg=BG2, fg=FG_MUT,
                               font=("Segoe UI", 9))
        self._badge.pack(pady=(2, 4))

        btns = tk.Frame(self, bg=BG2)
        btns.pack(pady=(0, 8))
        ttk.Button(btns, text="Set as A",
                   command=lambda: self._assign("A")).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Set as B",
                   command=lambda: self._assign("B")).pack(side=tk.LEFT, padx=3)

    # ── Streaming ────────────────────────────────────────────────────────

    def start(self) -> bool:
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.device_index, backend)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            self._video.configure(text="cannot open", fg="#e24b4a")
            return False
        # Modest capture size keeps the preview grid light
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._tick()
        return True

    def _loop(self):
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def _tick(self):
        """Refresh the displayed image on the Tk main loop (~20 fps)."""
        if not self._running:
            return
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self._width, self._height))
            photo = ImageTk.PhotoImage(Image.fromarray(rgb), master=self._video)
            self._video.configure(image=photo)
            self._video.image = photo
        self.after(50, self._tick)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Role assignment ──────────────────────────────────────────────────

    def _assign(self, role: str):
        self._on_assign(self.device_index, role)

    def set_role(self, role: Optional[str]):
        self._role = role
        if role:
            self._badge.configure(text=f"Camera {role}", fg=FG_OK)
        else:
            self._badge.configure(text="unassigned", fg=FG_MUT)


class CameraAssignDialog(tk.Toplevel):
    """
    Modal dialog showing all detected cameras as live streams.
    Returns the chosen (device_a, device_b) via the on_confirm callback.
    """

    def __init__(self, parent, device_indices: list[int], on_confirm):
        super().__init__(parent)
        self.title("Assign cameras — live view")
        self.configure(bg=BG)
        self._on_confirm = on_confirm
        self._assigned: dict[str, Optional[int]] = {"A": None, "B": None}
        self._previews: dict[int, CameraPreview] = {}

        tk.Label(self, text="Assign camera views", bg=BG, fg=FG,
                 font=("Segoe UI", 14, "bold")).pack(pady=(12, 2))
        tk.Label(self, text="Watch the live feeds, then set one camera as A "
                            "(frontal) and one as B (45° oblique).",
                 bg=BG, fg=FG_MUT, font=("Segoe UI", 9)).pack(pady=(0, 8))

        grid = tk.Frame(self, bg=BG)
        grid.pack(padx=12, pady=4)

        pw, ph = 320, 240
        for col, dev in enumerate(device_indices):
            pv = CameraPreview(grid, dev, pw, ph, self._on_assign)
            pv.grid(row=0, column=col, padx=8, pady=4, sticky="n")
            self._previews[dev] = pv
            pv.start()

        # Footer
        foot = tk.Frame(self, bg=BG)
        foot.pack(fill=tk.X, pady=10)
        self._status = tk.StringVar(value="Assign both A and B to continue.")
        tk.Label(foot, textvariable=self._status, bg=BG, fg=FG_MUT,
                 font=("Segoe UI", 9)).pack()
        self._confirm_btn = ttk.Button(foot, text="Confirm", state=tk.DISABLED,
                                       command=self._confirm)
        self._confirm_btn.pack(pady=6)

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.transient(parent)
        self.grab_set()

    # ── Assignment logic ─────────────────────────────────────────────────

    def _on_assign(self, device_index: int, role: str):
        # If this device already held the other role, clear that first
        for r in ("A", "B"):
            if self._assigned[r] == device_index and r != role:
                self._assigned[r] = None
        # If another device held this role, release it
        prev = self._assigned[role]
        if prev is not None and prev in self._previews:
            self._previews[prev].set_role(None)
        self._assigned[role] = device_index
        # Update badges
        for dev, pv in self._previews.items():
            role_for_dev = next((r for r, d in self._assigned.items() if d == dev), None)
            pv.set_role(role_for_dev)

        a, b = self._assigned["A"], self._assigned["B"]
        if a is not None and b is not None and a != b:
            self._confirm_btn.configure(state=tk.NORMAL)
            self._status.set(f"Camera A = device {a},  Camera B = device {b}")
        else:
            self._confirm_btn.configure(state=tk.DISABLED)
            if a == b and a is not None:
                self._status.set("A and B must be different cameras.")
            else:
                self._status.set("Assign both A and B to continue.")

    def _confirm(self):
        a, b = self._assigned["A"], self._assigned["B"]
        self._teardown()
        self._on_confirm(a, b)
        self.destroy()

    def _cancel(self):
        self._teardown()
        self.destroy()

    def _teardown(self):
        for pv in self._previews.values():
            pv.stop()
