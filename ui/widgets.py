"""
ui/widgets.py — Reusable widgets and helpers for the FluoroSim UI.
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Callable


# ── Colour palette ─────────────────────────────────────────────────────────────
BG          = "#1a1a2e"
BG2         = "#16213e"
BG3         = "#0f3460"
ACCENT      = "#e94560"
FG          = "#e0e0e0"
FG_MUTED    = "#a0a0c0"
FG_SUCCESS  = "#4caf50"
FG_WARN     = "#ff9800"
FG_ERR      = "#f44336"
FONT_BODY   = ("Segoe UI", 10)
FONT_LABEL  = ("Segoe UI", 9)
FONT_TITLE  = ("Segoe UI", 13, "bold")
FONT_MONO   = ("Courier New", 9)


# ── Frame → Tkinter image ─────────────────────────────────────────────────────

def frame_to_tk(
    frame: np.ndarray,
    width: int,
    height: int,
) -> ImageTk.PhotoImage:
    """Convert an OpenCV BGR frame to a Tkinter PhotoImage, resized to (width, height)."""
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil   = Image.fromarray(rgb).resize((width, height), Image.BILINEAR)
    return ImageTk.PhotoImage(image=pil)


# ── Styled base frame ─────────────────────────────────────────────────────────

class DarkFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)


# ── Camera preview panel ──────────────────────────────────────────────────────

class CameraPreview(tk.Frame):
    """
    A labelled camera preview that refreshes itself via after().
    Call start(camera_capture_fn, width, height) to begin.
    Call stop() to halt the refresh loop.
    """

    def __init__(self, parent, label: str, width: int = 640, height: int = 480, **kwargs):
        super().__init__(parent, bg=BG2, relief=tk.FLAT, **kwargs)
        self._width     = width
        self._height    = height
        self._running   = False
        self._after_id  = None
        self._source_fn : Optional[Callable] = None     # returns np.ndarray | None
        self._process_fn: Optional[Callable] = None     # frame → annotated frame
        self._tk_img    = None
        self._on_frame  : Optional[Callable] = None     # callback(frame)

        # Header
        hdr = tk.Frame(self, bg=BG3)
        hdr.pack(fill=tk.X)
        self._label_var = tk.StringVar(value=label)
        tk.Label(hdr, textvariable=self._label_var,
                 font=FONT_LABEL, fg=FG_MUTED, bg=BG3, pady=4).pack()

        # Image label
        self._img_lbl = tk.Label(self, bg="#000", width=width, height=height)
        self._img_lbl.pack()

        # Status bar
        self._status_var = tk.StringVar(value="Waiting…")
        tk.Label(self, textvariable=self._status_var,
                 font=FONT_LABEL, fg=FG_MUTED, bg=BG2, pady=3).pack(fill=tk.X)

    def set_label(self, text: str):
        self._label_var.set(text)

    def set_status(self, text: str, color: str = FG_MUTED):
        self._status_var.set(text)
        # Could update color too — keep simple for now

    def start(
        self,
        source_fn: Callable,
        process_fn: Optional[Callable] = None,
        on_frame: Optional[Callable] = None,
        interval_ms: int = 66,   # ~15 fps for preview
    ):
        self._source_fn  = source_fn
        self._process_fn = process_fn
        self._on_frame   = on_frame
        self._running    = True
        self._interval   = interval_ms
        self._tick()

    def stop(self):
        self._running = False
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _tick(self):
        if not self._running:
            return
        frame = self._source_fn() if self._source_fn else None
        if frame is not None:
            if self._process_fn:
                frame = self._process_fn(frame)
            if self._on_frame:
                self._on_frame(frame)
            try:
                tk_img = frame_to_tk(frame, self._width, self._height)
                self._img_lbl.configure(image=tk_img)
                self._img_lbl.image = tk_img
                self._tk_img = tk_img
            except Exception:
                pass
        self._after_id = self.after(self._interval, self._tick)


# ── Button helpers ─────────────────────────────────────────────────────────────

def primary_btn(parent, text: str, command=None, width: int = 18) -> tk.Button:
    return tk.Button(
        parent, text=text, command=command,
        font=FONT_BODY, fg="#fff", bg=BG3,
        activeforeground="#fff", activebackground=ACCENT,
        relief=tk.FLAT, width=width, pady=8, padx=6, cursor="hand2",
    )


def success_btn(parent, text: str, command=None, width: int = 18) -> tk.Button:
    return tk.Button(
        parent, text=text, command=command,
        font=("Segoe UI", 10, "bold"), fg="#fff", bg="#1b5e20",
        activeforeground="#fff", activebackground="#4caf50",
        relief=tk.FLAT, width=width, pady=8, cursor="hand2",
    )


def section_label(parent, text: str) -> tk.Label:
    return tk.Label(parent, text=text, font=FONT_TITLE, fg=FG, bg=BG)


def info_label(parent, text: str, color: str = FG_MUTED) -> tk.Label:
    return tk.Label(parent, text=text, font=FONT_LABEL, fg=color, bg=BG,
                    wraplength=500, justify=tk.LEFT)


# ── Status indicator ───────────────────────────────────────────────────────────

class StatusLight(tk.Frame):
    """Small coloured dot + text label to indicate a pass/fail/pending state."""

    COLORS = {
        "pending": ("#888", "—"),
        "ok":      (FG_SUCCESS, "✔"),
        "fail":    (FG_ERR, "✘"),
        "warn":    (FG_WARN, "!"),
    }

    def __init__(self, parent, label: str, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self._dot = tk.Label(self, text="●", font=("Segoe UI", 12), bg=BG)
        self._dot.pack(side=tk.LEFT)
        self._lbl = tk.Label(self, text=label, font=FONT_LABEL, fg=FG, bg=BG)
        self._lbl.pack(side=tk.LEFT, padx=4)
        self.set("pending")

    def set(self, state: str, extra: str = ""):
        color, sym = self.COLORS.get(state, ("#888", "?"))
        self._dot.configure(text=sym, fg=color)
        if extra:
            self._lbl.configure(text=extra)