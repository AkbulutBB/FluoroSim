"""
ui/widgets.py — Shared UI primitives and theme.

All views import colours, fonts, and widget factory functions from here
to keep the visual language consistent across the application.
"""

import tkinter as tk
from typing import Optional, Callable
import cv2
import numpy as np
from PIL import Image, ImageTk

# ── Colour palette ─────────────────────────────────────────────────────────────
BG      = "#0d1117"   # page background
BG2     = "#161b22"   # card / panel background
BG3     = "#21262d"   # inset / input background
ACCENT  = "#238636"   # primary action (green)
ACCENT2 = "#1f6feb"   # secondary action (blue)

FG         = "#e6edf3"   # primary text
FG_MUTED   = "#8b949e"   # secondary text
FG_SUCCESS = "#3fb950"   # success
FG_ERR     = "#f85149"   # error
FG_WARN    = "#d29922"   # warning

# ── Typography ─────────────────────────────────────────────────────────────────
FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_BODY  = ("Segoe UI", 10)
FONT_LABEL = ("Segoe UI",  9)
FONT_MONO  = ("Consolas",  9)


# ── Base frame ─────────────────────────────────────────────────────────────────

class DarkFrame(tk.Frame):
    """Base class for all full-screen views."""
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", BG)
        super().__init__(parent, **kwargs)

    def on_show(self, **kwargs):
        """Called when this view becomes active. Override as needed."""

    def on_hide(self):
        """Called when this view is hidden. Override to release resources."""


# ── Button factories ───────────────────────────────────────────────────────────

def _btn(parent, text, command, bg, fg, active_bg, width=14, **kw):
    return tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, activebackground=active_bg, activeforeground=fg,
        relief=tk.FLAT, padx=10, pady=6,
        font=FONT_BODY, cursor="hand2", width=width, **kw,
    )

def primary_btn(parent, text, command=None, width=14, **kw):
    return _btn(parent, text, command, ACCENT2, FG, "#388bfd", width, **kw)

def success_btn(parent, text, command=None, width=14, **kw):
    return _btn(parent, text, command, ACCENT, FG, "#2ea043", width, **kw)

def danger_btn(parent, text, command=None, width=14, **kw):
    return _btn(parent, text, command, "#b91c1c", FG, "#dc2626", width, **kw)


# ── Label helpers ──────────────────────────────────────────────────────────────

def section_label(parent, text: str, **kw) -> tk.Label:
    return tk.Label(parent, text=text, font=FONT_TITLE,
                    fg=FG, bg=BG2, **kw)

def info_label(parent, text: str, color: str = FG_MUTED, **kw) -> tk.Label:
    return tk.Label(parent, text=text, font=FONT_LABEL,
                    fg=color, bg=parent.cget("bg"),
                    wraplength=700, justify=tk.LEFT, **kw)


# ── Live camera preview widget ─────────────────────────────────────────────────

class CameraPreview(tk.Frame):
    """
    Displays a continuously updated camera frame inside a Label widget.

    Call start(source_fn) to begin pulling frames, stop() to halt.
    source_fn is a callable that returns an Optional[np.ndarray].
    """

    def __init__(self, parent, label: str = "", width: int = 320, height: int = 240, **kw):
        kw.setdefault("bg", BG2)
        super().__init__(parent, **kw)

        self._w = width
        self._h = height
        self._after_id  = None
        self._source_fn: Optional[Callable] = None
        self._tk_img    = None

        tk.Label(self, text=label, font=FONT_LABEL, fg=FG_MUTED, bg=BG2).pack()
        self._lbl = tk.Label(self, bg="#000", width=width, height=height)
        self._lbl.pack(padx=4, pady=4)
        self._status_lbl = tk.Label(self, text="", font=FONT_LABEL,
                                    fg=FG_MUTED, bg=BG2)
        self._status_lbl.pack()

    def start(self, source_fn: Callable):
        self._source_fn = source_fn
        self._tick()

    def stop(self):
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def set_label(self, text: str):
        # Update the top label of the preview
        for widget in self.winfo_children():
            if isinstance(widget, tk.Label) and widget is not self._lbl and widget is not self._status_lbl:
                widget.configure(text=text)
                break

    def set_status(self, text: str, color: str = FG_MUTED):
        self._status_lbl.configure(text=text, fg=color)

    def _tick(self):
        if self._source_fn:
            frame = self._source_fn()
            if frame is not None:
                tk_img = frame_to_tk(frame, self._w, self._h)
                self._lbl.configure(image=tk_img)
                self._lbl.image = tk_img
                self._tk_img    = tk_img
        self._after_id = self.after(33, self._tick)   # ~30 fps


# ── Image conversion utility ───────────────────────────────────────────────────

def frame_to_tk(
    frame: np.ndarray,
    target_w: int,
    target_h: int,
) -> ImageTk.PhotoImage:
    """
    Convert an OpenCV BGR frame to a Tkinter-compatible PhotoImage,
    scaling it to fit within target_w × target_h while preserving aspect ratio.
    """
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
    return ImageTk.PhotoImage(image=pil)
