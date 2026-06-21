"""
ui/widgets.py  —  Reusable Tkinter widgets
===========================================

  cv2_to_photo        : convert an OpenCV BGR frame to a Tk PhotoImage (fitted)
  VideoPanel          : a label that shows live frames, updated from the main thread
  ClickableImage      : show a static image and let the user drop numbered points,
                        returning click coordinates in ORIGINAL image pixels
  StatusChip          : a small coloured readiness indicator
"""

from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


# ── colour palette ───────────────────────────────────────────────────────────
BG      = "#0f1419"
PANEL   = "#1b232c"
FG      = "#e6edf3"
MUTED   = "#8b98a5"
ACCENT  = "#4ea1ff"
OK      = "#3fb950"
WARN    = "#d29922"
BAD     = "#f85149"


def cv2_to_photo(frame: np.ndarray, max_w: int, max_h: int):
    """Return (PhotoImage, scale) where scale maps original->displayed pixels."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    disp = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(Image.fromarray(rgb))
    return photo, scale


class VideoPanel(ttk.Label):
    """A label that displays OpenCV frames.  Call show(frame) from the main thread."""

    def __init__(self, master, max_w=480, max_h=360, **kw):
        super().__init__(master, **kw)
        self.max_w, self.max_h = max_w, max_h
        self._photo = None
        self.configure(anchor="center")

    def show(self, frame: Optional[np.ndarray], placeholder: str = "no signal"):
        if frame is None:
            self.configure(image="", text=placeholder, foreground=MUTED)
            self._photo = None
            return
        self._photo, _ = cv2_to_photo(frame, self.max_w, self.max_h)
        self.configure(image=self._photo, text="")


class ClickableImage(tk.Canvas):
    """
    Displays a static image scaled to fit, and records left-clicks as numbered
    points.  Clicks are stored and reported in ORIGINAL image-pixel coordinates.
    Right-click removes the last point.
    """

    def __init__(self, master, max_w=560, max_h=560, on_change: Optional[Callable] = None, **kw):
        super().__init__(master, width=max_w, height=max_h, bg=PANEL,
                         highlightthickness=1, highlightbackground="#30363d", **kw)
        self.max_w, self.max_h = max_w, max_h
        self._img_photo = None
        self._scale = 1.0
        self._offx = self._offy = 0
        self._points: List[Tuple[float, float]] = []     # original-pixel coords
        self._predicted: List[Tuple[float, float]] = []   # DLT-reprojected fiducials
        self._outliers: set = set()                        # indices flagged by robust fit
        self._orig_size = (0, 0)
        self._on_change = on_change
        self.bind("<Button-1>", self._add_point)
        self.bind("<Button-3>", self._remove_last)

    def set_image(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        self._orig_size = (w, h)
        self._scale = min(self.max_w / w, self.max_h / h)
        dw, dh = int(w * self._scale), int(h * self._scale)
        self._offx = (self.max_w - dw) // 2
        self._offy = (self.max_h - dh) // 2
        disp = cv2.resize(bgr, (dw, dh))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._img_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._redraw()

    def clear_points(self):
        self._points.clear()
        self._redraw()
        self._notify()

    @property
    def points(self) -> List[Tuple[float, float]]:
        return list(self._points)

    def set_points(self, pts: List[Tuple[float, float]]):
        self._points = [tuple(map(float, p)) for p in pts]
        self._redraw()

    def set_predicted(self, pts: List[Tuple[float, float]], outliers=None):
        """Overlay where the DLT predicts each fiducial, paired to your clicks.

        A short green cross sitting on its blue circle = good. A red cross with
        a line running off to its circle = that bearing doesn't fit (re-click
        it); ``outliers`` are the indices the robust fit rejected.
        """
        self._predicted = [tuple(map(float, p)) for p in pts]
        self._outliers = set(int(i) for i in (outliers or []))
        self._redraw()

    # ---- internals ---------------------------------------------------------
    def _add_point(self, event):
        if self._img_photo is None:
            return
        ox = (event.x - self._offx) / self._scale
        oy = (event.y - self._offy) / self._scale
        w, h = self._orig_size
        if 0 <= ox <= w and 0 <= oy <= h:
            self._points.append((ox, oy))
            self._redraw()
            self._notify()

    def _remove_last(self, _event):
        if self._points:
            self._points.pop()
            self._redraw()
            self._notify()

    def _notify(self):
        if self._on_change:
            self._on_change()

    def _redraw(self):
        self.delete("all")
        if self._img_photo is not None:
            self.create_image(self._offx, self._offy, anchor="nw", image=self._img_photo)
        for i, (ox, oy) in enumerate(self._points, start=1):
            x = self._offx + ox * self._scale
            y = self._offy + oy * self._scale
            r = 6
            self.create_oval(x - r, y - r, x + r, y + r, outline=ACCENT, width=2)
            self.create_text(x + 10, y - 10, text=str(i), fill=ACCENT,
                             font=("Segoe UI", 10, "bold"))
        # registration-check crosses, paired to the matching click by index.
        for i, (ox, oy) in enumerate(self._predicted):
            if not (np.isfinite(ox) and np.isfinite(oy)):
                continue
            x = self._offx + ox * self._scale
            y = self._offy + oy * self._scale
            bad = i in self._outliers
            col = BAD if bad else OK
            # connector from this prediction to its click (same index)
            if i < len(self._points):
                px = self._offx + self._points[i][0] * self._scale
                py = self._offy + self._points[i][1] * self._scale
                self.create_line(px, py, x, y, fill=col,
                                 width=2 if bad else 1,
                                 dash=() if bad else (3, 2))
            r = 7
            self.create_line(x - r, y, x + r, y, fill=col, width=2)
            self.create_line(x, y - r, x, y + r, fill=col, width=2)
            self.create_text(x + 11, y + 11, text=str(i + 1), fill=col,
                             font=("Segoe UI", 9, "bold"))


class StatusChip(ttk.Frame):
    """A coloured dot + label that reflects a ready / not-ready state."""

    def __init__(self, master, text: str, **kw):
        super().__init__(master, **kw)
        self._canvas = tk.Canvas(self, width=14, height=14, highlightthickness=0, bg=BG)
        self._dot = self._canvas.create_oval(2, 2, 12, 12, fill=BAD, outline="")
        self._canvas.pack(side="left", padx=(0, 6))
        self._label = ttk.Label(self, text=text)
        self._label.pack(side="left")

    def set_state(self, ready: bool, text: Optional[str] = None):
        self._canvas.itemconfigure(self._dot, fill=OK if ready else BAD)
        if text is not None:
            self._label.configure(text=text)
