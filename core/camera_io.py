"""
core/camera_io.py  —  Threaded webcam capture
==============================================

A small wrapper around cv2.VideoCapture that grabs frames on a background
thread, so the Tkinter UI never blocks waiting on a camera.  ``read()`` always
returns the most recent frame (or None if nothing has arrived yet).
"""

from __future__ import annotations
from typing import Optional
import threading

import cv2
import numpy as np


class CameraStream:
    def __init__(self, index: int, width: int = 1280, height: int = 720):
        self.index   = index
        self._cap     = cv2.VideoCapture(index, cv2.CAP_DSHOW if _is_windows() else 0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame: Optional[np.ndarray] = None
        self._lock   = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def start(self) -> "CameraStream":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def list_available_cameras(max_index: int = 6) -> list:
    """Probe device indices 0..max_index-1 and return those that open."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if _is_windows() else 0)
        if cap is not None and cap.isOpened():
            found.append(i)
        if cap is not None:
            cap.release()
    return found


def _is_windows() -> bool:
    import sys
    return sys.platform.startswith("win")
