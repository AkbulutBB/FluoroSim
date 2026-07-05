"""
tools/camera_utils.py — Shared camera helpers used by the launcher.

list_cameras()         : probe device indices 0..N and report which open.
CalibrationSession     : run an interactive checkerboard calibration inside
                         a Tk-driven OpenCV window, callable from the launcher.

These wrap the primitives in core/camera.py so the launcher never needs the
command-line tools — though tools/calibrate_intrinsics.py still works
standalone for anyone who prefers the terminal.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from core.camera import (
    make_object_points, find_checkerboard,
    compute_intrinsics, save_intrinsics,
    make_charuco_board, detect_charuco, compute_intrinsics_charuco,
)


def list_cameras(max_index: int = 6) -> list[int]:
    """Return the indices of cameras that successfully open."""
    found = []
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found


def grab_one_frame(device_index: int) -> Optional[np.ndarray]:
    """Open a camera, grab a single frame, release. For preview thumbnails."""
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    cap = cv2.VideoCapture(device_index, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
    frame = None
    for _ in range(5):                 # warm up; first frames are often blank
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    return frame


class CalibrationSession:
    """
    Interactive ChArUco intrinsic calibration in an OpenCV window.

    Uses the SAME ChArUco board the tracker uses (config CHARUCO_*), so no
    separate checkerboard is needed — just move the platform/board to varied
    angles and distances in front of the camera.

    Usage (from launcher):
        sess = CalibrationSession(device_index=1, out_path="data/intrinsics/cam0.npz")
        rms  = sess.run(on_progress=lambda n, total: ...)   # blocks until done
        # returns rms float on success, or None if cancelled / too few frames
    """

    def __init__(self, device_index: int, out_path: str,
                 target_frames: int = cfg.CALIB_FRAMES):
        self.device_index  = device_index
        self.out_path      = out_path
        self.target_frames = target_frames

    def run(self, on_progress: Optional[Callable[[int, int], None]] = None
            ) -> Optional[float]:
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.device_index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
        if not cap.isOpened():
            return None

        board    = make_charuco_board()
        detector = cv2.aruco.CharucoDetector(board)
        captured: list = []          # list of (charuco_corners, charuco_ids)
        image_size = None
        last_capture = frame_count = 0
        win = f"Calibrate camera {self.device_index}  —  SPACE capture, c compute, q cancel"

        result_rms: Optional[float] = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image_size = gray.shape[::-1]
                found, ch_corners, ch_ids, n_corners = detect_charuco(gray, detector)

                disp = frame.copy()
                if ch_ids is not None and len(ch_ids) > 0:
                    cv2.aruco.drawDetectedCornersCharuco(disp, ch_corners, ch_ids)

                auto = (found and len(captured) < self.target_frames
                        and frame_count - last_capture > 15)

                n = len(captured)
                cv2.putText(disp, f"Captured {n}/{self.target_frames}",
                            (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (60, 220, 60), 2)
                # Live board-detection status so it's obvious when SPACE will work
                if found:
                    cv2.putText(disp, f"Board detected ({n_corners} corners) — press SPACE",
                                (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (80, 220, 80), 2)
                else:
                    cv2.putText(disp, "Show the ChArUco board to the camera",
                                (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 140, 255), 2)
                cv2.putText(disp, "Move board to varied angles & distances",
                            (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 80), 2)
                if n >= self.target_frames:
                    cv2.putText(disp, "Enough frames — press 'c' to compute",
                                (16, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (80, 200, 255), 2)
                cv2.imshow(win, disp)
                key = cv2.waitKey(1) & 0xFF

                if (auto or key == ord(" ")) and found:
                    captured.append((ch_corners, ch_ids))
                    last_capture = frame_count
                    if on_progress:
                        on_progress(len(captured), self.target_frames)

                if key == ord("c") and len(captured) >= 8:
                    res = compute_intrinsics_charuco(board, captured, image_size)
                    if res is not None:
                        cam_mtx, dist, rms = res
                        save_intrinsics(self.out_path, cam_mtx, dist, rms)
                        result_rms = rms
                    break

                if key == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyWindow(win)
        return result_rms
