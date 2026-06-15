"""
probe_orientation_check.py
Standalone FluoroSim probe-orientation verifier.

Run from the FluoroSim project root, next to main.py:
    python probe_orientation_check.py --role ap --camera 0

Keys:
    ESC / q  : quit
    s        : save current frame as probe_orientation_snapshot.png

What to check:
    The red projected rod should lie exactly over the physical K-wire.
    The red dot should sit on the real probe tip.
    If the red rod is rotated, mirrored, or exits the wrong cube face, check
    marker IDs, face winding/top-edge orientation, and ROD_TIP_IN_CUBE.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _ensure_project_imports() -> None:
    """Allow running this file from either project root or another folder."""
    here = Path(__file__).resolve().parent
    candidates = [Path.cwd(), here, here.parent]
    for p in candidates:
        if (p / "core").exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            return


def _draw_text(img, lines, x=20, y=30, line_h=28):
    for i, line in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(img, line, (x + 1, yy + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 255, 80), 2, cv2.LINE_AA)


def main() -> int:
    _ensure_project_imports()

    import config
    from core.intrinsics import CameraIntrinsics
    from core.tracking import ProbeTracker, draw_probe_pose

    parser = argparse.ArgumentParser(description="Standalone FluoroSim probe orientation checker")
    parser.add_argument("--role", default="ap", choices=list(config.CAMERA_ROLES),
                        help="Which saved camera intrinsics to use: ap or lat")
    parser.add_argument("--camera", type=int, default=0,
                        help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=1280,
                        help="Requested camera width")
    parser.add_argument("--height", type=int, default=720,
                        help="Requested camera height")
    args = parser.parse_args()

    intr = CameraIntrinsics.load(args.role)
    if intr is None:
        print(f"ERROR: No saved intrinsics for role '{args.role}'.")
        print("First calibrate that camera inside FluoroSim, then rerun this script.")
        return 1

    tracker = ProbeTracker()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {args.camera}.")
        return 1

    win = "FluoroSim probe orientation check"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_save = None
    print("Running. Press ESC or q to quit, s to save a snapshot.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue

        obs = tracker.estimate(frame, intr.mtx, intr.dist)
        lines = [f"role={args.role}  camera={args.camera}"]

        if obs is not None:
            draw_probe_pose(frame, obs, intr.mtx, intr.dist)
            rod_len = float(np.linalg.norm(obs.rod_tip_cam - obs.rod_base_cam))
            tip = obs.rod_tip_cam
            base = obs.rod_base_cam
            lines.extend([
                f"cube detected: {obs.n_faces} face(s), primary ID {obs.primary_id}",
                f"rod length: {rod_len:.1f} mm  (expected {config.ROD_LENGTH_MM:.1f} mm)",
                f"base_cam: [{base[0]:.0f}, {base[1]:.0f}, {base[2]:.0f}] mm",
                f"tip_cam:  [{tip[0]:.0f}, {tip[1]:.0f}, {tip[2]:.0f}] mm",
                "CHECK: red line should overlap physical K-wire; red dot = real tip",
            ])
        else:
            lines.extend([
                "probe not detected",
                "Show the ArUco cube to the camera; try 2+ visible faces.",
            ])

        if last_save:
            lines.append(f"saved: {last_save}")

        _draw_text(frame, lines)
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            out = Path.cwd() / "probe_orientation_snapshot.png"
            cv2.imwrite(str(out), frame)
            last_save = str(out)
            print(f"Saved {out}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
