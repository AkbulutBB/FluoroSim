"""
tools/calibrate_intrinsics.py — ChArUco intrinsic calibration (standalone).

Uses the SAME ChArUco board the tracker uses (config CHARUCO_*), so no
separate checkerboard is needed. Normally you'd run this from the launcher
(python main.py → Calibrate button), but this standalone form still works.

Usage
-----
    python tools/calibrate_intrinsics.py --device 0 --out data/intrinsics/cam0.npz
    python tools/calibrate_intrinsics.py --device 1 --out data/intrinsics/cam1.npz

Move the platform/board to varied angles and distances. Keys:
    SPACE  capture current view (when board is detected)
    c      compute + save once enough views collected
    q      quit without saving
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from tools.camera_utils import CalibrationSession


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, required=True, help="Camera device index")
    ap.add_argument("--out",    type=str, required=True, help="Output .npz path")
    ap.add_argument("--frames", type=int, default=cfg.CALIB_FRAMES,
                    help="Target number of views")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"Calibrating device {args.device} using the ChArUco board.")
    print("Show the board at varied angles/distances. SPACE=capture, c=compute, q=quit.")

    sess = CalibrationSession(device_index=args.device, out_path=args.out,
                              target_frames=args.frames)
    rms = sess.run(on_progress=lambda n, t: print(f"  captured {n}/{t}"))
    if rms is not None:
        print(f"Done. RMS reprojection error: {rms:.4f} px  →  saved to {args.out}")
    else:
        print("Calibration cancelled or too few views.")


if __name__ == "__main__":
    main()
