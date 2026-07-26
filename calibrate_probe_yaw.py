"""
tools/calibrate_probe_yaw.py — MEASURE the probe's inter-cube yaw angle.

Why this exists
---------------
One ArUco cube is mounted rotated relative to the other so a camera always sees
two faces per cube. The nominal angle is 45 deg, but two things are unknown:

  1. The SIGN. +45 and -45 differ by 90 deg, which amounts to relabeling which
     marker ID sits on which face. A wrong sign is a genuine error in the model
     of the cubes' relative orientation and wrecks any solve that uses markers
     from both cubes at once (symptom: reprojection RMS in the tens of pixels).
  2. The exact VALUE. The mount is 3-D printed and hand-assembled, so the true
     angle may be 45 +/- a few degrees.

Rather than guess, this sweeps candidate yaw values, re-solves the probe pose
with each, and reports which one minimises reprojection error. That measures
the physical mounting angle instead of assuming it.

It reuses the production ProbeTracker solve path (by swapping
cfg.PROBE_FACE_OBJ_PTS for each candidate), so there is no duplicated pose
maths that could drift from the real tracker.

Requirements
------------
- One camera, already intrinsically calibrated (data/intrinsics/cam0.npz).
- Hold the probe so that BOTH cubes are visible, at several orientations.
  Frames where only one cube is seen carry no information about the RELATIVE
  yaw and are skipped automatically.

Usage
-----
    python tools/calibrate_probe_yaw.py

Press SPACE to capture a frame, ENTER when done (aim for 8-15 frames across a
range of probe orientations), or Q to abort.

Then put the reported value into config.py:
    PROBE_CUBE_BOTTOM_YAW_DEG = <best>
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
from core.camera import CameraStream, load_intrinsics  # noqa: E402
from core.markers import ProbeTracker  # noqa: E402

BOTTOM_IDS = (0, 1, 2, 3)
TOP_IDS = (4, 5, 6, 7)

# Coarse pass then fine pass around the winner.
COARSE = np.arange(-90.0, 90.1, 5.0)
FINE_HALF_WIDTH = 6.0
FINE_STEP = 0.5


def _ask(prompt: str, default: str) -> str:
    """Prompt with a default; Enter (or no stdin) accepts the default."""
    try:
        raw = input(f"{prompt} [{default}]: ").strip().strip('"').strip("'")
    except EOFError:
        return default
    return raw if raw else default


def prompt_args() -> tuple[int, Path, str]:
    """
    Interactive fallback so the script is F5-runnable in Spyder.

    Asks for the camera SLOT (A or B) rather than an intrinsics file path,
    because the two are coupled: ui/app.py pairs INTRINSICS_PATH_0 with
    CAMERA_IDS[0] (Camera A) and INTRINSICS_PATH_1 with CAMERA_IDS[1]
    (Camera B). Asking for a path separately invites a silent mismatch between
    the camera being opened and the intrinsics being applied.
    """
    slot = _ask("Camera slot — A or B", "A").upper()
    if slot not in ("A", "B"):
        print(f"  (unrecognised slot {slot!r}, using A)")
        slot = "A"
    idx = 0 if slot == "A" else 1

    intr = Path(cfg.INTRINSICS_PATH_0 if idx == 0 else cfg.INTRINSICS_PATH_1)

    # cfg.CAMERA_IDS is assigned at runtime by the launcher and is NOT saved to
    # disk, so in a standalone tool it is only the config default. Offer it, but
    # let the user correct it.
    default_dev = str(cfg.CAMERA_IDS[idx]) if idx < len(cfg.CAMERA_IDS) else str(idx)
    dev_raw = _ask(f"OS device index for camera {slot}", default_dev)
    try:
        device = int(dev_raw)
    except ValueError:
        print(f"  (not a number: {dev_raw!r}, using {default_dev})")
        device = int(default_dev)

    return device, intr, slot


def build_faces(bottom_yaw_deg: float, top_yaw_deg: float = 0.0) -> dict:
    """Probe face geometry for a candidate yaw, via config's own generator."""
    return {
        **cfg._cube_face_obj_pts(0.0, bottom_yaw_deg, BOTTOM_IDS),
        **cfg._cube_face_obj_pts(cfg.CUBE_GAP_MM, top_yaw_deg, TOP_IDS),
    }


def capture_frames(cam_index: int, slot: str) -> list[np.ndarray]:
    """Collect frames in which markers from BOTH cubes are visible."""
    detector = cv2.aruco.ArucoDetector(
        cfg.ARUCO_PROBE_DICT, cv2.aruco.DetectorParameters())

    # Reuse the app's camera class so backend selection / resolution handling
    # is identical to the working navigation window.
    cam = CameraStream(cam_index, cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT,
                       getattr(cfg, "CAMERA_FPS", 30))
    if not cam.start():
        print(f"Could not open camera device {cam_index}.\n"
              f"Tip: the device index for slot {slot} is assigned in the "
              f"launcher and not saved, so it may differ from the config "
              f"default. Try another index.", file=sys.stderr)
        return []

    kept: list[np.ndarray] = []
    print("\nSPACE = capture, ENTER = done, Q = abort")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
                    break
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            n_bot = n_top = 0
            vis = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                flat = ids.flatten().tolist()
                n_bot = sum(1 for i in flat if i in BOTTOM_IDS)
                n_top = sum(1 for i in flat if i in TOP_IDS)

            usable = n_bot >= 1 and n_top >= 1 and (n_bot + n_top) >= 3
            colour = (80, 220, 80) if usable else (0, 140, 255)
            cv2.putText(vis, f"bottom:{n_bot}  top:{n_top}  "
                             f"{'USABLE' if usable else 'need both cubes'}",
                        (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.putText(vis, f"captured: {len(kept)}", (14, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
            cv2.imshow("Probe yaw calibration — SPACE capture / ENTER done", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and usable:
                kept.append(frame.copy())
                print(f"  captured frame {len(kept)} (bottom={n_bot}, top={n_top})")
            elif key in (13, 10):
                break
            elif key in (ord('q'), 27):
                kept = []
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    return kept


def score_yaw(frames, mtx, dist, yaw: float) -> float | None:
    """Median probe reprojection RMS across frames for one candidate yaw."""
    original = cfg.PROBE_FACE_OBJ_PTS
    cfg.PROBE_FACE_OBJ_PTS = build_faces(yaw)
    try:
        tracker = ProbeTracker()
        rms_vals = []
        for f in frames:
            pose = tracker.estimate(f, mtx, dist)
            if pose is not None and np.isfinite(pose.rms_reproj):
                rms_vals.append(float(pose.rms_reproj))
        if not rms_vals:
            return None
        return float(np.median(rms_vals))
    finally:
        cfg.PROBE_FACE_OBJ_PTS = original


def main() -> int:
    cam_index, intr_path, slot = prompt_args()
    if not intr_path.exists():
        print(f"\nIntrinsics not found: {intr_path}\n"
              f"Run the calibration step for camera {slot} in main.py first.",
              file=sys.stderr)
        return 1

    # Use the app's loader — the .npz keys are "cam_mtx"/"dist", not "mtx".
    intr = load_intrinsics(str(intr_path))
    if intr is None:
        print(f"Could not read intrinsics from {intr_path}.", file=sys.stderr)
        return 1
    mtx, dist = intr
    print(f"Loaded intrinsics for camera {slot}: {intr_path}")

    frames = capture_frames(cam_index, slot)
    if len(frames) < 3:
        print("Need at least 3 usable frames (both cubes visible). Aborted.",
              file=sys.stderr)
        return 1
    print(f"\nScoring {len(frames)} frames...")

    results = []
    for yaw in COARSE:
        s = score_yaw(frames, mtx, dist, float(yaw))
        if s is not None:
            results.append((float(yaw), s))
    if not results:
        print("No candidate yaw produced a usable pose. Check marker IDs and "
              "that the probe geometry constants match the physical build.",
              file=sys.stderr)
        return 1

    results.sort(key=lambda r: r[1])
    best_coarse = results[0][0]

    fine = np.arange(best_coarse - FINE_HALF_WIDTH,
                     best_coarse + FINE_HALF_WIDTH + 1e-9, FINE_STEP)
    fine_results = []
    for yaw in fine:
        s = score_yaw(frames, mtx, dist, float(yaw))
        if s is not None:
            fine_results.append((float(yaw), s))
    fine_results.sort(key=lambda r: r[1])
    best_yaw, best_rms = fine_results[0]

    print("\n" + "=" * 62)
    print("Coarse sweep — 8 best candidates (yaw deg -> median RMS px)")
    print("=" * 62)
    for yaw, s in results[:8]:
        print(f"  {yaw:+7.1f} deg   {s:8.2f} px")

    current = float(getattr(cfg, "PROBE_CUBE_BOTTOM_YAW_DEG", 0.0))
    current_rms = score_yaw(frames, mtx, dist, current)

    print("\n" + "=" * 62)
    print("RESULT")
    print("=" * 62)
    print(f"  Best yaw          : {best_yaw:+.1f} deg   (median RMS {best_rms:.2f} px)")
    if current_rms is not None:
        print(f"  Currently in config: {current:+.1f} deg   (median RMS {current_rms:.2f} px)")

    if best_rms > 10.0:
        print("\n[WARNING] Even the best yaw leaves a large reprojection error.")
        print("  That points at something other than yaw -- check CUBE_SIDE_MM,")
        print("  MARKER_SIZE_MM, CUBE_GAP_MM against the physical probe, and")
        print("  confirm which marker ID is on which face.")
    else:
        print(f"\n  Put this in config.py:")
        print(f"      PROBE_CUBE_BOTTOM_YAW_DEG = {best_yaw:.1f}")

    print("\nNote: only the RELATIVE yaw between the cubes is identifiable "
          "here.\nA global yaw of the whole probe is unobservable in tip "
          "position or\ntrajectory, because the K-wire is coaxial with the "
          "probe's +Z axis.")
    return 0


if __name__ == "__main__":
    # Deliberately not sys.exit(): SystemExit surfaces in Spyder as
    # "An exception has occurred", which looks like a crash even on a clean
    # early return. This script is meant to be F5-runnable.
    _rc = main()
    if _rc != 0:
        print(f"\n(finished with status {_rc} — see messages above)")
