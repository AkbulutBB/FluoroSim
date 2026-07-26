"""
tools/diagnose_probe_geometry.py — find out WHY probe reprojection error is high.

Motivation
----------
tools/calibrate_probe_yaw.py swept the inter-cube yaw and found an almost flat
error curve with a high floor (~24 px on one camera, ~33 px on the other), and
the two cameras disagreed on the sign. That pattern means yaw is not the
dominant error: a real 45 deg mounting error would produce a deep, unambiguous
minimum that both cameras agree on. Something more basic in the probe model is
wrong.

This script isolates which, bottom-up, instead of guessing:

  TEST 1 — within-cube vs cross-cube
      Solve using ONLY the bottom cube's faces, ONLY the top cube's faces, and
      both together.
        * both single-cube solves good, combined bad
              -> the cubes' RELATIVE geometry is wrong
                 (CUBE_GAP_MM, inter-cube yaw, or the ID->cube assignment)
        * single-cube solves already bad
              -> the error is INSIDE a cube
                 (sticker rotation, MARKER_SIZE_MM, CUBE_SIDE_MM,
                  or ID->face assignment)
      A single face gives an exactly-determined PnP (RMS ~ 0 always), so only
      frames with >= 2 faces of the same cube are informative here.

  TEST 2 — per-face sticker rotation
      Each marker was physically glued to its face at some orientation. If a
      sticker is on at 90/180/270 deg from what config assumes, the detected
      "top-left" corner maps to the wrong 3-D corner and that face contributes
      a large systematic residual that NO global yaw or scale can absorb.
      Solved by coordinate descent over the 4 possible rotations per face.

  TEST 3 — re-fit yaw and cube gap
      Only meaningful once TEST 2 is applied, since sticker errors otherwise
      swamp both parameters.

Usage
-----
    python tools/diagnose_probe_geometry.py

Capture 15-30 frames with the probe at varied orientations. Frames showing two
faces of the SAME cube are the valuable ones for TEST 1 and 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
from core.camera import CameraStream, load_intrinsics  # noqa: E402

BOTTOM_IDS = (0, 1, 2, 3)
TOP_IDS = (4, 5, 6, 7)
ALL_IDS = BOTTOM_IDS + TOP_IDS


# ─────────────────────────────────────────────────────────────────────────────
# Geometry construction
# ─────────────────────────────────────────────────────────────────────────────

def build_faces(
    bottom_yaw: float,
    top_yaw: float = 0.0,
    gap_mm: float | None = None,
    rots: dict[int, int] | None = None,
    bottom_order: tuple = BOTTOM_IDS,
    top_order: tuple = TOP_IDS,
) -> dict[int, np.ndarray]:
    """
    Probe face geometry for candidate parameters, without touching config.

    bottom_order / top_order give which marker ID sits on (+X, -X, +Y, -Y).
    """
    if gap_mm is None:
        gap_mm = cfg.CUBE_GAP_MM
    rots = rots or {}

    _h = cfg.MARKER_SIZE_MM / 2.0
    _s = cfg.CUBE_SIDE_MM / 2.0

    def cube(z_centre: float, yaw_deg: float, ids) -> dict[int, np.ndarray]:
        base = {
            ids[0]: [[ _s, -_h,  _h], [ _s,  _h,  _h], [ _s,  _h, -_h], [ _s, -_h, -_h]],
            ids[1]: [[-_s,  _h,  _h], [-_s, -_h,  _h], [-_s, -_h, -_h], [-_s,  _h, -_h]],
            ids[2]: [[-_h,  _s,  _h], [ _h,  _s,  _h], [ _h,  _s, -_h], [-_h,  _s, -_h]],
            ids[3]: [[ _h, -_s,  _h], [-_h, -_s,  _h], [-_h, -_s, -_h], [ _h, -_s, -_h]],
        }
        c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        out = {}
        for mid, pts in base.items():
            p = (R @ np.asarray(pts, dtype=np.float64).T).T
            p[:, 2] += z_centre
            r = int(rots.get(mid, 0)) % 4
            if r:
                p = np.roll(p, r, axis=0)
            out[mid] = p.astype(np.float32)
        return out

    return {**cube(0.0, bottom_yaw, bottom_order),
            **cube(gap_mm, top_yaw, top_order)}


# ─────────────────────────────────────────────────────────────────────────────
# Solve / scoring
# ─────────────────────────────────────────────────────────────────────────────

def solve_rms(faces, det, mtx, dist, id_filter=None) -> float | None:
    """
    Reprojection RMS for one detection set. Mirrors ProbeTracker.estimate()'s
    solver choice so results transfer directly to the live tracker.
    `det` is a list of (marker_id, 4x2 image corners).
    """
    obj_l, img_l = [], []
    for mid, corners in det:
        if id_filter is not None and mid not in id_filter:
            continue
        if mid not in faces:
            continue
        obj_l.append(faces[mid])
        img_l.append(corners)
    n_faces = len(obj_l)
    if n_faces < 2:          # single face = exact fit, RMS is meaningless
        return None

    obj = np.concatenate(obj_l, 0).reshape(-1, 3).astype(np.float32)
    img = np.concatenate(img_l, 0).reshape(-1, 2).astype(np.float32)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, mtx, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineVVS(obj, img, mtx, dist, rvec, tvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, mtx, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        rms = float(np.sqrt(np.mean(err ** 2)))
        return rms if np.isfinite(rms) else None
    except cv2.error:
        return None


def median_rms(faces, dets, mtx, dist, id_filter=None) -> tuple[float | None, int]:
    vals = [r for d in dets
            if (r := solve_rms(faces, d, mtx, dist, id_filter)) is not None]
    return (float(np.median(vals)) if vals else None), len(vals)


# ─────────────────────────────────────────────────────────────────────────────
# Capture
# ─────────────────────────────────────────────────────────────────────────────

def _ask(prompt: str, default: str) -> str:
    try:
        raw = input(f"{prompt} [{default}]: ").strip().strip('"').strip("'")
    except EOFError:
        return default
    return raw if raw else default


def capture(device: int, slot: str) -> list[list[tuple[int, np.ndarray]]]:
    detector = cv2.aruco.ArucoDetector(
        cfg.ARUCO_PROBE_DICT, cv2.aruco.DetectorParameters())
    cam = CameraStream(device, cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT,
                       getattr(cfg, "CAMERA_FPS", 30))
    if not cam.start():
        print(f"Could not open camera device {device}.", file=sys.stderr)
        return []

    dets: list[list[tuple[int, np.ndarray]]] = []
    print("\nSPACE = capture, ENTER = done, Q = abort")
    print("Most useful: frames showing TWO faces of the SAME cube.")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
                    break
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            cur, n_bot, n_top = [], 0, 0
            vis = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                for i, mid in enumerate(ids.flatten()):
                    mid = int(mid)
                    if mid in ALL_IDS:
                        cur.append((mid, corners[i][0].astype(np.float32)))
                        n_bot += mid in BOTTOM_IDS
                        n_top += mid in TOP_IDS

            same_cube_pair = n_bot >= 2 or n_top >= 2
            usable = (n_bot + n_top) >= 2
            colour = ((80, 220, 80) if same_cube_pair
                      else (0, 200, 220) if usable else (0, 140, 255))
            label = ("2+ faces same cube (BEST)" if same_cube_pair
                     else "usable" if usable else "need >=2 faces")
            cv2.putText(vis, f"bottom:{n_bot} top:{n_top}  {label}", (14, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.putText(vis, f"captured: {len(dets)}", (14, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
            cv2.imshow("Probe geometry diagnosis — SPACE / ENTER / Q", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and usable:
                dets.append(cur)
                print(f"  captured {len(dets)} (bottom={n_bot}, top={n_top})")
            elif key in (13, 10):
                break
            elif key in (ord('q'), 27):
                dets = []
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    return dets


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    slot = _ask("Camera slot — A or B", "A").upper()
    if slot not in ("A", "B"):
        print(f"  (unrecognised slot {slot!r}, using A)")
        slot = "A"
    idx = 0 if slot == "A" else 1
    intr_path = Path(cfg.INTRINSICS_PATH_0 if idx == 0 else cfg.INTRINSICS_PATH_1)
    default_dev = str(cfg.CAMERA_IDS[idx]) if idx < len(cfg.CAMERA_IDS) else str(idx)
    try:
        device = int(_ask(f"OS device index for camera {slot}", default_dev))
    except ValueError:
        device = int(default_dev)

    if not intr_path.exists():
        print(f"\nIntrinsics not found: {intr_path}", file=sys.stderr)
        return 1
    intr = load_intrinsics(str(intr_path))
    if intr is None:
        print(f"Could not read {intr_path}", file=sys.stderr)
        return 1
    mtx, dist = intr
    print(f"Loaded intrinsics for camera {slot}: {intr_path}")

    dets = capture(device, slot)
    if len(dets) < 5:
        print("Need at least 5 usable frames. Aborted.", file=sys.stderr)
        return 1

    yaw0 = float(getattr(cfg, "PROBE_CUBE_BOTTOM_YAW_DEG", 0.0))
    faces0 = build_faces(yaw0)

    # ── TEST 1 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("TEST 1 — is the error INSIDE a cube, or BETWEEN the cubes?")
    print("=" * 66)
    r_bot, n_bot = median_rms(faces0, dets, mtx, dist, set(BOTTOM_IDS))
    r_top, n_top = median_rms(faces0, dets, mtx, dist, set(TOP_IDS))
    r_all, n_all = median_rms(faces0, dets, mtx, dist, None)
    for nm, r, n in (("bottom cube only", r_bot, n_bot),
                     ("top cube only", r_top, n_top),
                     ("both cubes", r_all, n_all)):
        print(f"  {nm:<18}: "
              + (f"{r:7.2f} px   ({n} frames)" if r is not None
                 else f"{'n/a':>7}       (need >=2 faces of that cube)"))

    singles = [r for r in (r_bot, r_top) if r is not None]
    if singles and max(singles) < 4.0 and r_all is not None and r_all > 3 * max(singles):
        verdict = "CROSS-CUBE"
        print("\n  -> Each cube alone fits well, but combining them does not.")
        print("     The error is in the cubes' RELATIVE geometry: CUBE_GAP_MM,")
        print("     inter-cube yaw, or which ID set is on which cube.")
    elif singles and max(singles) >= 4.0:
        verdict = "WITHIN-CUBE"
        print("\n  -> A single cube already fits badly. The error is INSIDE a")
        print("     cube: sticker rotation, MARKER_SIZE_MM, CUBE_SIDE_MM, or")
        print("     the ID->face assignment. Yaw cannot fix this.")
    else:
        verdict = "UNKNOWN"
        print("\n  -> Not enough same-cube frames to classify. Recapture with")
        print("     more views showing two faces of one cube.")

    # ── TEST 2 ───────────────────────────────────────────────────────────
    # Solve which marker ID sits on which face, per cube, INDEPENDENTLY.
    # Uses only that cube's own frames, so the result does not depend on the
    # inter-cube yaw or gap. 24 permutations per cube; four of them are
    # equivalent under a 90 deg yaw step, so ties here are expected and
    # harmless -- the yaw fit in TEST 4 absorbs the difference.
    print("\n" + "=" * 66)
    print("TEST 2 — ID -> face assignment around each cube")
    print("=" * 66)
    import itertools

    sub = dets if len(dets) <= 120 else [dets[i] for i in
                                         np.linspace(0, len(dets) - 1, 120).astype(int)]

    def search_order(ids, filt):
        results = []
        for perm in itertools.permutations(ids):
            f = build_faces(yaw0, bottom_order=perm, top_order=perm)
            s_, n_ = median_rms(f, sub, mtx, dist, filt)
            if s_ is not None:
                results.append((s_, perm, n_))
        results.sort(key=lambda r: r[0])
        return results

    bot_res = search_order(BOTTOM_IDS, set(BOTTOM_IDS))
    top_res = search_order(TOP_IDS, set(TOP_IDS))

    best_bot_order = BOTTOM_IDS
    best_top_order = TOP_IDS
    for name, res, default in (("bottom", bot_res, BOTTOM_IDS),
                               ("top", top_res, TOP_IDS)):
        if not res:
            print(f"  {name} cube: no scorable frames (need >=2 faces of it)")
            continue
        print(f"  {name} cube — best 4 of 24 orderings (+X, -X, +Y, -Y):")
        for s_, perm, n_ in res[:4]:
            flag = "  <= current" if tuple(perm) == tuple(default) else ""
            print(f"      {perm}  -> {s_:7.2f} px  ({n_} frames){flag}")
        cur = [r for r in res if tuple(r[1]) == tuple(default)]
        if cur:
            print(f"      current {tuple(default)} ranks "
                  f"{res.index(cur[0]) + 1}/24 at {cur[0][0]:.2f} px")
        if name == "bottom":
            best_bot_order = tuple(res[0][1])
        else:
            best_top_order = tuple(res[0][1])

    # ── TEST 3 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("TEST 3 — per-face sticker rotation (given the best face orders)")
    print("=" * 66)
    rots = {mid: 0 for mid in ALL_IDS}

    def faces_now(yaw=None, gap=None, r=None):
        return build_faces(yaw0 if yaw is None else yaw,
                           gap_mm=gap, rots=r if r is not None else rots,
                           bottom_order=best_bot_order, top_order=best_top_order)

    best, _ = median_rms(faces_now(), sub, mtx, dist)
    if best is None:
        print("  Could not score baseline — skipping.")
    else:
        print(f"  start: {best:.2f} px")
        for sweep in range(4):
            improved = False
            for mid in ALL_IDS:
                scores = {}
                for r in range(4):
                    rots[mid] = r
                    s_, _ = median_rms(faces_now(), sub, mtx, dist)
                    if s_ is not None:
                        scores[r] = s_
                if not scores:
                    rots[mid] = 0
                    continue
                rb = min(scores, key=scores.get)
                rots[mid] = rb
                if scores[rb] < best - 1e-6:
                    best, improved = scores[rb], True
            print(f"  sweep {sweep + 1}: {best:.2f} px   rots={rots}")
            if not improved:
                break

    # ── TEST 4 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("TEST 4 — inter-cube yaw and gap")
    print("=" * 66)
    best_yaw, best_gap, best_rms = yaw0, float(cfg.CUBE_GAP_MM), best
    for yaw in np.arange(-180.0, 180.1, 5.0):
        for gap in (cfg.CUBE_GAP_MM - 6, cfg.CUBE_GAP_MM - 3,
                    cfg.CUBE_GAP_MM, cfg.CUBE_GAP_MM + 3, cfg.CUBE_GAP_MM + 6):
            s_, _ = median_rms(faces_now(yaw=float(yaw), gap=float(gap)),
                               sub, mtx, dist)
            if s_ is not None and (best_rms is None or s_ < best_rms):
                best_rms, best_yaw, best_gap = s_, float(yaw), float(gap)
    if best_rms is not None:
        print(f"  best yaw {best_yaw:+.1f} deg, gap {best_gap:.1f} mm "
              f"-> {best_rms:.2f} px")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("SUMMARY")
    print("=" * 66)
    print(f"  classification : {verdict}")
    if best_rms is not None:
        print(f"  best RMS found : {best_rms:.2f} px")
        if best_rms < 3.0:
            print("\n  Good fit. Put these in config.py:")
            print(f"      PROBE_BOTTOM_FACE_ORDER = {best_bot_order}")
            print(f"      PROBE_TOP_FACE_ORDER    = {best_top_order}")
            print(f"      PROBE_CUBE_BOTTOM_YAW_DEG = {best_yaw:.1f}")
            print(f"      CUBE_GAP_MM = {best_gap:.1f}")
            print(f"      PROBE_FACE_STICKER_ROT = {rots}")
        else:
            print("\n  Still a poor fit. Remaining suspects, in order:")
            print("    1. Physically confirm the ID->face map printed above")
            print("       against the actual cube (read the IDs off it).")
            print("    2. MARKER_SIZE_MM: measure a printed marker's black")
            print(f"       square with calipers (config says "
                  f"{cfg.MARKER_SIZE_MM} mm).")
            print(f"    3. CUBE_SIDE_MM: measure a cube (config says "
                  f"{cfg.CUBE_SIDE_MM} mm).")
            print("    4. Camera intrinsics for this slot — recalibrate and")
            print("       check the reported RMS is well under 1 px.")
    return 0


if __name__ == "__main__":
    _rc = main()
    if _rc != 0:
        print(f"\n(finished with status {_rc})")
