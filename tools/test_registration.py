"""
tools/test_registration.py — Registration sanity check, no hardware required.

Unlike stl_xray_preview.py (which deliberately ignores BOARD_TO_WORLD and
SPINE_TO_WORLD to test the spine STL/material in isolation), this script
renders the REAL config.py exactly as main.py's live app would — spine
mesh placed via SPINE_TO_WORLD, bearings placed via BEARING_POSITIONS, both
in the same gVXR world frame — but with zero camera, board, or probe
involved. It answers one question: "given today's config.py values, does
the spine + bearing geometry actually look/measure like a coherent scene?"

Run this AFTER tools/test_gvxr.py (confirms gVXR itself works) and BEFORE
main.py (which additionally needs cameras, an intrinsic calibration, and
the physical rig).

Checks performed
-----------------
1. Renders AP + lateral backgrounds (spine + 8 bearings) via the real
   XRaySimulator(config) — the exact same call ui/app.py makes.
2. Numerically projects every BEARING_POSITIONS entry through both views'
   pinhole geometry and reports whether each lands inside the detector
   frame. A bearing projecting off-frame usually means GVXR_ISOCENTER is
   far from where the bearings/spine actually are, not a rendering bug.
3. Reports the spine mesh's world-space bounding box (via trimesh, if
   installed) so you can eyeball whether it's anywhere near the bearings
   and isocentre, before even opening the PNGs.

Usage
-----
    python tools/test_registration.py

Output
------
    outputs/registration_test/ap_background.png
    outputs/registration_test/lat_background.png

What to look for in the PNGs
-----------------------------
- Bearings should appear as small bright dots roughly along the spine's
  periphery (not floating in empty background space, not buried deep
  inside/outside the visible anatomy).
- AP and LAT should look like two different projections of the SAME
  object, not two unrelated renders.
- If the spine is offset far to one side or mostly/entirely off-frame,
  SPINE_TO_WORLD is very likely still wrong (check rotation sign/axis
  first — that's the part solved from the fewest points).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
from core.xray_sim import XRaySimulator, GVXR_AVAILABLE  # noqa: E402

# Bearings live on the platform, not the spine, and are off by default for
# trainee-facing renders -- but they are the whole point of this test, so
# force them on for this run only.
cfg.RENDER_BEARINGS = True

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    trimesh = None  # type: ignore
    TRIMESH_AVAILABLE = False

OUTDIR = Path("outputs/registration_test")

# BGR (OpenCV order)
GREEN = (0, 255, 0)


def annotate_bearings(gray: np.ndarray, carm, bearings, world_to_spine) -> np.ndarray:
    """
    Draw a labelled green ring at each bearing's ANALYTICALLY projected pixel
    position on a colour copy of the grayscale render.

    gVXR output is an X-ray attenuation map — single-channel by physics — so a
    bearing cannot be "coloured" inside the render itself. Instead we project
    each bearing's known CAD position through the same pinhole geometry the
    renderer used (VirtualCArm.project_point) and mark it.

    This doubles as an independent cross-check: the ring is computed from the
    transform chain + projection maths, while the white dot underneath it comes
    from gVXR's ray-casting through the actual sphere mesh. If ring and dot
    coincide, both paths agree. If they are offset, one of them is wrong — and
    the pixel gap is a direct readout of the registration error.
    """
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    n_drawn = 0
    for b in bearings:
        pos_world = np.append(np.asarray(b["position_mm"], dtype=np.float64), 1.0)
        pos_render = (np.asarray(world_to_spine, dtype=np.float64) @ pos_world)[:3]
        px = carm.project_point(pos_render)
        if px is None:
            continue
        col, row = px
        # Ring, not filled disc — so the rendered bearing stays visible inside.
        cv2.circle(rgb, (col, row), 10, GREEN, 1, cv2.LINE_AA)
        cv2.putText(rgb, str(b["label"]), (col + 14, row + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1, cv2.LINE_AA)
        n_drawn += 1
    cv2.putText(rgb, f"{n_drawn} bearing(s) in frame", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)
    return rgb


def main() -> int:
    if not GVXR_AVAILABLE:
        print("gvxrPython3 is not importable in this environment. "
              "Run tools/test_gvxr.py first.", file=sys.stderr)
        return 1

    print("=" * 60)
    print("FluoroSim — Registration Sanity Check (no hardware)")
    print("=" * 60)

    spine_path = Path(cfg.SPINE_STL_PATH)
    print(f"Spine STL     : {spine_path}  "
          f"({'found' if spine_path.exists() else 'MISSING -> cuboid placeholder will be used'})")
    print(f"BOARD_TO_WORLD:\n{np.round(np.asarray(cfg.BOARD_TO_WORLD), 3)}")
    print(f"SPINE_TO_WORLD:\n{np.round(np.asarray(cfg.SPINE_TO_WORLD), 3)}")
    print(f"BOARD_TO_SPINE (derived, render frame):\n"
          f"{np.round(np.asarray(cfg.BOARD_TO_SPINE), 3)}")
    print(f"Isocentre     : {cfg.GVXR_ISOCENTER if cfg.GVXR_ISOCENTER is not None else 'AUTO (clipped spine bbox centre)'}")
    print(f"Baseplate clip: {cfg.SPINE_CLIP_MM} mm along local {cfg.SPINE_CLIP_AXIS}")
    print(f"Bearings      : {len(cfg.BEARING_POSITIONS)} configured")

    if TRIMESH_AVAILABLE and spine_path.exists():
        try:
            m = trimesh.load_mesh(str(spine_path), force="mesh")
            print(f"Spine bbox (spine-local = RENDER frame, unclipped): "
                  f"min={m.bounds[0].round(1)}  max={m.bounds[1].round(1)}")
        except Exception as exc:
            print(f"  (bbox check skipped: {exc})")

    # Where does the ChArUco board origin land in the render frame? This is the
    # single number that ties the tracker to the anatomy -- if it is nowhere
    # near the spine bbox above, the probe will render in the wrong place.
    board_origin_render = (np.asarray(cfg.BOARD_TO_SPINE)
                           @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    print(f"ChArUco origin in render frame: {board_origin_render.round(2)}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    sim = XRaySimulator(cfg)
    print("\nInitialising gVXR context + scene ... ", end="", flush=True)
    if not sim.initialise():
        print("FAILED — see log above.", file=sys.stderr)
        return 1
    print("OK")

    print("Rendering AP + lateral (spine + bearings) ... ", end="", flush=True)
    ap_img, lat_img = sim.render_background()
    if ap_img is None or lat_img is None:
        print("FAILED — render_background() returned None.", file=sys.stderr)
        sim.shutdown()
        return 1
    print("OK")

    ap_path = OUTDIR / "ap_background.png"
    lat_path = OUTDIR / "lat_background.png"
    cv2.imwrite(str(ap_path), ap_img)
    cv2.imwrite(str(lat_path), lat_img)

    # Annotated copies: green labelled rings at the analytically projected
    # bearing positions. Compare ring vs rendered white dot -- see docstring.
    w2s = np.asarray(cfg.WORLD_TO_SPINE, dtype=np.float64)
    ap_annot_path = OUTDIR / "ap_background_bearings.png"
    lat_annot_path = OUTDIR / "lat_background_bearings.png"
    cv2.imwrite(str(ap_annot_path),
                annotate_bearings(ap_img, sim._ap, cfg.BEARING_POSITIONS, w2s))
    cv2.imwrite(str(lat_annot_path),
                annotate_bearings(lat_img, sim._lat, cfg.BEARING_POSITIONS, w2s))

    # ── Numeric bearing-projection check ────────────────────────────────
    print("\nBearing projection check:")
    npix = cfg.GVXR_DETECTOR_PIXELS
    n_in_ap = n_in_lat = 0
    w2s = np.asarray(cfg.WORLD_TO_SPINE, dtype=np.float64)
    for b in cfg.BEARING_POSITIONS:
        pos_world = np.append(np.array(b["position_mm"], dtype=np.float64), 1.0)
        pos = (w2s @ pos_world)[:3]   # into the spine-local render frame
        px_ap = sim._ap.project_point(pos)
        px_lat = sim._lat.project_point(pos)
        in_ap = px_ap is not None
        in_lat = px_lat is not None
        n_in_ap += in_ap
        n_in_lat += in_lat
        print(f"  {b['label']:<14} render={tuple(pos.round(1))}  "
              f"AP={'in-frame ' + str(px_ap) if in_ap else 'OFF-FRAME'}  "
              f"LAT={'in-frame ' + str(px_lat) if in_lat else 'OFF-FRAME'}")

    sim.shutdown()

    print(f"\n{n_in_ap}/{len(cfg.BEARING_POSITIONS)} bearings in-frame on AP, "
          f"{n_in_lat}/{len(cfg.BEARING_POSITIONS)} on LAT "
          f"(detector {npix[0]}x{npix[1]} px)")
    fov_mm = npix[0] * cfg.GVXR_PIXEL_SIZE_MM
    print(f"\nSaved: {ap_path}")
    print(f"Saved: {lat_path}")
    print(f"Saved: {ap_annot_path}   <- green labelled bearing rings")
    print(f"Saved: {lat_annot_path}  <- green labelled bearing rings")

    print(f"\nNote: the detector covers only ~{fov_mm:.0f} x {fov_mm:.0f} mm "
          f"({npix[0]}px x {cfg.GVXR_PIXEL_SIZE_MM} mm/px), centred on the "
          f"spine. The bearings are pressed into the PLATFORM walls, which span "
          f"a much wider footprint than that -- so most of them physically "
          f"cannot appear in frame. Few/no bearings visible is EXPECTED here and "
          f"is not evidence of a registration error.")

    if n_in_ap < len(cfg.BEARING_POSITIONS) or n_in_lat < len(cfg.BEARING_POSITIONS):
        print("\n[INFO] Not all bearings are in-frame on both views (see note above).")

    print("\nWhat to check in the *_bearings.png images:")
    print("  For any bearing that IS in frame, the green ring (analytic")
    print("  projection of its CAD position) should sit directly on top of the")
    print("  small bright dot (gVXR's ray-cast through the actual sphere mesh).")
    print("  Coincident  -> transform chain and projection maths agree.")
    print("  Offset      -> the pixel gap is your registration error readout.")
    print("\nThe decisive end-to-end test is still the probe: put it in the")
    print("platform hole and use the Snapshot button in main.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
