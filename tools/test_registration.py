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

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    trimesh = None  # type: ignore
    TRIMESH_AVAILABLE = False

OUTDIR = Path("outputs/registration_test")


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
    print(f"Isocentre     : {cfg.GVXR_ISOCENTER}")
    print(f"Bearings      : {len(cfg.BEARING_POSITIONS)} configured")

    if TRIMESH_AVAILABLE and spine_path.exists():
        try:
            m = trimesh.load_mesh(str(spine_path), force="mesh")
            R = np.asarray(cfg.SPINE_TO_WORLD)[:3, :3]
            t = np.asarray(cfg.SPINE_TO_WORLD)[:3, 3]
            verts_world = (R @ m.vertices.T).T + t
            print(f"Spine bbox (world, via SPINE_TO_WORLD): "
                  f"min={verts_world.min(axis=0).round(1)}  "
                  f"max={verts_world.max(axis=0).round(1)}")
        except Exception as exc:
            print(f"  (bbox check skipped: {exc})")

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

    # ── Numeric bearing-projection check ────────────────────────────────
    print("\nBearing projection check:")
    npix = cfg.GVXR_DETECTOR_PIXELS
    n_in_ap = n_in_lat = 0
    for b in cfg.BEARING_POSITIONS:
        pos = np.array(b["position_mm"], dtype=np.float64)
        px_ap = sim._ap.project_point(pos)
        px_lat = sim._lat.project_point(pos)
        in_ap = px_ap is not None
        in_lat = px_lat is not None
        n_in_ap += in_ap
        n_in_lat += in_lat
        print(f"  {b['label']:<14} world={tuple(pos.round(1))}  "
              f"AP={'in-frame ' + str(px_ap) if in_ap else 'OFF-FRAME'}  "
              f"LAT={'in-frame ' + str(px_lat) if in_lat else 'OFF-FRAME'}")

    sim.shutdown()

    print(f"\n{n_in_ap}/{len(cfg.BEARING_POSITIONS)} bearings in-frame on AP, "
          f"{n_in_lat}/{len(cfg.BEARING_POSITIONS)} on LAT "
          f"(detector {npix[0]}x{npix[1]} px)")
    print(f"\nSaved: {ap_path}")
    print(f"Saved: {lat_path}")

    if n_in_ap < len(cfg.BEARING_POSITIONS) or n_in_lat < len(cfg.BEARING_POSITIONS):
        print("\n[WARNING] Not all bearings are in-frame on both views.")
        print("This does NOT necessarily mean SPINE_TO_WORLD/BOARD_TO_WORLD are")
        print("wrong -- GVXR_ISOCENTER may just need centring on your actual")
        print("bearing cluster. Check the printed world-space bbox above.")

    print("\nOpen both PNGs and check the criteria described in this script's")
    print("module docstring before moving on to camera/board/probe testing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
