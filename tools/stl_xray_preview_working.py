"""
tools/stl_xray_preview.py — Standalone AP + Lateral X-ray preview for one STL.

Bottom-up sanity check, before anything else in the pipeline is trusted:
    "Does this STL actually look like a spine on a synthetic X-ray?"

This script does NOT touch the ChArUco/probe/registration side of FluoroSim
at all. It loads a single STL, assigns it a bone-equivalent material, clips
away the bottom N mm (the platform attachment plate, which is print
hardware, not anatomy, and should never appear on the simulated image),
and renders AP + lateral views through the same XRaySimulator /
VirtualCArm classes core/xray_sim.py uses in the full app. Re-using that
class (rather than re-deriving gVXR calls here) means: if this script
works, the rendering core of the real pipeline works.

The isocentre is auto-computed from the CLIPPED mesh's own bounding-box
centre, so this tool has no dependency on BOARD_TO_WORLD or
SPINE_ORIGIN_IN_WORLD, both of which are still placeholders. Good enough
for "does the geometry/density look right" — not for registration.

Usage
-----
    python tools/stl_xray_preview.py path/to/spine.stl
    python tools/stl_xray_preview.py path/to/spine.stl --clip-mm 6 --up-axis z
    python tools/stl_xray_preview.py path/to/spine.stl --no-clip
    python tools/stl_xray_preview.py path/to/spine.stl --compound C5H8O2 --density 1.27

If no STL path is given, falls back to config.SPINE_STL_PATH.

Requires (on top of the existing FluoroSim requirements.txt):
    pip install trimesh shapely mapbox_earcut --break-system-packages
    (shapely + mapbox_earcut are trimesh's dependencies for capping the cut
    face closed with cap=True — without them, clipping still runs but raises
    "No available triangulation engine!" on the cap step.)

Output
------
    outputs/stl_preview/clipped_preview.stl   (the mesh actually rendered —
                                                open this in Fusion/MeshLab
                                                first if the render looks wrong)
    outputs/stl_preview/ap_preview.png
    outputs/stl_preview/lat_preview.png

ASSUMPTIONS — flagged, verify before trusting the output
----------------------------------------------------------------------
1. "Bottom" = the minimum coordinate along --up-axis (default z) in the
   STL's OWN local frame. If your Fusion export uses Y as vertical, or the
   model is exported upside-down, this clips the wrong end. Check
   clipped_preview.stl visually — if the plate is still there, or the
   spine is gone, your STL's up-axis differs from the default.
2. "Invisible" is implemented as geometric removal (the bottom slab is cut
   away and the cut face is triangulated closed), not a zero-density
   region. gVXR assigns density per whole mesh, not per sub-region, so a
   partial-density plate isn't directly supported — and physically, actual
   absence is the correct way to model "this part isn't there in the
   simulated beam path" anyway. The ORIGINAL STL on disk is untouched;
   only the copy in outputs/stl_preview/ is modified.
3. Bone material defaults to hydroxyapatite (Ca10(PO4)6(OH)2, 1.92 g/cm3),
   matching config.SPINE_MATERIAL_COMPOUND/DENSITY. Override with
   --compound / --density if you're previewing an uncoated PETG print
   instead (config.py suggests C5H8O2 @ 1.27 g/cm3 for that case).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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


DEFAULT_OUTDIR = Path("outputs/stl_preview")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "stl_path", nargs="?", default=None,
        help="Path to the spine STL. Defaults to config.SPINE_STL_PATH.",
    )
    p.add_argument("--clip-mm", type=float, default=8.0,
                    help="mm to remove from the bottom (platform attachment "
                         "plate). Default 6.0. Use --no-clip to disable.")
    p.add_argument("--no-clip", action="store_true",
                    help="Skip clipping entirely; render the STL as-is.")
    p.add_argument("--up-axis", choices=["x", "y", "z"], default="z",
                    help="Which local STL axis is 'up' (bottom = min on this "
                         "axis). Default z.")
    p.add_argument("--compound", default=cfg.SPINE_MATERIAL_COMPOUND,
                    help=f"gVXR chemical compound string. "
                         f"Default: {cfg.SPINE_MATERIAL_COMPOUND!r} (from config.py).")
    p.add_argument("--density", type=float, default=cfg.SPINE_MATERIAL_DENSITY,
                    help=f"Density in g/cm3. Default: {cfg.SPINE_MATERIAL_DENSITY} "
                         f"(from config.py).")
    p.add_argument("--pixels", type=int, default=cfg.GVXR_DETECTOR_PIXELS[0],
                    help="Square detector resolution (pixels per side).")
    p.add_argument("--pixel-size-mm", type=float, default=cfg.GVXR_PIXEL_SIZE_MM)
    p.add_argument("--energy-mev", type=float, default=cfg.GVXR_ENERGY_MEV)
    p.add_argument("--photon-count", type=int, default=cfg.GVXR_PHOTON_COUNT)
    p.add_argument("--sod-mm", type=float, default=cfg.GVXR_SOD_MM,
                    help="Source-to-isocentre distance.")
    p.add_argument("--det-offset-mm", type=float, default=cfg.GVXR_DET_OFFSET_MM)
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    return p.parse_args()


def clip_bottom(
    stl_path: Path, clip_mm: float, up_axis: str, out_path: Path
) -> tuple[Path, np.ndarray]:
    """
    Remove the bottom `clip_mm` of the mesh along `up_axis` and save the
    remainder (capped closed) as a new STL at `out_path`.

    Returns (out_path, clipped_bounds) where clipped_bounds is the (2, 3)
    [min_xyz, max_xyz] array of the clipped mesh. Raises RuntimeError if the
    clip would remove the entire mesh (almost certainly the wrong --up-axis).
    """
    if not TRIMESH_AVAILABLE:
        raise RuntimeError(
            "trimesh is required for clipping. Install with:\n"
            "    pip install trimesh --break-system-packages\n"
            "Or pass --no-clip to render the STL unmodified."
        )

    mesh = trimesh.load_mesh(str(stl_path), force="mesh")
    axis_idx = {"x": 0, "y": 1, "z": 2}[up_axis]

    bounds = mesh.bounds  # (2, 3): [min_xyz, max_xyz]
    total_extent = bounds[1][axis_idx] - bounds[0][axis_idx]
    if clip_mm >= total_extent:
        raise RuntimeError(
            f"--clip-mm {clip_mm} >= mesh extent {total_extent:.1f} mm along "
            f"'{up_axis}' — this would delete the whole mesh. Check --up-axis."
        )

    normal = np.zeros(3)
    normal[axis_idx] = 1.0
    plane_origin = np.zeros(3)
    plane_origin[axis_idx] = bounds[0][axis_idx] + clip_mm

    clipped = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=normal, plane_origin=plane_origin, cap=True,
    )
    if clipped is None or len(clipped.vertices) == 0:
        raise RuntimeError(
            f"Clipping at {clip_mm} mm along '{up_axis}' removed everything. "
            "Check --up-axis matches your STL's orientation (try the other axes)."
        )

    if not clipped.is_watertight:
        print("  [warn] clipped mesh is not watertight — attempting fill_holes()")
        clipped.fill_holes()
        if not clipped.is_watertight:
            print("  [warn] still not watertight after fill_holes() — gVXR may "
                  "produce artefacts at the cut face. Consider capping in Fusion "
                  "instead and re-exporting.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clipped.export(str(out_path))
    return out_path, clipped.bounds


def build_preview_cfg(
    stl_path: Path,
    compound: str,
    density: float,
    pixels: int,
    pixel_size_mm: float,
    energy_mev: float,
    photon_count: int,
    sod_mm: float,
    det_offset_mm: float,
    isocenter_local: np.ndarray,
) -> SimpleNamespace:
    """
    Build a lightweight config namespace for XRaySimulator that renders ONLY
    the given STL — no platform, no bearings, no board registration.
    Reuses config.py's AP/LAT beam directions and context type so the
    preview matches the real pipeline's viewing convention.
    """
    return SimpleNamespace(
        GVXR_CONTEXT=cfg.GVXR_CONTEXT,
        SPINE_STL_PATH=str(stl_path),
        PLATFORM_STL_PATH="",
        SPINE_MATERIAL_COMPOUND=compound,
        SPINE_MATERIAL_DENSITY=density,
        SPINE_ORIGIN_IN_WORLD=(0.0, 0.0, 0.0),
        BEARING_POSITIONS=[],
        GVXR_ISOCENTER=tuple(float(v) for v in isocenter_local),
        GVXR_SOD_MM=sod_mm,
        GVXR_DET_OFFSET_MM=det_offset_mm,
        GVXR_AP_BEAM_DIR=cfg.GVXR_AP_BEAM_DIR,
        GVXR_AP_UP=cfg.GVXR_AP_UP,
        GVXR_LAT_BEAM_DIR=cfg.GVXR_LAT_BEAM_DIR,
        GVXR_LAT_UP=cfg.GVXR_LAT_UP,
        GVXR_DETECTOR_PIXELS=(pixels, pixels),
        GVXR_PIXEL_SIZE_MM=pixel_size_mm,
        GVXR_ENERGY_MEV=energy_mev,
        GVXR_PHOTON_COUNT=photon_count,
    )


def main() -> int:
    args = parse_args()

    if not GVXR_AVAILABLE:
        print("gvxrPython3 is not importable in this environment. "
              "Run tools/test_gvxr.py first.", file=sys.stderr)
        return 1

    stl_path = Path(args.stl_path) if args.stl_path else Path(cfg.SPINE_STL_PATH)
    if not stl_path.exists():
        print(f"STL not found: {stl_path}", file=sys.stderr)
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FluoroSim — STL X-ray Preview")
    print("=" * 60)
    print(f"Input STL     : {stl_path}")

    render_path = stl_path
    isocenter_local = np.zeros(3)

    if args.no_clip or args.clip_mm <= 0:
        print("Clipping      : disabled")
        if TRIMESH_AVAILABLE:
            m = trimesh.load_mesh(str(stl_path), force="mesh")
            isocenter_local = m.bounds.mean(axis=0)
    else:
        print(f"Clipping      : bottom {args.clip_mm} mm along '{args.up_axis}'")
        clipped_path = outdir / "clipped_preview.stl"
        try:
            render_path, clipped_bounds = clip_bottom(
                stl_path, args.clip_mm, args.up_axis, clipped_path
            )
        except RuntimeError as exc:
            print(f"[FAILED] {exc}", file=sys.stderr)
            return 1
        isocenter_local = np.mean(clipped_bounds, axis=0)
        print(f"  clipped mesh saved -> {clipped_path}")
        print(f"  clipped bounds (mm): "
              f"min={clipped_bounds[0].round(1)}  max={clipped_bounds[1].round(1)}")

    print(f"Material      : {args.compound} @ {args.density} g/cm3")
    print(f"Isocentre     : {isocenter_local.round(1)} (auto, mesh-local)")
    print(f"Detector      : {args.pixels}x{args.pixels} px @ "
          f"{args.pixel_size_mm} mm/px  "
          f"({args.pixels * args.pixel_size_mm:.0f} mm FOV)")

    preview_cfg = build_preview_cfg(
        render_path, args.compound, args.density,
        args.pixels, args.pixel_size_mm, args.energy_mev, args.photon_count,
        args.sod_mm, args.det_offset_mm, isocenter_local,
    )

    sim = XRaySimulator(preview_cfg)
    print("\nInitialising gVXR context + scene ... ", end="", flush=True)
    t0 = time.perf_counter()
    if not sim.initialise():
        print("FAILED — see log above.", file=sys.stderr)
        return 1
    print(f"OK ({time.perf_counter() - t0:.1f} s)")

    print("Rendering AP + lateral ... ", end="", flush=True)
    t0 = time.perf_counter()
    ap_img, lat_img = sim.render_background()
    elapsed = time.perf_counter() - t0
    print(f"OK ({elapsed:.1f} s)")

    if ap_img is None or lat_img is None:
        print("[FAILED] render_background() returned None.", file=sys.stderr)
        sim.shutdown()
        return 1

    ap_path = outdir / "ap_preview.png"
    lat_path = outdir / "lat_preview.png"
    cv2.imwrite(str(ap_path), ap_img)
    cv2.imwrite(str(lat_path), lat_img)
    sim.shutdown()

    print(f"\nSaved: {ap_path}")
    print(f"Saved: {lat_path}")
    print("\nOpen both PNGs and check:")
    print("  1. Does the spine read as bone (bright anatomy, dark background)?")
    print("  2. Is the platform attachment plate gone from BOTH views?")
    print("  3. Do AP and lateral look like two different projections of the")
    print("     same object (not two copies of the same view)?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
