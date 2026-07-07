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


def parse_axis(spec: str) -> tuple[float, float, float]:
    """
    Parse a signed-axis spec like 'x', '-x', '+y', 'z' into a unit vector.
    Lets an orientation hypothesis (e.g. "AP beam is actually along Z, not
    Y") be tested directly against a render instead of hard-coded and
    guessed at.
    """
    s = spec.strip().lower()
    sign = -1.0 if s.startswith("-") else 1.0
    axis = s.lstrip("+-")
    if axis not in ("x", "y", "z"):
        raise argparse.ArgumentTypeError(
            f"invalid axis spec {spec!r} — use one of: x -x +x y -y +y z -z +z"
        )
    vec = [0.0, 0.0, 0.0]
    vec["xyz".index(axis)] = sign
    return tuple(vec)


def _build_arg_parser() -> argparse.ArgumentParser:
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
    p.add_argument("--ap-beam", type=parse_axis, default=None,
                    help="Override AP beam direction, e.g. 'z' or '-x'. "
                         f"Default from config.py: {cfg.GVXR_AP_BEAM_DIR}.")
    p.add_argument("--ap-up", type=parse_axis, default=None,
                    help="Override AP up-vector. "
                         f"Default from config.py: {cfg.GVXR_AP_UP}.")
    p.add_argument("--lat-beam", type=parse_axis, default=None,
                    help="Override lateral beam direction. "
                         f"Default from config.py: {cfg.GVXR_LAT_BEAM_DIR}.")
    p.add_argument("--lat-up", type=parse_axis, default=None,
                    help="Override lateral up-vector. "
                         f"Default from config.py: {cfg.GVXR_LAT_UP}.")
    p.add_argument("--shell-core-split", action="store_true",
                    help="Split the clipped mesh into a cortical shell + "
                         "trabecular core (two densities) instead of one "
                         "uniform-density mesh. See split_shell_core().")
    p.add_argument("--shell-mm", type=float, default=1.0,
                    help="Cortical shell thickness for --shell-core-split. "
                         "Default 1.0mm (see Chawla/Odeh pedicle+posterior-"
                         "element data discussed for this project).")
    p.add_argument("--shell-pitch-mm", type=float, default=0.2,
                    help="Voxel pitch for --shell-core-split. Finer = more "
                         "accurate, slower. Default 0.2mm.")
    p.add_argument("--trabecular-density", type=float,
                    default=getattr(cfg, "SPINE_TRABECULAR_DENSITY", 0.2),
                    help=f"Trabecular core density, g/cm3. Default: "
                         f"{getattr(cfg, 'SPINE_TRABECULAR_DENSITY', 0.2)} "
                         f"(from config.py; literature range is roughly "
                         f"0.09-0.35 g/cm3 for vertebral trabecular "
                         f"apparent density).")
    p.add_argument("--shell-target-faces", type=int, default=30000,
                    help="Decimate shell/core meshes to ~this many faces "
                         "each after marching cubes (which produces far "
                         "more detail than needed). Default 30000.")
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def prompt_args() -> argparse.Namespace:
    """
    Interactive fallback for Spyder's Run button (F5) — when the script is
    launched that way, sys.argv has no extra arguments, and configuring
    Spyder's "command line arguments" run-config each time is exactly the
    friction this is meant to avoid.

    Prompts only for the handful of parameters actually worth changing
    run-to-run right now (STL path, clip depth/axis, shell/core split).
    Everything else — beam axes, pixel size, photon count, energy — keeps
    the exact defaults from _build_arg_parser(), i.e. the values already
    confirmed working and baked into config.py; those are still reachable
    via real --flags from a terminal if you ever need to override them,
    this path just isn't the place to re-litigate them every run.

    Press Enter on any prompt to keep the default shown in [brackets].
    """
    args = _build_arg_parser().parse_args([])  # every field at its default

    def ask(prompt: str, default, cast=str):
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return cast(raw)
        except Exception:
            print(f"  couldn't parse {raw!r} — keeping default {default!r}")
            return default

    print("=" * 60)
    print("FluoroSim — STL X-ray Preview (interactive mode)")
    print("No command-line arguments detected — prompting instead.")
    print("Press Enter to accept each default. Pass --flags from a")
    print("terminal to skip this and use the non-interactive CLI.")
    print("=" * 60)

    stl_default = args.stl_path or cfg.SPINE_STL_PATH
    args.stl_path = ask("STL path", stl_default, str)
    args.clip_mm = ask("Clip depth, mm (0 = no clip)", args.clip_mm, float)
    args.no_clip = args.clip_mm <= 0
    if not args.no_clip:
        args.up_axis = ask("Up axis (x/y/z)", args.up_axis, str)

    do_split = ask("Shell/core split? (y/n)", "n", str).lower().startswith("y")
    args.shell_core_split = do_split
    if do_split:
        args.shell_mm = ask("Shell thickness, mm", args.shell_mm, float)
        args.trabecular_density = ask(
            "Trabecular density, g/cm3", args.trabecular_density, float)

    print()
    return args


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


def _ensure_outward(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """
    Force outward-facing normals (positive signed volume). is_watertight()
    passes regardless of winding direction, so an inside-out mesh can slip
    through that check silently — caught this happening after decimation in
    testing (a shell mesh reporting -237705.5 mm3), so this is called at
    every point a mesh is about to be exported, not just after decimation.
    """
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _decimate_and_repair(mesh: "trimesh.Trimesh", target_faces: int, label: str = "mesh") -> "trimesh.Trimesh":
    """
    Reduce a mesh to ~target_faces and repair the watertightness that
    quadric decimation reliably breaks (confirmed in testing: decimating a
    dense marching-cubes mesh left it non-watertight every time, and
    trimesh's own merge_vertices/fix_normals/fill_holes did not recover it).
    pymeshfix — already proven earlier in this project for exactly this
    kind of defect — does recover it, at negligible volume cost (<0.1% in
    testing).
    """
    if len(mesh.faces) <= target_faces:
        return _ensure_outward(mesh)
    import pymeshfix
    print(f"  Decimating {label} ({len(mesh.faces)} -> ~{target_faces} faces)... ",
          end="", flush=True)
    t0 = time.perf_counter()
    decimated = mesh.simplify_quadric_decimation(face_count=target_faces)
    v = np.ascontiguousarray(decimated.vertices, dtype=np.float64)
    f = np.ascontiguousarray(decimated.faces, dtype=np.int32)
    tin = pymeshfix.PyTMesh()
    tin.set_quiet(True)
    tin.load_array(v, f)
    tin.join_closest_components()
    tin.fill_small_boundaries()
    tin.clean(max_iters=10, inner_loops=3)
    v2, f2 = tin.return_arrays()
    fixed = trimesh.Trimesh(vertices=v2, faces=f2)
    fixed = _ensure_outward(fixed)
    print(f"done ({time.perf_counter() - t0:.1f}s, {len(fixed.faces)} faces)")
    if not fixed.is_watertight:
        raise RuntimeError(
            f"Decimation to {target_faces} faces could not be repaired to "
            "watertight. Try a higher target_faces."
        )
    return fixed


def split_shell_core(
    stl_path: Path,
    shell_mm: float,
    out_shell_path: Path,
    out_core_path: Path,
    pitch: float = 0.2,
    target_faces: int = 30000,
) -> tuple[Path, Path | None]:
    """
    Split a mesh into a cortical SHELL (within shell_mm of every surface
    point) and a trabecular CORE (everything deeper), saving each as its
    own STL.

    Method: voxelize the solid, compute a Euclidean distance transform from
    the boundary, threshold at shell_mm to get the core region, extract the
    core as a mesh via marching cubes, then get the shell via manifold3d
    boolean difference (outer minus core) — reusing the same boolean engine
    already validated for platform-plate clipping, rather than a naive
    normal-offset (which self-intersects on thin, concave geometry).

    This degrades gracefully exactly where a naive offset would break: if a
    local cross-section is thinner than 2*shell_mm (e.g. a thin pedicle
    wall), the distance transform never exceeds shell_mm there, so that
    region simply has NO core — i.e. it's correctly classified as fully
    cortical, with no special-casing required. Verified against a tapering
    wedge test case (thick end retains a core, thin end doesn't) before
    this was wired in.

    Returns (shell_path, core_path). core_path is None if no voxel in the
    entire mesh is farther than shell_mm from every surface (the whole
    object is thinner than 2*shell_mm everywhere) — in that case there's no
    meaningful trabecular compartment and only a shell should be used.

    pitch: voxel size in mm. Finer = more accurate but slower/more memory;
    0.2mm was sufficient in testing to resolve ~1mm shell thickness against
    a ~1180-face test mesh with <10% volume discretization error. If your
    real spine mesh is much larger, consider profiling before committing to
    a pitch — this is a real memory/time tradeoff, not a fixed constant.

    target_faces: marching cubes on a voxel grid produces far more faces
    than the shape needs (a small test object came out at 620k+ faces at
    pitch=0.3mm) — decimated down to this count for both shell and core
    before export. Quadric decimation reliably breaks watertightness on its
    own (confirmed in testing); this repairs it with pymeshfix afterward
    rather than trusting the decimated output directly.
    """
    if not TRIMESH_AVAILABLE:
        raise RuntimeError("trimesh is required. pip install trimesh manifold3d "
                            "scikit-image scipy --break-system-packages")
    try:
        from scipy import ndimage
        from skimage import measure
    except ImportError:
        raise RuntimeError("scikit-image and scipy are required for shell/core "
                            "splitting. pip install scikit-image scipy "
                            "--break-system-packages")

    mesh = trimesh.load_mesh(str(stl_path), force="mesh")
    if not mesh.is_watertight:
        raise RuntimeError(
            f"{stl_path} is not watertight — shell/core split needs a valid "
            "closed solid, same requirement as clip_bottom()."
        )

    print(f"  Voxelizing at {pitch}mm pitch (this is the slow step — a few "
          "minutes is normal on a real spine-sized mesh, not a hang)... ",
          end="", flush=True)
    t0 = time.perf_counter()
    vox = mesh.voxelized(pitch=pitch).fill()
    occ = vox.matrix.astype(bool)
    print(f"done ({time.perf_counter() - t0:.1f}s, "
          f"{occ.shape[0]}x{occ.shape[1]}x{occ.shape[2]} voxels)")

    print("  Computing distance transform... ", end="", flush=True)
    t0 = time.perf_counter()
    # Pad with a false border so marching_cubes always sees a closed boundary
    # (without this, a core region touching the array edge would produce an
    # open, non-watertight surface).
    occ_p = np.pad(occ, 1, mode="constant", constant_values=False)
    dist_mm = ndimage.distance_transform_edt(occ_p) * pitch
    core_mask = dist_mm > shell_mm
    print(f"done ({time.perf_counter() - t0:.1f}s)")

    out_shell_path.parent.mkdir(parents=True, exist_ok=True)

    if not core_mask.any():
        print(f"  [info] no voxel exceeds {shell_mm} mm from every surface — "
              "this mesh is thinner than 2x shell everywhere; treating the "
              "whole thing as cortical, no core.")
        _ensure_outward(mesh).export(str(out_shell_path))
        return out_shell_path, None

    print("  Extracting core surface (marching cubes)... ", end="", flush=True)
    t0 = time.perf_counter()
    verts, faces, _, _ = measure.marching_cubes(core_mask.astype(float), level=0.5)
    core_verts_world = vox.indices_to_points(verts - 1.0)  # undo the pad offset
    core = trimesh.Trimesh(vertices=core_verts_world, faces=faces)
    core.merge_vertices()
    core.fix_normals()  # marching_cubes winding direction isn't guaranteed
    core = _ensure_outward(core)
    print(f"done ({time.perf_counter() - t0:.1f}s, {len(core.faces)} raw faces)")

    core = _decimate_and_repair(core, target_faces, label="core")

    if not core.is_watertight:
        raise RuntimeError(
            "Extracted core is not watertight — try a finer --pitch. This "
            "would reproduce the earlier white-render failure if used as-is."
        )

    print("  Boolean difference (shell = outer - core)... ", end="", flush=True)
    t0 = time.perf_counter()
    shell = trimesh.boolean.difference([mesh, core], engine="manifold")
    shell = _ensure_outward(shell)
    print(f"done ({time.perf_counter() - t0:.1f}s, {len(shell.faces)} faces)")
    if not shell.is_watertight:
        raise RuntimeError(
            "Boolean difference (shell = outer - core) produced a "
            "non-watertight result. Try a finer --pitch."
        )
    # NOT decimating the shell: confirmed in testing that decimating a thin
    # (~1mm) sheet is a much harder case for quadric decimation than a
    # thick solid, and can silently collapse the inner/outer surfaces
    # together — is_watertight() still passes on the collapsed result, so
    # it doesn't get caught by that check; it showed up as shell volume
    # coming out equal to core volume, which is geometrically impossible
    # for a thin shell. The shell doesn't need decimating anyway: its face
    # count comes from `mesh` (the original, already-reasonable clipped
    # STL) minus the now-small decimated core, not from marching cubes —
    # core is the only thing that gets inflated to 600k+ faces.

    out_core_path.parent.mkdir(parents=True, exist_ok=True)
    shell.export(str(out_shell_path))
    core.export(str(out_core_path))
    print(f"  Shell: {out_shell_path}  ({len(shell.faces)} faces, "
          f"volume {shell.volume:.1f} mm3)")
    print(f"  Core:  {out_core_path}  ({len(core.faces)} faces, "
          f"volume {core.volume:.1f} mm3)")
    return out_shell_path, out_core_path


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
    ap_beam: tuple[float, float, float] | None = None,
    ap_up: tuple[float, float, float] | None = None,
    lat_beam: tuple[float, float, float] | None = None,
    lat_up: tuple[float, float, float] | None = None,
    shell_path: Path | None = None,
    core_path: Path | None = None,
    trabecular_compound: str = "Ca10(PO4)6(OH)2",
    trabecular_density: float = 0.35,
) -> SimpleNamespace:
    """
    Build a lightweight config namespace for XRaySimulator that renders ONLY
    the given STL — no platform, no bearings, no board registration.
    Reuses config.py's AP/LAT beam directions and context type so the
    preview matches the real pipeline's viewing convention, unless
    overridden (ap_beam/ap_up/lat_beam/lat_up) for testing an orientation
    hypothesis.

    shell_path/core_path: if both given (from split_shell_core()), the
    renderer uses the two-material shell+core mode instead of a single
    uniform-density mesh. trabecular_density defaults to 0.35 g/cm3 — the
    top of the 0.09-0.35 g/cm3 range reported across studies of vertebral
    trabecular apparent (wet) density (raised from an initial 0.2 midpoint
    after visual comparison favored more contrast — still within the cited
    range, not an arbitrary number).
    """
    return SimpleNamespace(
        GVXR_CONTEXT=cfg.GVXR_CONTEXT,
        SPINE_STL_PATH=str(stl_path),
        SPINE_SHELL_STL_PATH=str(shell_path) if shell_path else "",
        SPINE_CORE_STL_PATH=str(core_path) if core_path else "",
        PLATFORM_STL_PATH="",
        SPINE_MATERIAL_COMPOUND=compound,
        SPINE_MATERIAL_DENSITY=density,
        SPINE_TRABECULAR_COMPOUND=trabecular_compound,
        SPINE_TRABECULAR_DENSITY=trabecular_density,
        SPINE_ORIGIN_IN_WORLD=(0.0, 0.0, 0.0),
        BEARING_POSITIONS=[],
        GVXR_ISOCENTER=tuple(float(v) for v in isocenter_local),
        GVXR_SOD_MM=sod_mm,
        GVXR_DET_OFFSET_MM=det_offset_mm,
        GVXR_AP_BEAM_DIR=ap_beam if ap_beam is not None else cfg.GVXR_AP_BEAM_DIR,
        GVXR_AP_UP=ap_up if ap_up is not None else cfg.GVXR_AP_UP,
        GVXR_LAT_BEAM_DIR=lat_beam if lat_beam is not None else cfg.GVXR_LAT_BEAM_DIR,
        GVXR_LAT_UP=lat_up if lat_up is not None else cfg.GVXR_LAT_UP,
        GVXR_DETECTOR_PIXELS=(pixels, pixels),
        GVXR_PIXEL_SIZE_MM=pixel_size_mm,
        GVXR_ENERGY_MEV=energy_mev,
        GVXR_PHOTON_COUNT=photon_count,
    )


def main() -> int:
    args = prompt_args() if len(sys.argv) == 1 else parse_args()

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

    shell_path = core_path = None
    if args.shell_core_split:
        print(f"Shell/core    : splitting at {args.shell_mm}mm "
              f"(pitch {args.shell_pitch_mm}mm)")
        try:
            shell_path, core_path = split_shell_core(
                render_path, args.shell_mm,
                outdir / "spine_shell.stl", outdir / "spine_core.stl",
                pitch=args.shell_pitch_mm, target_faces=args.shell_target_faces,
            )
        except RuntimeError as exc:
            print(f"[FAILED] {exc}", file=sys.stderr)
            return 1
        if core_path is None:
            print("  No trabecular core found — whole mesh is thinner than "
                  f"2x{args.shell_mm}mm everywhere; rendering as fully "
                  "cortical (falling back to single-material mode).")
            shell_path = None  # fall back to build_preview_cfg's default path

    ap_beam = args.ap_beam if args.ap_beam is not None else cfg.GVXR_AP_BEAM_DIR
    ap_up = args.ap_up if args.ap_up is not None else cfg.GVXR_AP_UP
    lat_beam = args.lat_beam if args.lat_beam is not None else cfg.GVXR_LAT_BEAM_DIR
    lat_up = args.lat_up if args.lat_up is not None else cfg.GVXR_LAT_UP
    print(f"AP  beam/up   : {ap_beam} / {ap_up}"
          f"{'  (override)' if args.ap_beam is not None or args.ap_up is not None else '  (from config.py)'}")
    print(f"LAT beam/up   : {lat_beam} / {lat_up}"
          f"{'  (override)' if args.lat_beam is not None or args.lat_up is not None else '  (from config.py)'}")

    preview_cfg = build_preview_cfg(
        render_path, args.compound, args.density,
        args.pixels, args.pixel_size_mm, args.energy_mev, args.photon_count,
        args.sod_mm, args.det_offset_mm, isocenter_local,
        ap_beam=args.ap_beam, ap_up=args.ap_up,
        lat_beam=args.lat_beam, lat_up=args.lat_up,
        shell_path=shell_path, core_path=core_path,
        trabecular_density=args.trabecular_density,
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
