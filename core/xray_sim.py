"""
core/xray_sim.py — Synthetic X-ray simulator using gVXR 2.x.

Replaces the DLT / real-fluoroscopy pipeline entirely.

Design principles
-----------------
- gVXR generates AP + lateral X-ray images from STL with exact pinhole
  geometry; every 3-D → 2-D projection is analytically known.
- Background (spine + bearings) is rendered ONCE at startup and cached.
- Live probe overlay uses analytic projection  (< 1 ms, no re-render).
- Snapshot "photorealistic" mode adds a K-wire cylinder and re-renders.

gVXR 2.1 initialisation order (critical — deviating causes zero images)
------------------------------------------------------------------------
  1. createNewContext
  2. makeMesh / setElement / setDensity / translateNode  (scenegraph)
  3. setSourcePosition → usePointSource
  4. resetBeamSpectrum → addEnergyBinToSpectrumPerPixelAtSDD
  5. setDetectorPosition → setDetectorUpVector
  6. setDetectorNumberOfPixels → setDetectorPixelSize
  7. addPolygonMeshAsOuterSurface  (register mesh with renderer)
  8. computeXRayImage

  To switch views: redo steps 3–6 with new geometry, then computeXRayImage.
  Meshes registered in step 7 persist across view switches.

Coordinate system: millimetres, matching CAD / Fusion model space.

Coordinate chain (zero physical calibration required)
------------------------------------------------------
    ChArUco board frame
        ↓  BOARD_TO_WORLD  (4×4 rigid from CAD — set in config.py)
    gVXR world = CAD platform origin
        ↓  SPINE_TO_WORLD  (4×4 rigid from CAD — set in config.py; the spine
        ↓                   STL's local axes are NOT world-aligned, rotation
        ↓                   confirmed via two independent corner fits)
    Spine STL local frame
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── gVXR import (graceful degradation) ──────────────────────────────────────
try:
    from gvxrPython3 import gvxr as _gvxr
    GVXR_AVAILABLE = True
except ImportError:
    _gvxr = None           # type: ignore
    GVXR_AVAILABLE = False
    logger.warning(
        "gvxrPython3 not found — synthetic X-ray disabled.\n"
        "Install with:  pip install gvxr"
    )

# gVXR permits exactly ONE OpenGL context per process. createNewContext()
# raises RuntimeError if called a second time after destroyAllWindows(), and
# every retry after that hangs. This flag makes context creation idempotent
# across however many XRaySimulator instances the app builds (the launcher's
# verification step and the navigation window each construct one).
_CONTEXT_CREATED = False


def _clip_baseplate(
    stl_path: Path,
    clip_mm: float,
    up_axis: str,
    out_path: Path,
) -> Tuple[Path, Optional[np.ndarray]]:
    """
    Remove the bottom `clip_mm` of the mesh along local `up_axis` and write the
    result to `out_path`. Same operation (and same defaults) as
    tools/stl_xray_preview.py's clip_bottom(), so the live renderer and the
    validated preview see the same geometry.

    CRITICAL: this does NOT recentre or reorient the mesh — the clipped STL
    stays in the original spine-local frame, so BOARD_TO_SPINE remains valid.

    Returns (path_to_render, clipped_bounds) where clipped_bounds is a (2, 3)
    [min_xyz, max_xyz] array, or (original_path, None) if clipping was skipped
    or failed (in which case the caller renders the unclipped mesh).
    """
    if clip_mm <= 0:
        return stl_path, None

    # ── Cache: skip the (slow) boolean clip if a valid result already exists ──
    # Valid = cached file is newer than the source STL AND was produced with the
    # same clip parameters (recorded in a sidecar file next to it).
    stamp_path = out_path.with_suffix(".params.json")
    want_stamp = {"clip_mm": float(clip_mm), "axis": str(up_axis).lower(),
                  "src": str(stl_path), "src_mtime": stl_path.stat().st_mtime
                  if stl_path.exists() else None}
    if out_path.exists() and stamp_path.exists():
        try:
            import json
            if json.loads(stamp_path.read_text()) == want_stamp:
                import trimesh
                bounds = np.asarray(
                    trimesh.load_mesh(str(out_path), force="mesh").bounds,
                    dtype=np.float64)
                logger.info("Using cached clipped spine: %s", out_path)
                return out_path, bounds
        except Exception:
            logger.info("Clipped-spine cache unreadable — regenerating.")

    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh not installed — cannot clip baseplate. "
                       "Rendering spine.stl unclipped (baseplate WILL be "
                       "visible). Install with: pip install trimesh")
        return stl_path, None

    try:
        mesh = trimesh.load_mesh(str(stl_path), force="mesh")
        axis_idx = {"x": 0, "y": 1, "z": 2}[up_axis.lower()]
        bounds = mesh.bounds
        extent = float(bounds[1][axis_idx] - bounds[0][axis_idx])
        if clip_mm >= extent:
            logger.error("SPINE_CLIP_MM (%.1f) >= mesh extent along %s "
                         "(%.1f mm) — clip would delete the whole model. "
                         "Rendering unclipped; check SPINE_CLIP_AXIS.",
                         clip_mm, up_axis, extent)
            return stl_path, None

        plane_normal = np.zeros(3)
        plane_normal[axis_idx] = 1.0
        plane_origin = bounds[0].copy()
        plane_origin[axis_idx] = bounds[0][axis_idx] + clip_mm

        clipped = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=plane_normal, plane_origin=plane_origin, cap=True)
        if clipped is None or len(clipped.vertices) == 0:
            logger.error("Baseplate clip produced an empty mesh — rendering "
                         "unclipped. Check SPINE_CLIP_AXIS (%s).", up_axis)
            return stl_path, None

        out_path.parent.mkdir(parents=True, exist_ok=True)
        clipped.export(str(out_path))
        try:
            import json
            stamp_path.write_text(json.dumps(want_stamp))
        except Exception:
            logger.warning("Could not write clip cache stamp — the clip will "
                           "be recomputed on every start.")
        logger.info("Clipped %.1f mm off local %s: %s -> %s",
                    clip_mm, up_axis, stl_path, out_path)
        return out_path, np.asarray(clipped.bounds, dtype=np.float64)

    except Exception:
        logger.exception("Baseplate clipping failed — rendering unclipped.")
        return stl_path, None


def _draw_dashed_line(img, p1, p2, color, thickness=1, dash=8, gap=6):
    """Draw a dashed line between two pixel points (OpenCV has no native one)."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    total = float(np.linalg.norm(p2 - p1))
    if total < 1e-6:
        return
    direction = (p2 - p1) / total
    pos = 0.0
    while pos < total:
        a = p1 + direction * pos
        b = p1 + direction * min(pos + dash, total)
        cv2.line(img, (int(round(a[0])), int(round(a[1]))),
                 (int(round(b[0])), int(round(b[1]))),
                 color, thickness, cv2.LINE_AA)
        pos += dash + gap


def _axis_angle_from_matrix(R: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Convert a 3x3 proper rotation matrix (det=+1) to (angle_degrees, axis).
    General Rodrigues-formula inversion, handles the near-identity and
    near-180-degree edge cases (both degenerate in the naive formula).
    """
    R = np.asarray(R, dtype=np.float64)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-8:
        return 0.0, np.array([0.0, 0.0, 1.0])  # no rotation; axis arbitrary

    if theta > np.pi - 1e-6:
        # Near 180 deg: off-diagonal formula is ill-conditioned. Extract the
        # axis from the symmetric part instead (eigenvector for eigenvalue +1).
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        # Fix signs using the off-diagonal terms
        if axis[0] > 1e-8:
            axis[1] *= np.sign(A[0, 1])
            axis[2] *= np.sign(A[0, 2])
        elif axis[1] > 1e-8:
            axis[2] *= np.sign(A[1, 2])
        return float(np.degrees(theta)), axis / np.linalg.norm(axis)

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(theta))
    return float(np.degrees(theta)), axis


def _place_node_rigid(label: str, T: np.ndarray):
    """
    Apply a full 4x4 rigid transform (local -> world) to an already-created
    gVXR node, via rotateNode (about the node's local origin) followed by
    translateNode. Mirrors the order already used for the K-wire cylinder
    in render_snapshot_with_probe(), which is the order gVXR expects:
    rotation is applied first (about local origin), then the translation
    places the rotated node at its final world position.
    """
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    angle_deg, axis = _axis_angle_from_matrix(R)
    if abs(angle_deg) > 1e-6:
        _gvxr.rotateNode(label, angle_deg, *axis)
    _gvxr.translateNode(label, *[float(v) for v in t], "mm")


# ─────────────────────────────────────────────────────────────────────────────
# VirtualCArm  — geometry of one virtual X-ray view
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VirtualCArm:
    """
    Defines source + detector geometry for one virtual C-arm position.
    Detector axes are derived orthonormally and match gVXR's internal frame.
    All positions in mm, gVXR world (= CAD platform) space.
    """

    label:          str
    source_pos:     np.ndarray          # (3,) source centre, mm
    detector_pos:   np.ndarray          # (3,) detector centre, mm
    detector_up:    np.ndarray          # (3,) up-vector hint
    n_pixels:       Tuple[int, int] = (512, 512)
    pixel_size_mm:  float             = 0.5

    # Derived — set by __post_init__
    _normal:   np.ndarray = field(init=False, repr=False)
    _right:    np.ndarray = field(init=False, repr=False)
    _up_ortho: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.source_pos   = np.asarray(self.source_pos,   dtype=np.float64)
        self.detector_pos = np.asarray(self.detector_pos, dtype=np.float64)
        self.detector_up  = np.asarray(self.detector_up,  dtype=np.float64)
        self._recompute_axes()

    def _recompute_axes(self):
        fwd = self.detector_pos - self.source_pos
        self._normal   = fwd / np.linalg.norm(fwd)
        up             = self.detector_up / np.linalg.norm(self.detector_up)
        self._right    = np.cross(self._normal, up)
        self._right   /= np.linalg.norm(self._right)
        self._up_ortho = np.cross(self._right, self._normal)
        self._up_ortho /= np.linalg.norm(self._up_ortho)

    @classmethod
    def from_beam(
        cls,
        label: str,
        isocenter: np.ndarray,
        beam_dir: np.ndarray,
        up_hint: np.ndarray,
        n_pixels: Tuple[int, int],
        pixel_size_mm: float,
        sod_mm: float = 1000.0,
        det_offset_mm: float = 50.0,
    ) -> "VirtualCArm":
        """
        Build a C-arm from a beam DIRECTION rather than absolute positions.

        beam_dir : unit vector along which X-rays travel (source → detector).
        isocenter: 3-D point the beam passes through (centre of the image).
        The source sits sod_mm behind the isocenter; the detector det_offset_mm
        in front of it. A structure whose long axis is parallel to beam_dir
        projects to a point; one perpendicular projects to its full length.
        """
        iso = np.asarray(isocenter, dtype=np.float64)
        d   = np.asarray(beam_dir,  dtype=np.float64)
        d  /= np.linalg.norm(d)
        source   = iso - d * sod_mm
        detector = iso + d * det_offset_mm
        return cls(label, source, detector, up_hint, n_pixels, pixel_size_mm)

    # ── Projection ─────────────────────────────────────────────────────

    def project_point(
        self, pt_world: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """
        Project a 3-D world point → detector pixel (col, row).
        Returns None if point is behind source or outside detector FOV.
        """
        d     = np.asarray(pt_world, dtype=np.float64) - self.source_pos
        denom = float(np.dot(d, self._normal))
        if abs(denom) < 1e-9:
            return None
        sod = float(np.dot(self.detector_pos - self.source_pos, self._normal))
        t   = sod / denom
        if t <= 0:
            return None
        hit   = self.source_pos + t * d
        delta = hit - self.detector_pos
        u_mm  = float(np.dot(delta, self._right))
        v_mm  = float(np.dot(delta, self._up_ortho))
        cx, cy = self.n_pixels[0] / 2.0, self.n_pixels[1] / 2.0
        col = int(round(cx + u_mm / self.pixel_size_mm))
        row = int(round(cy - v_mm / self.pixel_size_mm))   # image Y flipped
        if 0 <= col < self.n_pixels[0] and 0 <= row < self.n_pixels[1]:
            return (col, row)
        return None

    def project_point_unclipped(
        self, pt_world: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """
        Project a 3-D world point → detector pixel (col, row) WITHOUT the
        field-of-view test. Returns None only if the point is behind the source
        or parallel to the detector plane (i.e. genuinely unprojectable).

        Needed for drawing line segments: a segment can cross the visible image
        while both of its endpoints lie outside it, so endpoints must be
        projected first and the LINE clipped afterwards (see project_ray).
        """
        d     = np.asarray(pt_world, dtype=np.float64) - self.source_pos
        denom = float(np.dot(d, self._normal))
        if abs(denom) < 1e-9:
            return None
        sod = float(np.dot(self.detector_pos - self.source_pos, self._normal))
        t   = sod / denom
        if t <= 0:
            return None
        hit   = self.source_pos + t * d
        delta = hit - self.detector_pos
        u_mm  = float(np.dot(delta, self._right))
        v_mm  = float(np.dot(delta, self._up_ortho))
        cx, cy = self.n_pixels[0] / 2.0, self.n_pixels[1] / 2.0
        col = int(round(cx + u_mm / self.pixel_size_mm))
        row = int(round(cy - v_mm / self.pixel_size_mm))   # image Y flipped
        return (col, row)

    def project_ray(
        self,
        pt_a: np.ndarray,
        pt_b: np.ndarray,
        extend_beyond_b_mm: float = 0.0,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Project a 3-D ray segment to two pixel endpoints, CLIPPED to the
        detector rectangle.

        Previously this returned None if either endpoint fell outside the
        detector, which silently discarded almost every K-wire: the wire plus
        its trajectory extension is far longer than the detector's field of
        view, so at least one end is nearly always off-image. The visible
        portion is now clipped and returned instead.

        Returns None only if the segment does not intersect the image at all.
        """
        pt_a = np.asarray(pt_a, dtype=np.float64)
        pt_b = np.asarray(pt_b, dtype=np.float64)
        if extend_beyond_b_mm > 0:
            d = pt_b - pt_a
            n = np.linalg.norm(d)
            if n > 1e-6:
                pt_b = pt_b + (d / n) * extend_beyond_b_mm

        pa = self.project_point_unclipped(pt_a)
        pb = self.project_point_unclipped(pt_b)
        if pa is None or pb is None:
            return None

        # Clip the projected segment to the detector rect. cv2.clipLine returns
        # (retval, pt1, pt2) with retval False when the line misses entirely.
        w, h = int(self.n_pixels[0]), int(self.n_pixels[1])
        inside, p1, p2 = cv2.clipLine((0, 0, w, h), pa, pb)
        if not inside:
            return None
        return (tuple(int(v) for v in p1), tuple(int(v) for v in p2))


# ─────────────────────────────────────────────────────────────────────────────
# XRaySimulator
# ─────────────────────────────────────────────────────────────────────────────

class XRaySimulator:
    """
    Manages the gVXR scene and produces synthetic AP + lateral X-ray images.

    Typical lifecycle
    -----------------
    1.  sim = XRaySimulator(cfg)
    2.  ok  = sim.initialise()                     # build scene, ~1–3 s
    3.  ap_bg, lat_bg = sim.render_background()    # render once, ~2–5 s
    4.  ap  = sim.overlay_probe(ap_bg,  tip, base, view="AP")
        lat = sim.overlay_probe(lat_bg, tip, base, view="LAT")
    5.  sim.shutdown()
    """

    _COL_TIP   = (0, 255,  50)    # BGR — green
    _COL_SHAFT = (0, 210, 255)    # BGR — yellow-cyan

    def __init__(self, cfg):
        self._cfg  = cfg
        # (2, 3) [min_xyz, max_xyz] bounds of the spine mesh actually rendered,
        # in spine-local coords. Populated during _build_scenegraph(); used for
        # the auto-isocentre when GVXR_ISOCENTER is None.
        self._spine_bounds: Optional[np.ndarray] = None
        self._ap:  Optional[VirtualCArm] = None
        self._lat: Optional[VirtualCArm] = None
        self._ap_bg:  Optional[np.ndarray] = None
        self._lat_bg: Optional[np.ndarray] = None
        self._ready           = False
        self._kwire_in_scene  = False
        # Track all mesh labels registered with the X-ray renderer
        self._registered_meshes: list[str] = []

    # ── Lifecycle ────────────────────────────────────────────────────────

    def initialise(self) -> bool:
        """
        Create the gVXR context (once per process), build scenegraph, register
        meshes, set up views.

        IMPORTANT — gVXR allows exactly ONE OpenGL context per process, and
        createNewContext() raises RuntimeError if called again after the
        context has been destroyed. Every subsequent attempt then hangs.
        So the context is created at most once here and is NEVER torn down
        mid-session; see shutdown() vs destroy_context().
        """
        global _CONTEXT_CREATED
        if not GVXR_AVAILABLE:
            logger.error("gVXR not available — cannot initialise.")
            return False
        try:
            if not _CONTEXT_CREATED:
                # gVXR negotiates the correct OpenGL version on its own when
                # called with just the backend. Passing extra positional args
                # (window id / version / visibility) corrupts the version request
                # on Windows GLFW ("Invalid OpenGL version 0.3") — so we don't.
                ctx = getattr(self._cfg, 'GVXR_CONTEXT', 'OPENGL')
                _gvxr.createNewContext(ctx)
                _CONTEXT_CREATED = True
                logger.info("Created gVXR OpenGL context (once per process).")
            else:
                # Reusing the existing context: clear the previous scene so
                # meshes are not registered twice (which would double their
                # attenuation contribution).
                logger.info("Reusing existing gVXR context.")
                try:
                    _gvxr.removePolygonMeshesFromXRayRenderer()
                except Exception:
                    logger.warning("Could not clear previous meshes from the "
                                   "X-ray renderer — attenuation may be "
                                   "doubled if the scene was already built.")

            # ── Steps 1–2: build scenegraph (no renderer registration yet) ──
            self._build_scenegraph()

            # ── Steps 3–7: set up first view (AP) and register meshes ───────
            self._setup_views()
            self._configure_renderer(self._ap)
            self._register_all_meshes()

            # Reduces salt-and-pepper noise from numerical imperfections in
            # the mesh (degenerate triangles, inconsistent normals at a cut
            # face from a boolean clip) — gVXR's own docs describe this as
            # "always useful if there are dodgy meshes", which describes our
            # boolean-clipped spine STL exactly. GPU variant used for speed;
            # enableArtefactFilteringOnCPU() exists if this ever needs to be
            # more thorough at the cost of render time.
            try:
                _gvxr.enableArtefactFilteringOnGPU()
            except Exception:
                logger.warning("enableArtefactFilteringOnGPU() unavailable "
                                "in this gVXR build — continuing without it.")

            self._ready = True
            try:
                renderer = _gvxr.getOpenGlRenderer()
            except Exception:
                renderer = "unknown"
            logger.info("XRaySimulator ready — %s", renderer)
            return True

        except Exception:
            logger.exception("XRaySimulator.initialise() failed")
            return False

    def shutdown(self):
        """
        Release this simulator's scene, but deliberately KEEP the OpenGL
        context alive — gVXR cannot create a second one in the same process,
        so destroying it here would make every later render fail with
        RuntimeError and then hang. Safe to call repeatedly.

        Only destroy_context() actually tears the context down, and that must
        happen at most once, at true process exit.
        """
        self._ready = False
        self._kwire_in_scene = False

    @staticmethod
    def destroy_context():
        """
        Tear down the process-wide gVXR OpenGL context. Call at most ONCE, at
        real process exit. After this, no further rendering is possible in this
        process — a new context cannot be created.
        """
        global _CONTEXT_CREATED
        if GVXR_AVAILABLE and _CONTEXT_CREATED:
            try:
                _gvxr.destroyAllWindows()
            except Exception:
                pass
            _CONTEXT_CREATED = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Scene construction ───────────────────────────────────────────────

    def _build_scenegraph(self):
        """
        Step 2: Create all meshes, assign materials, set positions.
        Does NOT call addPolygonMeshAsOuterSurface — that happens after
        source / detector are configured (gVXR 2.1 ordering requirement).
        """
        cfg = self._cfg

        # ── Spine STL ──────────────────────────────────────────────────
        # Two modes, chosen automatically:
        #   (a) shell/core: SPINE_SHELL_STL_PATH + SPINE_CORE_STL_PATH both
        #       set and present on disk -> two meshes, two materials
        #       (cortical shell + trabecular core), from split_shell_core()
        #       in tools/stl_xray_preview.py.
        #   (b) single mesh: original behavior, completely unchanged,
        #       whenever (a)'s paths aren't both configured.
        shell_str = getattr(cfg, 'SPINE_SHELL_STL_PATH', '') or ''
        core_str  = getattr(cfg, 'SPINE_CORE_STL_PATH', '') or ''
        shell_path = Path(shell_str) if shell_str else None
        core_path  = Path(core_str) if core_str else None

        if shell_path and shell_path.exists() and core_path and core_path.exists():
            _gvxr.loadMeshFile("spine_shell", str(shell_path), "mm")
            _gvxr.setCompound("spine_shell", cfg.SPINE_MATERIAL_COMPOUND)
            _gvxr.setDensity ("spine_shell", cfg.SPINE_MATERIAL_DENSITY, "g/cm3")

            _gvxr.loadMeshFile("spine_core", str(core_path), "mm")
            _gvxr.setCompound("spine_core", getattr(
                cfg, 'SPINE_TRABECULAR_COMPOUND', cfg.SPINE_MATERIAL_COMPOUND))
            _gvxr.setDensity ("spine_core", getattr(
                cfg, 'SPINE_TRABECULAR_DENSITY', 0.2), "g/cm3")

            # RENDER FRAME = SPINE-LOCAL — both meshes stay at identity.
            # (Shell/core STLs come out of split_shell_core() already in the
            # spine-local frame, and are assumed pre-clipped of the baseplate;
            # SPINE_CLIP_MM is not applied to this branch.)
            self._registered_meshes.append("spine_shell")
            self._registered_meshes.append("spine_core")

            try:
                import trimesh
                self._spine_bounds = np.asarray(
                    trimesh.load_mesh(str(shell_path), force="mesh").bounds,
                    dtype=np.float64)
            except Exception:
                logger.warning("Could not read shell bounds for auto-isocentre; "
                               "set GVXR_ISOCENTER explicitly if framing is off.")
            logger.info("Loaded spine as shell+core: %s / %s", shell_path, core_path)

        else:
            spine_path = Path(cfg.SPINE_STL_PATH)
            if spine_path.exists():
                # Clip the mounting baseplate off first (spine-local frame,
                # no recentring) so the trainee sees anatomy only — same
                # operation tools/stl_xray_preview.py performs.
                render_path, clipped_bounds = _clip_baseplate(
                    spine_path,
                    float(getattr(cfg, "SPINE_CLIP_MM", 0.0)),
                    str(getattr(cfg, "SPINE_CLIP_AXIS", "z")),
                    Path("outputs/render_cache/spine_clipped.stl"),
                )
                _gvxr.loadMeshFile("spine", str(render_path), "mm")
                self._spine_bounds = clipped_bounds
                logger.info("Loaded spine STL: %s", render_path)
            else:
                logger.warning("Spine STL not found (%s) — using cuboid placeholder.", spine_path)
                _gvxr.makeCuboid("spine", 80, 60, 40, "mm")
                _gvxr.moveToCentre("spine")

            _gvxr.setCompound("spine", cfg.SPINE_MATERIAL_COMPOUND)
            _gvxr.setDensity ("spine", cfg.SPINE_MATERIAL_DENSITY, "g/cm3")

            # RENDER FRAME = SPINE-LOCAL: the spine stays at identity, exactly
            # as in tools/stl_xray_preview.py (whose images are validated as
            # anatomically correct). The probe is brought INTO this frame via
            # BOARD_TO_SPINE instead of pushing the spine out into world space
            # — which would also force re-deriving every beam vector.
            # SPINE_TO_WORLD is still the ground truth it's composed from; see
            # config.py's derived-transforms section.
            self._registered_meshes.append("spine")

        # ── Platform STL (optional) ────────────────────────────────────
        plat_str  = getattr(cfg, 'PLATFORM_STL_PATH', '') or ''
        plat_path = Path(plat_str) if plat_str else None
        if plat_path and plat_path.exists():
            _gvxr.loadMeshFile("platform", str(plat_path), "mm")
            _gvxr.setCompound("platform", "C2H4")   # PETG ≈ polyethylene
            _gvxr.setDensity ("platform", 1.27, "g/cm3")
            self._registered_meshes.append("platform")

        # ── Steel ball bearings (platform, not anatomy — off by default) ──
        # BEARING_POSITIONS are in world/CAD coordinates, so they must be
        # transformed into the spine-local render frame.
        if getattr(cfg, "RENDER_BEARINGS", False):
            w2s = np.asarray(getattr(cfg, "WORLD_TO_SPINE", np.eye(4)),
                             dtype=np.float64).reshape(4, 4)
            for i, b in enumerate(cfg.BEARING_POSITIONS):
                lbl = f"bearing_{i}"
                r   = float(b.get('radius_mm', 1.5))
                _gvxr.makeSphere(lbl, 20, 20, r, "mm")
                _gvxr.setElement (lbl, "Fe")
                _gvxr.setDensity (lbl, 7.87, "g/cm3")
                pos_world = np.append(
                    np.asarray(b['position_mm'], dtype=np.float64), 1.0)
                x, y, z = (w2s @ pos_world)[:3]
                _gvxr.translateNode(lbl, float(x), float(y), float(z), "mm")
                self._registered_meshes.append(lbl)

    def _setup_views(self):
        """Instantiate VirtualCArm objects from config beam directions."""
        cfg = self._cfg
        # Isocentre: explicit config value if given, else AUTO from the clipped
        # spine mesh's own bbox centre in spine-local coords — the same choice
        # tools/stl_xray_preview.py makes, so framing matches its known-good
        # output instead of depending on a hand-tuned constant.
        iso_cfg = getattr(cfg, 'GVXR_ISOCENTER', None)
        if iso_cfg is not None:
            iso = np.array(iso_cfg, dtype=np.float64)
        elif self._spine_bounds is not None:
            iso = np.asarray(self._spine_bounds, dtype=np.float64).mean(axis=0)
            logger.info("Isocentre AUTO from clipped spine bbox: %s", iso.round(2))
        else:
            iso = np.zeros(3)
            logger.warning("GVXR_ISOCENTER is None and spine bounds unknown — "
                           "falling back to origin. Framing may be off; set "
                           "GVXR_ISOCENTER explicitly or install trimesh.")
        sod   = float(getattr(cfg, 'GVXR_SOD_MM', 1000.0))
        doff  = float(getattr(cfg, 'GVXR_DET_OFFSET_MM', 50.0))
        npix  = tuple(cfg.GVXR_DETECTOR_PIXELS)
        psz   = float(cfg.GVXR_PIXEL_SIZE_MM)

        self._ap = VirtualCArm.from_beam(
            "AP", iso,
            np.array(cfg.GVXR_AP_BEAM_DIR, float),
            np.array(cfg.GVXR_AP_UP,       float),
            npix, psz, sod, doff,
        )
        self._lat = VirtualCArm.from_beam(
            "LAT", iso,
            np.array(cfg.GVXR_LAT_BEAM_DIR, float),
            np.array(cfg.GVXR_LAT_UP,        float),
            npix, psz, sod, doff,
        )

    def set_views_from_probe_axis(
        self,
        tip_world:  np.ndarray,
        base_world: np.ndarray,
    ):
        """
        Lock AP + lateral to the current probe trajectory.

        AP is aimed straight down the probe axis (probe → dot); lateral is
        placed 90° across it (probe → full-length line). The isocentre is set
        to the probe midpoint so the trajectory is centred in both panels.

        Call this once, with the probe held along the *ideal* trajectory, then
        the views stay fixed and any later deviation becomes visible — exactly
        as in intra-operative fluoroscopy.
        """
        cfg  = self._cfg
        tip  = np.asarray(tip_world,  dtype=np.float64)
        base = np.asarray(base_world, dtype=np.float64)
        axis = tip - base
        n = np.linalg.norm(axis)
        if n < 1e-6:
            return
        ap_dir = axis / n

        # Build an AP up-vector perpendicular to the beam.
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(ref, ap_dir))) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        ap_up = ref - ap_dir * float(np.dot(ref, ap_dir))
        ap_up /= np.linalg.norm(ap_up)

        # Lateral beam: perpendicular to the probe axis, so the probe projects
        # to its full length. Lateral "up" = probe axis, so depth reads vertically.
        lat_dir = np.cross(ap_dir, ap_up)
        lat_dir /= np.linalg.norm(lat_dir)
        lat_up  = ap_dir

        iso  = (tip + base) / 2.0
        sod  = float(getattr(cfg, 'GVXR_SOD_MM', 1000.0))
        doff = float(getattr(cfg, 'GVXR_DET_OFFSET_MM', 50.0))
        npix = tuple(cfg.GVXR_DETECTOR_PIXELS)
        psz  = float(cfg.GVXR_PIXEL_SIZE_MM)

        self._ap  = VirtualCArm.from_beam("AP",  iso, ap_dir,  ap_up,  npix, psz, sod, doff)
        self._lat = VirtualCArm.from_beam("LAT", iso, lat_dir, lat_up, npix, psz, sod, doff)
        self.invalidate_background()

    def _configure_renderer(self, view: VirtualCArm):
        """
        Steps 3–6: Configure source + energy + detector for one view.
        Must be called BEFORE addPolygonMeshAsOuterSurface.
        """
        cfg = self._cfg
        _gvxr.setSourcePosition(*view.source_pos, "mm")
        _gvxr.usePointSource()
        _gvxr.resetBeamSpectrum()
        _gvxr.addEnergyBinToSpectrumPerPixelAtSDD(
            float(cfg.GVXR_ENERGY_MEV), "MeV", int(cfg.GVXR_PHOTON_COUNT)
        )
        _gvxr.setDetectorPosition(*view.detector_pos, "mm")
        _gvxr.setDetectorUpVector(*view._up_ortho)
        _gvxr.setDetectorNumberOfPixels(*view.n_pixels)
        _gvxr.setDetectorPixelSize(view.pixel_size_mm, view.pixel_size_mm, "mm")

    def _register_all_meshes(self):
        """Step 7: Add all scenegraph meshes to the X-ray renderer."""
        for lbl in self._registered_meshes:
            _gvxr.addPolygonMeshAsOuterSurface(lbl)

    # ── Rendering ────────────────────────────────────────────────────────

    # ── Background cache (disk-backed) ───────────────────────────────────

    BACKGROUND_CACHE_DIR = Path("outputs/render_cache")

    def _background_cache_key(self) -> str:
        """
        Stable hash of everything that affects the background image. If any of
        these change the cache misses and we re-render; otherwise the cached
        PNGs are reused, including across process restarts.
        """
        import hashlib
        import json
        cfg = self._cfg
        spine = Path(getattr(cfg, "SPINE_STL_PATH", ""))

        def _lst(v):
            if v is None:
                return None
            if isinstance(v, np.ndarray):
                return v.round(6).tolist()
            if isinstance(v, (tuple, list)):
                return [float(x) for x in v]
            return v

        payload = {
            "stl":          str(spine),
            "stl_mtime":    spine.stat().st_mtime if spine.exists() else None,
            "clip_mm":      float(getattr(cfg, "SPINE_CLIP_MM", 0.0)),
            "clip_axis":    str(getattr(cfg, "SPINE_CLIP_AXIS", "z")),
            "iso_cfg":      _lst(getattr(cfg, "GVXR_ISOCENTER", None)),
            # resolved bounds matter because they drive the AUTO isocentre
            "spine_bounds": (np.asarray(self._spine_bounds).round(4).tolist()
                             if self._spine_bounds is not None else None),
            "sod":          float(cfg.GVXR_SOD_MM),
            "det_offset":   float(cfg.GVXR_DET_OFFSET_MM),
            "pixels":       _lst(cfg.GVXR_DETECTOR_PIXELS),
            "pixel_mm":     float(cfg.GVXR_PIXEL_SIZE_MM),
            "ap_beam":      _lst(cfg.GVXR_AP_BEAM_DIR),
            "ap_up":        _lst(cfg.GVXR_AP_UP),
            "lat_beam":     _lst(cfg.GVXR_LAT_BEAM_DIR),
            "lat_up":       _lst(cfg.GVXR_LAT_UP),
            "energy_mev":   float(cfg.GVXR_ENERGY_MEV),
            "photons":      int(cfg.GVXR_PHOTON_COUNT),
            "compound":     str(cfg.SPINE_MATERIAL_COMPOUND),
            "density":      float(cfg.SPINE_MATERIAL_DENSITY),
            "trabecular":   float(getattr(cfg, "SPINE_TRABECULAR_DENSITY", 0.0)),
            "bearings":     bool(getattr(cfg, "RENDER_BEARINGS", False)),
            "bearing_pos":  ([[float(x) for x in b["position_mm"]]
                              for b in cfg.BEARING_POSITIONS]
                             if getattr(cfg, "RENDER_BEARINGS", False) else None),
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def _background_cache_paths(self) -> Tuple[Path, Path]:
        key = self._background_cache_key()
        d = self.BACKGROUND_CACHE_DIR
        return d / f"bg_ap_{key}.png", d / f"bg_lat_{key}.png"

    def load_cached_background(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load the pre-rendered background from disk if one exists for the current
        configuration. Requires initialise() first (the cache key depends on the
        resolved spine bounds), but needs NO rendering — so navigation can start
        instantly from a background prepared earlier.

        Returns (None, None) on a cache miss.
        """
        try:
            ap_p, lat_p = self._background_cache_paths()
            if ap_p.exists() and lat_p.exists():
                ap = cv2.imread(str(ap_p), cv2.IMREAD_GRAYSCALE)
                lat = cv2.imread(str(lat_p), cv2.IMREAD_GRAYSCALE)
                if ap is not None and lat is not None:
                    self._ap_bg, self._lat_bg = ap, lat
                    logger.info("Loaded cached background: %s", ap_p.name)
                    return ap, lat
        except Exception:
            logger.exception("Failed reading background cache — will re-render.")
        return None, None

    def render_background(
        self,
        use_cache: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Render AP + lateral background images (spine + bearings, no probe).

        Caching is three-tiered:
          1. in-memory (this instance, free)
          2. on disk, keyed by config hash (survives restarts and hand-off from
             the launcher to the navigation window)
          3. actual gVXR render (~2-5 s), which then populates 1 and 2

        Pass use_cache=False to force a fresh render (e.g. to verify the GPU
        path still works).
        Returns (ap_image, lat_image) as 8-bit display images.
        """
        if not self._ready:
            return None, None

        if use_cache and self._ap_bg is not None:
            return self._ap_bg, self._lat_bg

        if use_cache:
            ap, lat = self.load_cached_background()
            if ap is not None:
                return ap, lat

        self._ap_bg  = self._render_view(self._ap)
        self._lat_bg = self._render_view(self._lat)

        # Persist for the next session / the navigation window.
        try:
            ap_p, lat_p = self._background_cache_paths()
            ap_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ap_p),  self._ap_bg)
            cv2.imwrite(str(lat_p), self._lat_bg)
            logger.info("Cached background to %s", ap_p.parent)
        except Exception:
            logger.warning("Could not write background cache — it will be "
                           "re-rendered next time.")

        return self._ap_bg, self._lat_bg

    def invalidate_background(self):
        """Force re-render on next render_background() call."""
        self._ap_bg = None
        self._lat_bg = None

    def _render_view(self, view: VirtualCArm) -> np.ndarray:
        """
        Switch gVXR to a given view and compute the X-ray.
        Steps 3–6 are re-run (spectrum resets on source change);
        meshes stay registered from initialise().
        """
        cfg = self._cfg
        _gvxr.setSourcePosition(*view.source_pos, "mm")
        _gvxr.usePointSource()
        _gvxr.resetBeamSpectrum()
        _gvxr.addEnergyBinToSpectrumPerPixelAtSDD(
            float(cfg.GVXR_ENERGY_MEV), "MeV", int(cfg.GVXR_PHOTON_COUNT)
        )
        _gvxr.setDetectorPosition(*view.detector_pos, "mm")
        _gvxr.setDetectorUpVector(*view._up_ortho)
        _gvxr.setDetectorNumberOfPixels(*view.n_pixels)
        _gvxr.setDetectorPixelSize(view.pixel_size_mm, view.pixel_size_mm, "mm")
        _gvxr.computeXRayImage()
        raw = np.array(_gvxr.getLastXRayImage())
        return self._to_display(raw)

    @staticmethod
    def _to_display(raw: np.ndarray) -> np.ndarray:
        """Log-compress energy fluence, invert → anatomy bright on dark background."""
        safe = np.clip(raw, 1e-10, None)
        log  = np.log(safe)
        norm = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.bitwise_not(norm)

    # ── Fast probe overlay (analytic projection, no re-render) ───────────

    def overlay_probe(
        self,
        background:       np.ndarray,
        probe_tip_world:  np.ndarray,
        probe_base_world: np.ndarray,
        view:             str   = "AP",
        extend_mm:        float = 80.0,
        color_tip:        Tuple[int, int, int] = _COL_TIP,
        color_shaft:      Tuple[int, int, int] = _COL_SHAFT,
        thickness:        int   = 2,
        tip_radius:       int   = 8,
    ) -> np.ndarray:
        """
        Draw probe trajectory on a pre-rendered background using analytic
        pinhole projection. No gVXR re-render — effectively instantaneous.

        Parameters
        ----------
        background       : uint8 image from render_background()
        probe_tip_world  : K-wire tip in mm, gVXR world space
        probe_base_world : K-wire base (cube centre) in mm, gVXR world space
        view             : "AP" or "LAT"
        extend_mm        : extra shaft drawn past the tip for visual clarity

        Returns
        -------
        BGR uint8 image with probe overlaid.
        """
        cam = self._ap if view == "AP" else self._lat
        if cam is None or background is None:
            return (background.copy() if background is not None
                    else np.zeros((512, 512, 3), np.uint8))

        tip  = np.asarray(probe_tip_world,  dtype=np.float64)
        base = np.asarray(probe_base_world, dtype=np.float64)

        out = (cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
               if background.ndim == 2 else background.copy())

        # ── Projected trajectory BEYOND the tip: where the wire is heading ──
        # Drawn dashed and thinner so it reads as prediction, not hardware.
        if extend_mm > 0:
            d = tip - base
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                far = tip + (d / n) * extend_mm
                seg_ext = cam.project_ray(tip, far)
                if seg_ext:
                    _draw_dashed_line(out, seg_ext[0], seg_ext[1],
                                      color_shaft, max(1, thickness - 1))

        # ── The K-wire itself: base → tip, solid and thicker ────────────────
        seg = cam.project_ray(base, tip)
        wire_len_px = 0.0
        if seg:
            # Dark border first, then bright core — reads like a radiopaque
            # wire against bone rather than a flat annotation line.
            cv2.line(out, seg[0], seg[1], (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.line(out, seg[0], seg[1], color_shaft, thickness + 1, cv2.LINE_AA)
            wire_len_px = float(np.hypot(seg[1][0] - seg[0][0],
                                         seg[1][1] - seg[0][1]))

        # ── Entry point (base) and tip markers ─────────────────────────────
        px_base = cam.project_point(base)
        if px_base:
            cv2.circle(out, px_base, 4, color_shaft, 1, cv2.LINE_AA)

        px_tip = cam.project_point(tip)
        if px_tip:
            cv2.circle(out, px_tip, tip_radius,     color_tip, -1,  cv2.LINE_AA)
            cv2.circle(out, px_tip, tip_radius + 2, (0, 0, 0),  1,  cv2.LINE_AA)

        depth_mm = float(np.linalg.norm(tip - base))
        cv2.putText(
            out, f"Depth: {depth_mm:.0f} mm", (12, out.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_tip, 2, cv2.LINE_AA,
        )

        # ── Down-the-barrel warning ────────────────────────────────────────
        # A wire nearly parallel to the beam foreshortens to a point. That is
        # correct physics and is diagnostically meaningful (it is how a true
        # "down the barrel" view looks), but without a note it looks identical
        # to a tracking failure. Threshold: real length >= 20 mm but projected
        # length < 15 px.
        if depth_mm >= 20.0 and wire_len_px < 15.0:
            cv2.putText(
                out, "wire ~parallel to beam (foreshortened)",
                (12, out.shape[0] - 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_tip, 1, cv2.LINE_AA,
            )
        return out

    # ── Photorealistic snapshot (K-wire as gVXR cylinder) ────────────────

    def render_snapshot_with_probe(
        self,
        probe_tip_world:  np.ndarray,
        probe_base_world: np.ndarray,
        kwire_radius_mm:  float = 1.0,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Render AP + lateral with the K-wire as an actual gVXR cylinder.
        Photorealistic (~2–10 s) — intended for final capture/save.
        Returns (ap_image, lat_image).
        """
        if not self._ready:
            return None, None

        tip  = np.asarray(probe_tip_world,  dtype=np.float64)
        base = np.asarray(probe_base_world, dtype=np.float64)
        d    = tip - base
        L    = float(np.linalg.norm(d))
        if L < 1e-3:
            return self._ap_bg, self._lat_bg

        self._remove_kwire()

        # Create cylinder along default +Y axis, length L, then rotate into place
        _gvxr.makeCylinder("kwire", 20, L, kwire_radius_mm, "mm")
        _gvxr.setElement("kwire", "Fe")
        _gvxr.setDensity ("kwire", 7.87, "g/cm3")

        unit   = d / L
        y_axis = np.array([0.0, 1.0, 0.0])
        axis   = np.cross(y_axis, unit)
        sin_a  = np.linalg.norm(axis)
        cos_a  = float(np.dot(y_axis, unit))
        if sin_a > 1e-6:
            axis /= sin_a
            _gvxr.rotateNode("kwire", float(np.degrees(np.arctan2(sin_a, cos_a))), *axis)
        elif cos_a < 0:
            _gvxr.rotateNode("kwire", 180.0, 1.0, 0.0, 0.0)

        centre = (base + tip) / 2.0
        _gvxr.translateNode("kwire", *centre, "mm")

        _gvxr.addPolygonMeshAsOuterSurface("kwire")
        self._registered_meshes.append("kwire")
        self._kwire_in_scene = True

        ap_snap  = self._render_view(self._ap)
        lat_snap = self._render_view(self._lat)
        return ap_snap, lat_snap

    def _remove_kwire(self):
        """Remove K-wire from X-ray renderer and scenegraph."""
        if not (self._kwire_in_scene and GVXR_AVAILABLE):
            return
        try:
            _gvxr.removePolygonMeshesFromXRayRenderer()
            self._registered_meshes = [l for l in self._registered_meshes
                                        if l != "kwire"]
            # Re-register remaining meshes
            for lbl in self._registered_meshes:
                _gvxr.addPolygonMeshAsOuterSurface(lbl)
        except Exception:
            pass
        self._kwire_in_scene = False

    # ── Coordinate transform ─────────────────────────────────────────────

    def board_to_world(self, pt_board: np.ndarray) -> np.ndarray:
        """
        Transform a point from ChArUco board frame → gVXR world (CAD) frame.
        Uses BOARD_TO_WORLD from config — 4×4 row-major rigid matrix
        read directly from the Fusion assembly.

        NOTE: world is NOT the render frame. For anything that will be placed
        into the gVXR scene (i.e. the probe / K-wire), use board_to_render().
        This method remains for CAD-space queries and validation only.
        """
        M  = np.asarray(self._cfg.BOARD_TO_WORLD, dtype=np.float64).reshape(4, 4)
        ph = np.append(np.asarray(pt_board, dtype=np.float64).flatten(), 1.0)
        return (M @ ph)[:3]

    def board_to_render(self, pt_board: np.ndarray) -> np.ndarray:
        """
        Transform a point from ChArUco board frame → the RENDER frame
        (spine-local), which is the frame every mesh in the gVXR scene lives
        in. This is the transform to use for tracked probe tip/base points.

        Uses BOARD_TO_SPINE from config, itself composed as
        inv(SPINE_TO_WORLD) @ BOARD_TO_WORLD — so it stays correct
        automatically if either source matrix is re-derived.
        """
        M = np.asarray(
            getattr(self._cfg, "BOARD_TO_SPINE", self._cfg.BOARD_TO_WORLD),
            dtype=np.float64).reshape(4, 4)
        ph = np.append(np.asarray(pt_board, dtype=np.float64).flatten(), 1.0)
        return (M @ ph)[:3]
