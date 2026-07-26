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

    def project_ray(
        self,
        pt_a: np.ndarray,
        pt_b: np.ndarray,
        extend_beyond_b_mm: float = 0.0,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Project a 3-D ray segment to two pixel endpoints.
        Returns None if either end falls outside the detector.
        """
        if extend_beyond_b_mm > 0:
            d = pt_b - pt_a
            n = np.linalg.norm(d)
            if n > 1e-6:
                pt_b = pt_b + (d / n) * extend_beyond_b_mm
        pa = self.project_point(pt_a)
        pb = self.project_point(pt_b)
        if pa is None or pb is None:
            return None
        return (pa, pb)


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
        """Create gVXR context, build scenegraph, register meshes, set up views."""
        if not GVXR_AVAILABLE:
            logger.error("gVXR not available — cannot initialise.")
            return False
        try:
            # gVXR negotiates the correct OpenGL version on its own when
            # called with just the backend. Passing extra positional args
            # (window id / version / visibility) corrupts the version request
            # on Windows GLFW ("Invalid OpenGL version 0.3") — so we don't.
            ctx = getattr(self._cfg, 'GVXR_CONTEXT', 'OPENGL')
            _gvxr.createNewContext(ctx)

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
        """Destroy gVXR context. Call once at application exit."""
        if GVXR_AVAILABLE:
            try:
                _gvxr.destroyAllWindows()
            except Exception:
                pass
        self._ready = False

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

            spine_to_world = np.asarray(
                getattr(cfg, "SPINE_TO_WORLD", np.eye(4)), dtype=np.float64
            ).reshape(4, 4)
            _place_node_rigid("spine_shell", spine_to_world)
            _place_node_rigid("spine_core",  spine_to_world)
            self._registered_meshes.append("spine_shell")
            self._registered_meshes.append("spine_core")
            logger.info("Loaded spine as shell+core: %s / %s", shell_path, core_path)

        else:
            spine_path = Path(cfg.SPINE_STL_PATH)
            if spine_path.exists():
                _gvxr.loadMeshFile("spine", str(spine_path), "mm")
                logger.info("Loaded spine STL: %s", spine_path)
            else:
                logger.warning("Spine STL not found (%s) — using cuboid placeholder.", spine_path)
                _gvxr.makeCuboid("spine", 80, 60, 40, "mm")
                _gvxr.moveToCentre("spine")

            _gvxr.setCompound("spine", cfg.SPINE_MATERIAL_COMPOUND)
            _gvxr.setDensity ("spine", cfg.SPINE_MATERIAL_DENSITY, "g/cm3")

            # Seat at the hard-stop position (from CAD) -- full rigid
            # transform, NOT translation-only: the spine STL's local axes
            # are rotated 120 deg about (1,-1,-1)/sqrt(3) relative to gVXR
            # world/CAD space (confirmed via two independent corner
            # correspondences, see SPINE_TO_WORLD derivation in config.py).
            spine_to_world = np.asarray(
                getattr(cfg, "SPINE_TO_WORLD", np.eye(4)), dtype=np.float64
            ).reshape(4, 4)
            _place_node_rigid("spine", spine_to_world)
            self._registered_meshes.append("spine")

        # ── Platform STL (optional) ────────────────────────────────────
        plat_str  = getattr(cfg, 'PLATFORM_STL_PATH', '') or ''
        plat_path = Path(plat_str) if plat_str else None
        if plat_path and plat_path.exists():
            _gvxr.loadMeshFile("platform", str(plat_path), "mm")
            _gvxr.setCompound("platform", "C2H4")   # PETG ≈ polyethylene
            _gvxr.setDensity ("platform", 1.27, "g/cm3")
            self._registered_meshes.append("platform")

        # ── Steel ball bearings ────────────────────────────────────────
        for i, b in enumerate(cfg.BEARING_POSITIONS):
            lbl = f"bearing_{i}"
            r   = float(b.get('radius_mm', 1.5))
            _gvxr.makeSphere(lbl, 20, 20, r, "mm")
            _gvxr.setElement (lbl, "Fe")
            _gvxr.setDensity (lbl, 7.87, "g/cm3")
            x, y, z = (float(v) for v in b['position_mm'])
            _gvxr.translateNode(lbl, x, y, z, "mm")
            self._registered_meshes.append(lbl)

    def _setup_views(self):
        """Instantiate VirtualCArm objects from config beam directions."""
        cfg = self._cfg
        iso   = np.array(getattr(cfg, 'GVXR_ISOCENTER', (0.0, 0.0, 0.0)), float)
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

    def render_background(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Render AP + lateral background images (spine + bearings, no probe).
        Expensive first call (~2–5 s); results are cached.
        Returns (ap_image, lat_image) as 8-bit display images.
        """
        if not self._ready:
            return None, None
        if self._ap_bg is None:
            self._ap_bg  = self._render_view(self._ap)
            self._lat_bg = self._render_view(self._lat)
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

        seg = cam.project_ray(base, tip, extend_beyond_b_mm=extend_mm)
        if seg:
            cv2.line(out, seg[0], seg[1], color_shaft, thickness, cv2.LINE_AA)

        px_tip = cam.project_point(tip)
        if px_tip:
            cv2.circle(out, px_tip, tip_radius,     color_tip, -1,  cv2.LINE_AA)
            cv2.circle(out, px_tip, tip_radius + 2, color_tip,  1,  cv2.LINE_AA)

        depth_mm = float(np.linalg.norm(tip - base))
        cv2.putText(
            out, f"Depth: {depth_mm:.0f} mm", (12, out.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_tip, 2, cv2.LINE_AA,
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
        """
        M  = np.asarray(self._cfg.BOARD_TO_WORLD, dtype=np.float64).reshape(4, 4)
        ph = np.append(np.asarray(pt_board, dtype=np.float64).flatten(), 1.0)
        return (M @ ph)[:3]
