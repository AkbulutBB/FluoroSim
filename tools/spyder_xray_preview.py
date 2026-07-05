"""
tools/spyder_xray_preview.py — Interactive, cell-based STL X-ray preview.

Same underlying pipeline as tools/stl_xray_preview.py (clip_bottom,
XRaySimulator, VirtualCArm — nothing is re-implemented here), just laid out
as Spyder cells (separated by "# %%") instead of a CLI script, so you can
tweak one value and re-run just the relevant cell instead of retyping a
whole command line each time.

How to use in Spyder
---------------------
Click inside a cell, press Ctrl+Enter to run just that cell (or Shift+Enter
to run it and jump to the next one). Run cells 1-4 in order once per
session — cell 4 is the slow part (creates the gVXR context, loads the
mesh, ~1-3 s). After that, cells 5 and 6 are fast and safe to re-run as
many times as you like:

    Cells 1-4  : run once at the start of your session (or after changing
                 CLIP_MM / UP_AXIS / STL_PATH / material, since those
                 require reloading the mesh).
    Cell 5     : edit AP_BEAM / AP_UP / LAT_BEAM / LAT_UP / PIXEL_SIZE_MM /
                 PHOTON_COUNT etc. here, then re-run.
    Cell 6     : re-render + display with whatever is currently set in
                 cell 5. This is the one you re-run over and over.
    Cell 7     : run once when you're completely done, to release the
                 OpenGL context cleanly.

Why this is safe to re-run fast: gVXR only needs the mesh-loading and
material steps done once. Changing the beam direction, up-vector, or pixel
size only touches the SOURCE/DETECTOR configuration, which gVXR is
designed to have redone per view — see the ordering notes in
core/xray_sim.py's docstring. Re-running cell 6 does NOT reload the STL or
redo the boolean clip.
"""

# %% [1] Imports and project path — run once
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Same convention as the other tools/ scripts: makes "import config" and
# "import core.xray_sim" work when Spyder runs this file directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config as cfg  # noqa: E402
from core.xray_sim import XRaySimulator, VirtualCArm, GVXR_AVAILABLE  # noqa: E402
from tools.stl_xray_preview import clip_bottom, build_preview_cfg  # noqa: E402

if not GVXR_AVAILABLE:
    print("WARNING: gvxrPython3 not importable in this kernel. Run "
          "tools/test_gvxr.py first, or restart the kernel if you just "
          "installed it (see the earlier trimesh/mapbox_earcut caching bug "
          "for why a stale kernel matters here too).")


# %% [2] Parameters — edit these freely, then re-run the cells below them
STL_PATH        = _PROJECT_ROOT / cfg.SPINE_STL_PATH
CLIP_MM         = 8.0      # platform attachment plate thickness
UP_AXIS         = "z"      # which local STL axis the plate sits at the min of

COMPOUND        = cfg.SPINE_MATERIAL_COMPOUND
DENSITY         = cfg.SPINE_MATERIAL_DENSITY

PIXELS          = 512
PIXEL_SIZE_MM   = 0.5
ENERGY_MEV      = cfg.GVXR_ENERGY_MEV
PHOTON_COUNT    = cfg.GVXR_PHOTON_COUNT
SOD_MM          = cfg.GVXR_SOD_MM
DET_OFFSET_MM   = cfg.GVXR_DET_OFFSET_MM

# Current hypothesis under test (see conversation): AP beam should be along
# Z (not Y as config.py currently has it), with "up" = Y so craniocaudal
# reads vertically in both AP and LAT. LAT is already confirmed correct —
# left untouched here. Once a combination looks right in cell 6, that's
# what goes into config.py's GVXR_AP_BEAM_DIR / GVXR_AP_UP.
AP_BEAM  = (0.0, 0.0, 1.0)
AP_UP    = (0.0, 1.0, 0.0)
LAT_BEAM = cfg.GVXR_LAT_BEAM_DIR
LAT_UP   = cfg.GVXR_LAT_UP

OUTDIR = _PROJECT_ROOT / "outputs" / "stl_preview"
OUTDIR.mkdir(parents=True, exist_ok=True)


# %% [3] Clip the platform plate off — re-run if CLIP_MM/UP_AXIS/STL_PATH change
clipped_path, clipped_bounds = clip_bottom(
    STL_PATH, CLIP_MM, UP_AXIS, OUTDIR / "clipped_preview.stl"
)
isocenter_local = np.mean(clipped_bounds, axis=0)
print(f"Clipped mesh  : {clipped_path}")
print(f"Clipped bounds: min={clipped_bounds[0].round(1)}  "
      f"max={clipped_bounds[1].round(1)}")
print(f"Isocentre     : {isocenter_local.round(1)} (auto, mesh-local)")


# %% [4] Build the gVXR scene — SLOW (~1-3 s), run once per session
preview_cfg = build_preview_cfg(
    clipped_path, COMPOUND, DENSITY,
    PIXELS, PIXEL_SIZE_MM, ENERGY_MEV, PHOTON_COUNT,
    SOD_MM, DET_OFFSET_MM, isocenter_local,
    ap_beam=AP_BEAM, ap_up=AP_UP, lat_beam=LAT_BEAM, lat_up=LAT_UP,
)

sim = XRaySimulator(preview_cfg)
t0 = time.perf_counter()
ok = sim.initialise()
print(f"initialise() -> {ok}  ({time.perf_counter() - t0:.1f} s)")


# %% [5] (Re)apply current beam/up/pixel settings — fast, re-run after editing cell 2
# Rebuilds the AP/LAT view geometry from whatever is currently set above,
# WITHOUT reloading the mesh or redoing the boolean clip. This is the part
# gVXR is designed to let you redo cheaply.
sim._ap = VirtualCArm.from_beam(
    "AP", isocenter_local, np.array(AP_BEAM, float), np.array(AP_UP, float),
    (PIXELS, PIXELS), PIXEL_SIZE_MM, SOD_MM, DET_OFFSET_MM,
)
sim._lat = VirtualCArm.from_beam(
    "LAT", isocenter_local, np.array(LAT_BEAM, float), np.array(LAT_UP, float),
    (PIXELS, PIXELS), PIXEL_SIZE_MM, SOD_MM, DET_OFFSET_MM,
)
sim.invalidate_background()
print(f"AP  beam/up : {AP_BEAM} / {AP_UP}")
print(f"LAT beam/up : {LAT_BEAM} / {LAT_UP}")
print(f"Detector    : {PIXELS}x{PIXELS} px @ {PIXEL_SIZE_MM} mm/px "
      f"({PIXELS * PIXEL_SIZE_MM:.0f} mm FOV)")


# %% [6] Render + display — fast, re-run this (and cell 5) as many times as you like
t0 = time.perf_counter()
ap_img, lat_img = sim.render_background()
print(f"Rendered in {time.perf_counter() - t0:.1f} s")

if ap_img is None or lat_img is None:
    print("[FAILED] render_background() returned None — check the log above.")
else:
    cv2.imwrite(str(OUTDIR / "ap_preview.png"), ap_img)
    cv2.imwrite(str(OUTDIR / "lat_preview.png"), lat_img)

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    axes[0].imshow(ap_img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("AP")
    axes[0].axis("off")
    axes[1].imshow(lat_img, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Lateral")
    axes[1].axis("off")
    fig.tight_layout()
    plt.show()

    print(f"Saved: {OUTDIR / 'ap_preview.png'}")
    print(f"Saved: {OUTDIR / 'lat_preview.png'}")


# %% [7] Shutdown — run once when you're completely done for the session
sim.shutdown()
