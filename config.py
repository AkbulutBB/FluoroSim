"""
config.py — FluoroSim complete configuration.

All physical measurements in millimetres.

SETUP CHECKLIST
===============
Before first run, fill in the sections marked  ← FROM CAD  by reading the
values from your Fusion 360 assembly.  Everything else works as-is.

1. Export STLs from Fusion (File → 3D Print → STL, units = mm):
       models/spine.stl
       models/platform.stl   (optional)

2. Fill in BEARING_POSITIONS from Fusion assembly coordinates.
3. Fill in SPINE_TO_WORLD (spine local->platform rigid transform, rotation
   included -- do NOT assume translation-only; see derivation notes below).
4. Fill in BOARD_TO_WORLD (ChArUco board origin + orientation in platform frame).
5. Run tools/test_gvxr.py — confirm "ALL TESTS PASSED".
6. Run main.py.
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# ArUco / ChArUco dictionaries
# ─────────────────────────────────────────────────────────────────────────────

ARUCO_PROBE_DICT_ID   = cv2.aruco.DICT_4X4_50
ARUCO_PROBE_DICT      = cv2.aruco.getPredefinedDictionary(ARUCO_PROBE_DICT_ID)

CHARUCO_BOARD_DICT_ID = cv2.aruco.DICT_5X5_50
CHARUCO_BOARD_DICT    = cv2.aruco.getPredefinedDictionary(CHARUCO_BOARD_DICT_ID)

# ─────────────────────────────────────────────────────────────────────────────
# ChArUco platform board
# ─────────────────────────────────────────────────────────────────────────────

CHARUCO_COLS        = 7      # number of squares horizontally
CHARUCO_ROWS        = 4      # number of squares vertically
CHARUCO_SQUARE_MM   = 18.0   # physical square side length (mm)
CHARUCO_MARKER_MM   = 13.5   # printed ArUco marker size inside each square
                              # (= 0.75 × square; adjust to match your print)

# ─────────────────────────────────────────────────────────────────────────────
# Two-cube probe geometry
# ─────────────────────────────────────────────────────────────────────────────
#
# Probe body frame:
#   Origin = bottom cube centre (the cube adjacent to the K-wire)
#   +Z axis = pointing toward the top/handle cube
#   K-wire exits from the −Z face of the bottom cube, 100 mm long
#   Top cube centre is at +45 mm along Z (40 mm cubes, 5 mm spacer)
#
#   Marker IDs:
#     Bottom cube: four side faces, IDs 0–3  (+X, −X, +Y, −Y)
#     Top    cube: four side faces, IDs 4–7  (+X, −X, +Y, −Y)
#

CUBE_SIDE_MM   = 40.0    # cube side length (mm)
MARKER_SIZE_MM = 32.0    # marker printed size per face (mm)
CUBE_GAP_MM    = 45.0    # bottom→top cube centre distance (40 mm + 5 mm spacer)
ROD_LENGTH_MM  = 100.0   # K-wire length from bottom cube face to tip

_h = MARKER_SIZE_MM / 2.0   # half marker side = 16 mm
_s = CUBE_SIDE_MM   / 2.0   # half cube side   = 20 mm
_g = CUBE_GAP_MM             # 45 mm

# ── Per-cube yaw about the probe long axis (+Z) ──────────────────────────────
# One cube is physically mounted rotated 45 deg relative to the other, so that
# any camera sees TWO faces per cube (corner-on) instead of one face-on. This
# gives more independent angular constraints in the joint PnP solve.
#
# Only the RELATIVE yaw between the two cubes affects accuracy. A global yaw of
# the whole probe is harmless here because the K-wire is coaxial with +Z
# (ROD_TIP_IN_PROBE lies on the Z axis), so rotating the entire probe about Z
# changes neither the tip position nor the trajectory direction.
#
# SIGN IS NOT YET FIELD-VERIFIED. +45 and -45 differ by 90 deg, which is a
# relabeling of which marker ID sits on which face — so a wrong sign is a real
# error in the model of the cubes' RELATIVE orientation, and it will corrupt any
# solve that uses markers from both cubes at once. It is easy to detect and fix:
# see "Verifying the yaw sign" below.
PROBE_CUBE_BOTTOM_YAW_DEG = 45.0   # cube with the K-wire — mounted corner-on
PROBE_CUBE_TOP_YAW_DEG    = 0.0    # handle-side cube — faces axis-aligned

# Verifying the yaw sign
# ----------------------
# Watch the probe reprojection RMS reported by the tracker while holding the
# probe so BOTH cubes are visible to one camera:
#   * low, stable RMS (a few px)      -> relative yaw is modeled correctly
#   * large RMS, or RMS that is fine  -> relative yaw is off by 90 deg;
#     on single-cube views but jumps     flip the sign below to -45.0
#     when the second cube appears
# Single-cube RMS looks fine under either sign, so judge it only on frames
# where both cubes are detected.


# ── ID -> face assignment around each cube ───────────────────────────────────
# Which marker ID sits on which physical face, in the order (+X, -X, +Y, -Y)
# BEFORE the cube's yaw is applied.
#
# NOTE the default is NOT "sequential around the cube": (0, 1, 2, 3) here means
# 0 and 1 are on OPPOSITE faces, and 2 and 3 are on opposite faces. If the
# stickers were instead applied going around the cube (0, 1, 2, 3 adjacent),
# the correct value is (0, 2, 1, 3).
#
# This matters enormously and is invisible until a camera sees two faces of the
# SAME cube at once: opposite faces can never be seen together, so a wrong
# ordering makes such frames geometrically impossible to fit and the solver
# produces a wildly wrong pose. It stays hidden in face-on views and only
# appears once a cube is mounted corner-on.
#
# Measure with tools/diagnose_probe_geometry.py — do not assume.
PROBE_BOTTOM_FACE_ORDER = (0, 1, 2, 3)   # ids on (+X, -X, +Y, -Y)
PROBE_TOP_FACE_ORDER    = (4, 5, 6, 7)   # ids on (+X, -X, +Y, -Y)

# ── Per-face sticker rotation ────────────────────────────────────────────────
# How many 90-degree steps each marker sticker is rotated on its cube face,
# relative to the corner order assumed below (TL, TR, BR, BL viewed from
# outside). 0 = as assumed, 1 = 90 deg, 2 = 180 deg, 3 = 270 deg.
#
# This matters because OpenCV's detectMarkers() returns corners in a fixed
# order relative to the MARKER's own orientation. If a sticker was applied
# rotated, the detected "top-left" corner corresponds to a different physical
# 3-D corner, and that face contributes a large systematic reprojection error
# that no global yaw or scale correction can absorb.
#
# Measure these with tools/diagnose_probe_geometry.py rather than guessing.
PROBE_FACE_STICKER_ROT: dict[int, int] = {
    0: 0, 1: 0, 2: 0, 3: 0,     # bottom cube
    4: 0, 5: 0, 6: 0, 7: 0,     # top cube
}


def _yaw_about_z(deg: float) -> np.ndarray:
    """Rotation matrix about the probe long axis (+Z)."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _cube_face_obj_pts(
    z_centre: float, yaw_deg: float, ids: tuple[int, int, int, int]
) -> dict[int, np.ndarray]:
    """
    Build the 4 side-face marker corner sets for one cube, in probe body frame.

    `ids` are assigned in (+X, -X, +Y, -Y) face order as seen BEFORE the yaw is
    applied — i.e. they follow the physical faces as the cube rotates, which is
    what actually happens when you rotate the printed cube on its mount.

    Corner order per marker: top-left, top-right, bottom-right, bottom-left
    (OpenCV ArUco convention, viewed from outside the cube face).
    """
    base = {
        ids[0]: [[ _s, -_h,  _h], [ _s,  _h,  _h], [ _s,  _h, -_h], [ _s, -_h, -_h]],  # +X
        ids[1]: [[-_s,  _h,  _h], [-_s, -_h,  _h], [-_s, -_h, -_h], [-_s,  _h, -_h]],  # −X
        ids[2]: [[-_h,  _s,  _h], [ _h,  _s,  _h], [ _h,  _s, -_h], [-_h,  _s, -_h]],  # +Y
        ids[3]: [[ _h, -_s,  _h], [-_h, -_s,  _h], [-_h, -_s, -_h], [ _h, -_s, -_h]],  # −Y
    }
    R = _yaw_about_z(yaw_deg)
    out: dict[int, np.ndarray] = {}
    for mid, pts in base.items():
        p = (R @ np.asarray(pts, dtype=np.float64).T).T   # yaw about probe axis
        p[:, 2] += z_centre                               # lift to cube centre height
        # Sticker rotation: rolling the 3-D corner order models the marker
        # having been applied to the face at a 90-degree multiple.
        rot = int(PROBE_FACE_STICKER_ROT.get(mid, 0)) % 4
        if rot:
            p = np.roll(p, rot, axis=0)
        out[mid] = p.astype(np.float32)
    return out


# 3-D corners of each marker in PROBE BODY FRAME.
#   Bottom cube: z range −20..+20 (centre z=0),  IDs 0–3
#   Top    cube: z range +25..+65 (centre z=+45), IDs 4–7
PROBE_FACE_OBJ_PTS: dict[int, np.ndarray] = {
    **_cube_face_obj_pts(0.0, PROBE_CUBE_BOTTOM_YAW_DEG, PROBE_BOTTOM_FACE_ORDER),
    **_cube_face_obj_pts(_g,  PROBE_CUBE_TOP_YAW_DEG,    PROBE_TOP_FACE_ORDER),
}

# K-wire endpoints in probe body frame
ROD_BASE_IN_PROBE = np.array([0.0, 0.0, -_s              ], np.float32)  # bottom face of bottom cube
ROD_TIP_IN_PROBE  = np.array([0.0, 0.0, -_s - ROD_LENGTH_MM], np.float32)  # 100 mm below

# ─────────────────────────────────────────────────────────────────────────────
# Camera setup
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_IDS      = [0, 1]    # OS device indices for the two webcams
CAMERA_WIDTH    = 1280      # capture resolution
CAMERA_HEIGHT   = 720
CAMERA_FPS      = 30
CAMERA_LABELS   = ["Camera A", "Camera B"]

# Paths to saved intrinsic calibration files
INTRINSICS_PATH_0 = "data/intrinsics/cam0.npz"
INTRINSICS_PATH_1 = "data/intrinsics/cam1.npz"

# ─────────────────────────────────────────────────────────────────────────────
# Intrinsic calibration (checkerboard)
# ─────────────────────────────────────────────────────────────────────────────

CHECKER_COLS    = 9     # inner corner count (squares − 1)
CHECKER_ROWS    = 6
CHECKER_SQ_MM   = 25.0  # physical square size (mm)
CALIB_FRAMES    = 25    # frames to collect before computing intrinsics

# ─────────────────────────────────────────────────────────────────────────────
# gVXR — virtual C-arm + X-ray simulation
# ─────────────────────────────────────────────────────────────────────────────

# Context type:
#   "OPENGL"  → Windows desktop / Anaconda  (always available with a display)
#   "EGL"     → Linux headless
GVXR_CONTEXT = "OPENGL"

# STL file paths (mm units assumed in the STL)
SPINE_STL_PATH    = "models/spine.stl"
PLATFORM_STL_PATH = ""           # set to "models/platform.stl" if available

# ── Baseplate clipping ───────────────────────────────────────────────────────
# spine.stl includes the mounting baseplate, which renders as a bright bar/slab
# and is not anatomy the trainee should see. Clip it off exactly the way
# tools/stl_xray_preview.py does (same defaults), in SPINE-LOCAL coordinates:
# remove the bottom SPINE_CLIP_MM along SPINE_CLIP_AXIS.
#
# Clipping does NOT recentre the mesh, so BOARD_TO_SPINE stays valid.
# Set SPINE_CLIP_MM = 0.0 to disable (baseplate will then appear in renders).
SPINE_CLIP_MM   = 8.0
SPINE_CLIP_AXIS = "z"            # local axis the baseplate sits at the bottom of

# Render the steel bearings? They live on the PLATFORM, not the spine, and were
# for the earlier registration approach — a trainee should not see them. Kept
# available because tools/test_registration.py turns them on to verify that the
# board→spine chain lands them where CAD says they should be.
RENDER_BEARINGS = False

# Spine material — cortical bone equivalent gives a realistic X-ray appearance.
# For a PETG-printed model without bone-equivalent coating, use:
#     SPINE_MATERIAL_COMPOUND = "C5H8O2"  ;  SPINE_MATERIAL_DENSITY = 1.27
SPINE_MATERIAL_COMPOUND = "Ca10(PO4)6(OH)2"   # hydroxyapatite
SPINE_MATERIAL_DENSITY  = 1.92                 # g/cm³  (cortical shell)

# Trabecular core (only used when SPINE_SHELL_STL_PATH/SPINE_CORE_STL_PATH
# are both set — see tools/stl_xray_preview.py's split_shell_core()).
# 0.35 g/cm3 = top of the 0.09-0.35 g/cm3 range reported across studies of
# vertebral trabecular apparent (wet) density; raised here from an initial
# 0.2 (literature-range midpoint) after visual comparison showed more
# contrast was preferred — this is a real, cited upper bound, not an
# arbitrary contrast tweak.
SPINE_TRABECULAR_COMPOUND = "Ca10(PO4)6(OH)2"
SPINE_TRABECULAR_DENSITY  = 0.35               # g/cm³  (was 0.2)

# Virtual C-arm geometry (mm, gVXR world = CAD platform space).
#
# Views are FIXED in the board frame (BOARD_TO_WORLD is now a real 90 deg
# axis-relabeling rotation + offset, solved below -- see that section), and
# defined by a BEAM DIRECTION (direction X-rays travel) plus a shared isocentre.
#
# Board frame (OpenCV ChArUco convention):
#   +X = long 126 mm edge  → left/right relative to the board
#   +Y = short 72 mm edge  → up/down relative to the board
#   +Z = toward cameras    → caudal (spine extends toward cameras from the
#                             cranial board)
#
# Motion → image response (matches the intended fluoroscopy behaviour):
#   AP  looks down board Y   → up/down doesn't move the dot; left/right moves it
#                              horizontally; caudal moves it vertically.
#   LAT looks down board X   → left/right doesn't move it; caudal moves it
#                              horizontally; up/down moves it vertically; the
#                              probe (inserted along Y) shows as a vertical line.
#
# If a specific motion comes out MIRRORED on your rig (e.g. caudal moves the
# AP dot up instead of down), flip the sign of the corresponding beam/up vector
# below — the axis PAIRING is correct, only per-axis signs depend on mounting.

# None = AUTO: computed at runtime from the CLIPPED spine mesh's own
# bounding-box centre, in spine-local coordinates -- exactly what
# tools/stl_xray_preview.py does to produce its known-good images. Leave this
# as None; it self-corrects if the STL or clip depth changes. Set an explicit
# (x, y, z) tuple only to deliberately override the framing.
GVXR_ISOCENTER   = None
GVXR_SOD_MM      = 1000.0              # source-to-isocentre distance
GVXR_DET_OFFSET_MM = 50.0             # detector offset past isocentre

# AP: beam along board Z, NOT the original +Y assumption above the old
# comment described. Evidence: with beam along Y, the AP render showed a
# single compressed axial "ring" (pedicles/spinous process/vertebral body
# all collapsed into one silhouette, no sense of separate levels) — that's
# what beam-along-the-craniocaudal-axis looks like, not true AP. Lateral
# (beam along X) is independently confirmed correct, so by elimination Z
# is anterior-posterior. Up-vector is Y (craniocaudal) so both AP and LAT
# share the same vertical convention.
#
# RENDER FRAME = SPINE-LOCAL. These vectors are expressed in the spine STL's
# OWN local frame, and are the exact values validated by
# tools/stl_xray_preview.py (which renders the unrotated STL and produces
# anatomically correct AP + lateral images). The renderer keeps the spine at
# identity and brings the PROBE to the spine via BOARD_TO_SPINE, rather than
# pushing the spine into world space and having to re-derive these — so these
# vectors do NOT need rotating when SPINE_TO_WORLD changes.
GVXR_AP_BEAM_DIR = (0.0,  0.0, 1.0)
GVXR_AP_UP       = (0.0,  1.0, 0.0)

# Lateral: beam along local -X (probe shows full length).
GVXR_LAT_BEAM_DIR = (-1.0, 0.0, 0.0)
GVXR_LAT_UP       = ( 0.0, 1.0, 0.0)   # image "up" = +Y

# Detector parameters
GVXR_DETECTOR_PIXELS = (512, 512)   # width × height
# Pixel pitch chosen for the clipped model's actual footprint (~100 x 82 x
# 76 mm from the last render). At 0.5 mm/px the 256 mm FOV left the object
# under half the frame — most detector resolution was spent on empty
# background. 0.25 mm/px -> 128 mm FOV: ~14 mm margin around the AP view's
# widest axis (100 mm), ~23-26 mm margin on the others. If a future STL is
# larger and gets clipped at the frame edge, raise this back up rather than
# assuming the geometry is wrong.
GVXR_PIXEL_SIZE_MM   = 0.25          # isotropic pixel pitch (128 mm FOV total)

# Beam parameters
GVXR_ENERGY_MEV   = 0.08    # 80 keV — standard fluoroscopy energy
GVXR_PHOTON_COUNT = 5000    # was 1000; higher = less Monte Carlo graininess, slower

# ─────────────────────────────────────────────────────────────────────────────
# CAD-derived registration values   ← FROM CAD (fill these in from Fusion 360)
# ─────────────────────────────────────────────────────────────────────────────

# Position of each steel ball bearing centre in gVXR world (= CAD platform) space.
# X, Y, Z in mm relative to platform origin.
#
# Solved via Kabsch/SVD fit of 8 measured bearing correspondences + the
# ChArUco board's own origin corner as a 9th point (see BOARD_TO_WORLD below
# for the same fit) -- RMS residual 0.000 mm, i.e. these are exact CAD design
# values, not yet field-validated against a live camera reading.
# ASSUMPTION FLAGGED: point-order 1-7 assumed to match physical bearing
# labels B1-B7 in the order originally measured; re-check against physical
# labels before trusting the reprojection-error validation step.
BEARING_POSITIONS = [
    {"label": "B1",            "position_mm": (-64.00, -70.50,  78.50), "radius_mm": 1.5},
    {"label": "B2",            "position_mm": ( 35.00, -67.00,  78.50), "radius_mm": 1.5},
    {"label": "B3",            "position_mm": ( 45.00, -74.00,  78.50), "radius_mm": 1.5},
    {"label": "B4",            "position_mm": ( 55.00, -70.50,  78.50), "radius_mm": 1.5},
    {"label": "B5",            "position_mm": ( 35.25,  60.50,  58.00), "radius_mm": 1.5},
    {"label": "B6",            "position_mm": (-25.50,  60.50,  35.00), "radius_mm": 1.5},
    {"label": "B7",            "position_mm": (-64.25,  60.50,  12.50), "radius_mm": 1.5},
    {"label": "B8_probe_tip",  "position_mm": (-77.00,  -8.00,  38.50), "radius_mm": 1.5},
]

# Spine STL local frame -> gVXR world (CAD platform) frame. Rigid 4x4,
# rotation included -- do NOT assume translation-only (an earlier version of
# this comment did; that was wrong).
#
# Solved from two local<->world corner correspondences on the spine's bottom
# plate:
#   A = cranial-right corner (genuine hard-stop, "locks in" against the
#       platform): local (-23.64995, 0.78463, 33.90369) <-> world (-69, 30, -6)
#   C = right-caudal corner (X/Z from real Fusion sketch geometry; Y carried
#       over from A since there is no caudal hard-stop -- least-trusted axis):
#       local (-23.29325, 0.88218, -47.59369) <-> world (12.50, 30.00, -6.00)
# Point C predicted from A's fit to within (0.003, 0.357, 0.098) mm -- the
# rotation is confirmed, not assumed. It also happens to equal BOARD_TO_WORLD's
# rotation exactly, consistent with both components sharing one Fusion
# local-axis convention.
#
# NOT yet used to predict the left-cranial corner's true seated position
# (predicts ~(-69.3, -32.6, -5.4), i.e. more slack there than the plate's
# nominal 5mm-each-side clearance) -- harmless for rendering, just flagging
# the plate isn't symmetric in its slot.
#
# Re-derive with tools/solve_board_to_world.py-style Kabsch fit (ideally with
# a 3rd non-collinear point) if the spine STL or platform mount ever changes.
SPINE_TO_WORLD = np.array([
    [ 0.0,  0.0, -1.0, -35.0963],
    [-1.0,  0.0,  0.0,   6.3500],
    [ 0.0,  1.0,  0.0,  -6.7846],
    [ 0.0,  0.0,  0.0,   1.0000],
], dtype=np.float64)

# ChArUco board → gVXR world transform  (4×4, row-major rigid matrix).
# This tells the system where the board origin sits in platform/world space.
#
# How to fill this in from Fusion 360:
#   1. Open platform assembly.
#   2. In Inspect → Measure, click the board component origin.
#      Record its (X, Y, Z) relative to platform origin → set last column below.
#   3. If the board is flat on the platform (XZ plane, normal toward camera = +Y):
#      rotation is identity (board X = world X, board Y = world Y, board Z = world Z).
#   4. If the board is rotated, use Component Properties to read the local
#      axis directions and build the 3×3 rotation part.
#
# Solved via Kabsch/SVD fit over 8 fiducial correspondences + the board's own
# origin corner (CAD -86, 53, 71). Fit converged to an EXACT signed-permutation
# rotation (RMS residual 0.000 mm across all 9 points) -- a pure 90 deg axis
# relabeling, no tilt component:
#     world_X =  board_Z
#     world_Y = -board_X
#     world_Z = -board_Y
# Re-derive if the platform/board mount geometry ever changes.
BOARD_TO_WORLD = np.array([
    [ 0.0,  0.0,  1.0, -86.0],
    [-1.0,  0.0,  0.0,  53.0],
    [ 0.0, -1.0,  0.0,  71.0],
    [ 0.0,  0.0,  0.0,   1.0],
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# DERIVED transforms — the render frame is SPINE-LOCAL
# ─────────────────────────────────────────────────────────────────────────────
# BOARD_TO_WORLD and SPINE_TO_WORLD above are the ground-truth CAD facts, both
# solved from measured correspondences. Everything below is COMPOSED from them
# — never hand-edit these; fix the two source matrices and these follow.
#
# Why spine-local rather than world as the render frame: the spine STL's own
# frame is where tools/stl_xray_preview.py produces known-good anatomically
# correct AP/lateral images. Rendering there means the spine sits at identity
# and the beam vectors stay at their validated values. Only the tracked probe
# needs transforming, via the single chain:
#
#     camera --[ChArUco solvePnP]--> board --[BOARD_TO_SPINE]--> spine-local
#
WORLD_TO_SPINE = np.linalg.inv(SPINE_TO_WORLD)

# board -> spine-local. Works out to a clean 180 deg rotation about X plus an
# offset (X preserved, Y and Z flipped) — a consequence of BOARD_TO_WORLD and
# SPINE_TO_WORLD sharing the same 120 deg local-axis convention.
BOARD_TO_SPINE = WORLD_TO_SPINE @ BOARD_TO_WORLD

# Fail loudly at import if either source matrix stops being a proper rigid
# transform (e.g. a typo introduces scale or a reflection) — a silently
# non-rigid transform would distort every probe position by a few mm without
# any obvious symptom in the rendered image.
assert np.allclose(BOARD_TO_SPINE[:3, :3] @ BOARD_TO_SPINE[:3, :3].T, np.eye(3), atol=1e-9), \
    "BOARD_TO_SPINE rotation is not orthonormal — check BOARD_TO_WORLD / SPINE_TO_WORLD."
assert abs(np.linalg.det(BOARD_TO_SPINE[:3, :3]) - 1.0) < 1e-9, \
    "BOARD_TO_SPINE has det != +1 (reflection or scale) — check the source matrices."


# ─────────────────────────────────────────────────────────────────────────────
# Tracking parameters
# ─────────────────────────────────────────────────────────────────────────────

MIN_CHARUCO_CORNERS  = 6     # minimum corners for a valid board pose estimate
MIN_PROBE_MARKERS    = 1     # minimum ArUco faces to attempt probe pose estimate
SMOOTHING_ALPHA      = 0.4   # exponential smoothing for tip position (0=frozen, 1=raw)
UPDATE_INTERVAL_MS   = 200   # UI refresh interval (5 Hz)

# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

APP_TITLE      = "FluoroSim — gVXR Simulated Fluoroscopy"
APP_VERSION    = "2.0.0"
PREVIEW_W      = 480
PREVIEW_H      = 360
XRAY_PANEL_W   = 512
XRAY_PANEL_H   = 512

# Overlay colours (BGR)
OVERLAY_TIP_COLOR   = (  0, 255,  50)
OVERLAY_SHAFT_COLOR = (  0, 210, 255)
OVERLAY_THICKNESS   = 2
OVERLAY_TIP_RADIUS  = 8
OVERLAY_EXTEND_MM   = 80.0   # extra shaft drawn past the tip
