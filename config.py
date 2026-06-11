"""
config.py  —  FluoroSim physical constants and tolerances
============================================================

Every hard number that describes the *physical* hardware lives here, so the
rest of the code never hard-codes a millimetre value.  These match the printed
ChArUco board and the ArUco probe cube you are already using — do not change
them unless you reprint the markers.

Coordinate convention (MODEL space, a.k.a. board space)
-------------------------------------------------------
The platform ChArUco board defines the model coordinate frame:

    origin  : bottom-left inner corner of the board, as it sits on the platform
    +X      : along the long (126 mm) edge of the board   -> "across"
    +Y      : along the short (72 mm) edge of the board    -> "up the platform"
    +Z      : out of the board face, toward the cameras
    units   : millimetres, everywhere

All fiducial (steel-bearing) coordinates and the calibration-hole coordinate
are expressed in this same frame.  That shared origin is the single most
important consistency requirement in the whole system.
"""

import cv2
import numpy as np

APP_NAME = "FluoroSim"

# ── Platform ChArUco board ──────────────────────────────────────────────────
# 7 columns x 4 rows, landscape.  (cols, rows) MUST match the printed board or
# the detector finds zero corners and the board never registers.
CHARUCO_DICT      = cv2.aruco.DICT_5X5_50
CHARUCO_COLS      = 7        # squares across  -> board width  = 7 * 18 = 126 mm  (+X)
CHARUCO_ROWS      = 4        # squares tall    -> board height = 4 * 18 =  72 mm  (+Y)
CHARUCO_SQUARE_MM = 18.0     # chessboard square edge
CHARUCO_MARKER_MM = 14.4     # ArUco marker edge inside each square (0.80 x square)

# Minimum interpolated ChArUco corners before we trust a board pose.
MIN_CHARUCO_CORNERS = 6

# ── ArUco probe cube ────────────────────────────────────────────────────────
PROBE_DICT      = cv2.aruco.DICT_4X4_50
PROBE_IDS       = (0, 1, 2, 3, 4, 5)   # one marker per cube face
CUBE_SIDE_MM    = 40.0                 # outer edge of the white PETG cube
PROBE_MARKER_MM = 32.0                 # printed marker edge on each face

# Rod / K-wire geometry, expressed in CUBE-LOCAL coordinates.
# Cube centre is the origin; +Z exits the FRONT face (marker ID 0).
ROD_EXIT_FACE_ID = 0
ROD_BASE_IN_CUBE = np.array([0.0, 0.0,  CUBE_SIDE_MM / 2.0],          dtype=np.float64)  # (0,0, 20)
ROD_LENGTH_MM    = 100.0
ROD_TIP_IN_CUBE  = np.array([0.0, 0.0,  CUBE_SIDE_MM / 2.0 + ROD_LENGTH_MM], dtype=np.float64)  # (0,0,120)

# ── Calibration / verification hole ─────────────────────────────────────────
# A drilled hole on the platform at a *known* model-space coordinate.  Insert
# the probe tip fully into it; the system should report the tip at this point.
# This is a property of YOUR platform — measure it once and set it here (it can
# also be edited and saved from the Cameras screen).
CALIB_HOLE_MODEL_MM = np.array([63.0, 36.0, 0.0], dtype=np.float64)  # placeholder: board centre, on the face

# ── Tolerances (your stated targets) ────────────────────────────────────────
TIP_ERROR_TOLERANCE_MM   = 5.0   # max acceptable tip error at the calibration hole
CAMERA_AGREEMENT_TOL_MM  = 5.0   # max acceptable disagreement between the two cameras
TRAJECTORY_ANGLE_TOL_DEG = 5.0
DLT_REPROJ_WARN_PX       = 2.0   # warn if X-ray fiducial reprojection exceeds this

# ── Simulation overlay appearance ────────────────────────────────────────────
OVERLAY_THICKNESS_PX = 5         # trajectory line thickness on the X-ray
OVERLAY_TIP_RADIUS_PX = 7        # tip marker radius

# ── Live smoothing / outlier rejection (reduces probe flicker) ────────────────
SMOOTH_ALPHA        = 0.6        # 0 = no smoothing, ->1 = heavier smoothing
SMOOTH_TRUST_PX     = 3.0        # a solve this accurate is always accepted
SMOOTH_MAX_JUMP_MM  = 40.0       # if a high-error solve jumps more than this, reject it

# ── Intrinsic calibration ───────────────────────────────────────────────────
MIN_CALIB_VIEWS = 8       # minimum board captures for a usable lens calibration
GOOD_CALIB_RMS  = 1.0     # px; calibration RMS below this is "good"

# ── Cameras and X-ray views (these are INDEPENDENT) ──────────────────────────
# The two tracking CAMERAS and the two X-RAY VIEWS are decoupled on purpose.
# The cameras exist only to fix the probe's position in 3-D model space; their
# physical angles can be anything that gives a good solve.  Each X-ray is then
# projected through ITS OWN matrix -- so the X-rays can be a TRUE AP and a TRUE
# lateral no matter where the webcams sit.
CAMERA_ROLES = ("ap", "lat")
ROLE_LABEL   = {"ap": "Camera 1 (head-on)", "lat": "Camera 2 (45 deg oblique)"}
XRAY_LABEL   = {"ap": "AP", "lat": "Lateral"}
