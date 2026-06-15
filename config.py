"""
config.py  —  FluoroSim physical constants and tolerances
============================================================

Every hard number that describes the *physical* hardware lives here, so the
rest of the code never hard-codes a millimetre value.  These match the printed
ChArUco board and the ArUco probe on the platform — do not change them unless
you reprint the markers or re-machine the probe.

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

# ── ArUco probe: TWO stacked cubes ──────────────────────────────────────────
# The probe is now a rigid body of two ArUco cubes stacked along one axis, with
# the K-wire exiting the BOTTOM cube's centre (collinear with the axis — no more
# lateral offset).  Only the 4 SIDE faces of each cube carry markers (the top
# and bottom faces are taken up by the handle and the inter-cube spacer), so
# there are 4 markers per cube, 8 in total.
#
# Why two cubes:  the tracker solves ONE rigid pose from every marker corner it
# can see in both cameras.  Spreading 8 markers over a ~53 mm baseline gives the
# pose a much longer angular lever than a single 40 mm cube, so the extrapolated
# tip is more stable.  "Drawing a line between the two cubes" is exactly the
# probe-local +Y axis below — it falls straight out of the rigid solve.
PROBE_DICT      = cv2.aruco.DICT_4X4_50
CUBE_SIDE_MM    = 40.0          # outer edge of EACH white PETG cube     <-- CONFIRM
PROBE_MARKER_MM = 32.0          # printed marker edge on each side face   <-- CONFIRM

# Clear gap between the two cubes' facing surfaces (the 5 mm connecting spacer).
CUBE_GAP_MM     = 5.0
CUBE_PITCH_MM   = CUBE_SIDE_MM + CUBE_GAP_MM        # centre-to-centre = 45 mm (CAD-verified)

# Probe-local frame:
#   origin : BOTTOM cube centre
#   +Y     : toward the TOP cube (handle side)
#   -Y     : toward the K-wire tip
#   +/-X, +/-Z : the four side-face normals
PROBE_BOTTOM_CENTER = np.array([0.0, 0.0,            0.0], dtype=np.float64)
PROBE_TOP_CENTER    = np.array([0.0, CUBE_PITCH_MM,  0.0], dtype=np.float64)

# Marker-ID -> (which cube, which side face).  ***SINGLE SOURCE OF TRUTH***.
# This table must match how you actually print and glue the stickers; the same
# table will drive the marker-sheet generator so the two can't drift apart.
# Faces: "PZ" (+Z front), "NZ" (-Z back), "PX" (+X right), "NX" (-X left).
PROBE_FACE_IDS = {
    # Bottom cube  ("Aruco 2" in your render — the one the K-wire exits)
    0: ("bottom", "PZ"),
    1: ("bottom", "NZ"),
    2: ("bottom", "PX"),
    3: ("bottom", "NX"),
    # Top cube  ("Aruco 1" — handle side)
    4: ("top", "PZ"),
    5: ("top", "NZ"),
    6: ("top", "PX"),
    7: ("top", "NX"),
}
PROBE_IDS = tuple(sorted(PROBE_FACE_IDS))

# ── K-wire geometry, in the probe-local frame above ──────────────────────────
# The wire is collinear with the probe axis and exits the bottom cube centre.
# CAD-verified: the tip stands 100 mm from the BOTTOM-CUBE CENTRE along -Y.
ROD_TIP_OFFSET_MM = 100.0
ROD_LENGTH_MM     = ROD_TIP_OFFSET_MM
ROD_BASE_IN_CUBE  = PROBE_BOTTOM_CENTER.copy()                                   # (0, 0, 0)
ROD_TIP_IN_CUBE   = PROBE_BOTTOM_CENTER + np.array([0.0, -ROD_TIP_OFFSET_MM, 0.0])  # (0, -100, 0)

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

# ── Aligning CAD fiducials to the tracker frame (probe best-fit) ──────────────
DIGITIZE_REJECT_PX   = 25.0      # ignore clearly-bad frames above this
DIGITIZE_STABLE_MM   = 5.0       # spread below this = a clean, steady reading
DIGITIZE_MIN_SAMPLES = 12        # frames to average before a capture is allowed

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
ROLE_LABEL   = {"ap": "Camera 1", "lat": "Camera 2 (oblique \u2248 45\u00b0)"}
XRAY_LABEL   = {"ap": "AP", "lat": "Lateral"}
