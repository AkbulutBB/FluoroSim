"""
FluoroSim v2 — Global configuration.

All physical measurements are in millimetres.

Coordinate systems
──────────────────
Probe cube   : origin at cube centre.
               +Z = rod-exit face (ID 0, FRONT).  +Y = top face (ID 4).
               The K-wire tip is 100 mm beyond the +Z face surface.

Model/Board  : origin at the bottom-left corner of the CharucoBoard
               (as seen when facing the cranial wall of the platform).
               +X = rightward across board width.
               +Y = upward along board height.
               +Z = out of the board face (toward the cameras).
               All X-ray fiducial coordinates must be measured from this origin.
"""

import cv2
import numpy as np

# ── ArUco dictionaries ─────────────────────────────────────────────────────────
# Two separate dictionaries eliminate any ID collision between probe and board.
PROBE_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

# ── Probe cube geometry ────────────────────────────────────────────────────────
CUBE_SIDE_MM   = 40.0   # printed cube side length
MARKER_SIZE_MM = 32.0   # printed ArUco marker size on each face (0.8 × side)
ROD_LENGTH_MM  = 100.0  # K-wire length from +Z face surface to tip

_h = MARKER_SIZE_MM / 2.0   # 16 mm — half marker
_s = CUBE_SIDE_MM   / 2.0   # 20 mm — half cube

# 3-D corners of each marker face in cube-local space.
# OpenCV ArUco convention: top-left → top-right → bottom-right → bottom-left.
# "Top" of each marker points toward +Y (face 4) EXCEPT faces 4 and 5
# whose top edges point toward face 0 (+Z).
CUBE_FACE_OBJ_PTS: dict[int, np.ndarray] = {
    # ID 0  +Z  FRONT  — rod exits here (most critical)
    0: np.array([[-_h,  _h,  _s], [ _h,  _h,  _s],
                 [ _h, -_h,  _s], [-_h, -_h,  _s]], np.float32),
    # ID 1  -Z  BACK
    1: np.array([[ _h,  _h, -_s], [-_h,  _h, -_s],
                 [-_h, -_h, -_s], [ _h, -_h, -_s]], np.float32),
    # ID 2  +X  RIGHT
    2: np.array([[ _s,  _h,  _h], [ _s,  _h, -_h],
                 [ _s, -_h, -_h], [ _s, -_h,  _h]], np.float32),
    # ID 3  -X  LEFT
    3: np.array([[-_s,  _h, -_h], [-_s,  _h,  _h],
                 [-_s, -_h,  _h], [-_s, -_h, -_h]], np.float32),
    # ID 4  +Y  TOP  (cranial camera sees this most during training)
    4: np.array([[-_h,  _s, -_h], [ _h,  _s, -_h],
                 [ _h,  _s,  _h], [-_h,  _s,  _h]], np.float32),
    # ID 5  -Y  BOTTOM
    5: np.array([[-_h, -_s,  _h], [ _h, -_s,  _h],
                 [ _h, -_s, -_h], [-_h, -_s, -_h]], np.float32),
}

# Rod geometry in cube-local space (rod exits +Z face)
ROD_BASE_IN_CUBE = np.array([0.0, 0.0,  _s],                    np.float32)
ROD_TIP_IN_CUBE  = np.array([0.0, 0.0,  _s + ROD_LENGTH_MM],    np.float32)

# ── Platform CharucoBoard ──────────────────────────────────────────────────────
# Mounted on the cranial face of the platform: 80 mm tall × 140 mm wide.
# A 4-column × 7-row grid of 18 mm squares → board size 72 × 126 mm.
# Fits inside 80 × 140 mm with clean 4 mm margins on all sides.
CHARUCO_COLS      = 4       # squares across (X direction)
CHARUCO_ROWS      = 7       # squares tall   (Y direction)
CHARUCO_SQUARE_MM = 18.0    # physical size of each chessboard square
CHARUCO_MARKER_MM = 13.0    # ArUco marker inside each square (~0.72 × square)

# Minimum Charuco inner corners required to accept a pose estimate.
# 4 is the mathematical minimum for solvePnP; 6 adds robustness.
MIN_CHARUCO_CORNERS = 6

# ── Calibration slot ──────────────────────────────────────────────────────────
# Single slot on the cranial wall, top-entry. Probe inserts 40 mm into the slot.
# Used only as a defined starting position for the trainee — not a computational step.
SLOT_DEPTH_MM = 40.0

# ── Intrinsic camera calibration ──────────────────────────────────────────────
CHECKERBOARD_COLS      = 9      # inner corner count (columns)
CHECKERBOARD_ROWS      = 6      # inner corner count (rows)
CHECKERBOARD_SQ_MM     = 25.0   # physical square size
INTRINSIC_CALIB_FRAMES = 20     # checkerboard frames to collect before computing

# ── Navigation / display ──────────────────────────────────────────────────────
PREVIEW_W          = 640
PREVIEW_H          = 480
XRAY_DISPLAY_W     = 480
XRAY_DISPLAY_H     = 600
NAV_UPDATE_MS      = 100    # 10 fps overlay refresh (real-time mode)
NAV_REALTIME       = True   # default to real-time on launch

OVERLAY_COLOR_TIP    = (  0, 255,   0)   # green  — K-wire tip dot
OVERLAY_COLOR_SHAFT  = (  0, 200, 255)   # cyan   — shaft trajectory line
OVERLAY_THICKNESS    = 3
OVERLAY_TIP_RADIUS   = 10
OVERLAY_SHAFT_EXTEND = 60    # mm to extend shaft line beyond tip for readability

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
MODELS_DIR  = f"{DATA_DIR}/models"
CAMERAS_DIR = f"{DATA_DIR}/cameras"

# ── Application ────────────────────────────────────────────────────────────────
APP_TITLE   = "FluoroSim — Simulated Fluoroscopy Training System"
APP_VERSION = "2.0.0"
MAX_CAMERAS = 8
