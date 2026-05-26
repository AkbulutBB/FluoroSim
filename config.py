"""
FluoroSim - Global configuration and probe geometry.

All physical measurements are in millimetres.

ArUco cube coordinate system (rod exits in +Z direction, pointing into tissue):
  +Y  = top face   — ID 0  (AP camera primary view)
  +X  = right face — ID 1  (LAT camera primary view)
  -X  = left face  — ID 2  (LAT camera alternate view)
  -Z  = back face  — ID 3  (occasionally visible)
  +Z  = rod-exit   — no marker (never visible; faces into tissue)
  -Y  = bottom     — no marker (rests on model surface)
"""

import cv2
import numpy as np

# ─── ArUco ────────────────────────────────────────────────────────────────────
ARUCO_DICT_ID   = cv2.aruco.DICT_4X4_50
ARUCO_DICT      = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)

# ─── Probe geometry ───────────────────────────────────────────────────────────
CUBE_SIDE_MM    = 40.0   # printed cube side length
MARKER_SIZE_MM  = 32.0   # printed marker size on each face  (= 0.8 × cube side)
ROD_LENGTH_MM   = 100.0  # rod extending from the +Z face

_h = MARKER_SIZE_MM / 2.0   # half marker
_s = CUBE_SIDE_MM   / 2.0   # half cube

# 3-D corners of each marker face in cube-local space.
# Order matches OpenCV ArUco convention: top-left, top-right, bottom-right, bottom-left.
# Only the 4 faces that are actually visible during use are included.
CUBE_FACE_OBJ_PTS: dict[int, np.ndarray] = {
    0: np.array([[-_h, _s, -_h], [ _h, _s, -_h], [ _h, _s,  _h], [-_h, _s,  _h]], np.float32),  # +Y (top)
    1: np.array([[ _s, _h,  _h], [ _s, _h, -_h], [ _s,-_h, -_h], [ _s,-_h,  _h]], np.float32),  # +X (right)
    2: np.array([[-_s, _h, -_h], [-_s, _h,  _h], [-_s,-_h,  _h], [-_s,-_h, -_h]], np.float32),  # -X (left)
    3: np.array([[ _h, _h, -_s], [-_h, _h, -_s], [-_h,-_h, -_s], [ _h,-_h, -_s]], np.float32),  # -Z (back)
}

# Rod geometry in cube-local space (+Z face is the rod-exit face)
ROD_BASE_IN_CUBE = np.array([0.0, 0.0,  _s],                   np.float32)
ROD_TIP_IN_CUBE  = np.array([0.0, 0.0,  _s + ROD_LENGTH_MM],   np.float32)

# ─── Model calibration slots (defined in model_config.json per model) ────────
DEFAULT_SLOTS = [
    {
        "label": "Slot 1",
        "cube_center_model": [0.0, 0.0, _s],
        "cube_R_model": [1, 0, 0,
                         0, 0, 1,
                         0,-1, 0],
    },
    {
        "label": "Slot 2",
        "cube_center_model": [60.0, 0.0, _s],
        "cube_R_model": [1, 0, 0,
                         0, 0, 1,
                         0,-1, 0],
    },
]

# ─── Intrinsic calibration ────────────────────────────────────────────────────
CHECKERBOARD_COLS      = 9
CHECKERBOARD_ROWS      = 6
CHECKERBOARD_SQ_MM     = 25.0
INTRINSIC_CALIB_FRAMES = 20

# ─── Display ──────────────────────────────────────────────────────────────────
PREVIEW_W = 640
PREVIEW_H = 480
XRAY_DISPLAY_W = 480
XRAY_DISPLAY_H = 600

OVERLAY_COLOR_TIP    = (0, 255,   0)
OVERLAY_COLOR_SHAFT  = (0, 200, 255)
OVERLAY_COLOR_SLOT   = (255, 165,  0)
OVERLAY_THICKNESS    = 3
OVERLAY_TIP_RADIUS   = 10
OVERLAY_SHAFT_EXTEND = 80

# ─── Navigation ───────────────────────────────────────────────────────────────
NAV_UPDATE_MS   = 500
NAV_REALTIME    = True

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
MODELS_DIR  = f"{DATA_DIR}/models"
CAMERAS_DIR = f"{DATA_DIR}/cameras"

# ─── Misc ─────────────────────────────────────────────────────────────────────
APP_TITLE          = "FluoroSim — Simulated Fluoroscopy Navigation"
APP_VERSION        = "1.0.0"
MAX_CAMERAS_TO_SCAN = 8
