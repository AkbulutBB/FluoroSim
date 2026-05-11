"""
FluoroSim - Global configuration and probe geometry.

All physical measurements are in millimetres.
The ArUco cube coordinate system:
  +Z  = face 0 (rod exits here)
  -Z  = face 1 (opposite)
  +X  = face 2 (right)
  -X  = face 3 (left)
  +Y  = face 4 (top)
  -Y  = face 5 (bottom)
"""

import cv2
import numpy as np

# ─── ArUco ────────────────────────────────────────────────────────────────────
ARUCO_DICT_ID   = cv2.aruco.DICT_4X4_50
ARUCO_DICT      = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)

# ─── Probe geometry ───────────────────────────────────────────────────────────
CUBE_SIDE_MM    = 40.0   # printed cube side length
MARKER_SIZE_MM  = 32.0   # printed marker size on each face  (= 0.8 × cube side)
ROD_LENGTH_MM   = 100.0  # rod extending from face 0
ROD_FACE_ID     = 0      # ArUco ID on the rod-exit face

_h = MARKER_SIZE_MM / 2.0   # half marker
_s = CUBE_SIDE_MM   / 2.0   # half cube

# 3-D corners of each marker face in cube-local space.
# Order matches OpenCV ArUco convention: top-left, top-right, bottom-right, bottom-left.
CUBE_FACE_OBJ_PTS: dict[int, np.ndarray] = {
    0: np.array([[-_h,  _h,  _s], [ _h,  _h,  _s], [ _h, -_h,  _s], [-_h, -_h,  _s]], np.float32),  # +Z
    1: np.array([[ _h,  _h, -_s], [-_h,  _h, -_s], [-_h, -_h, -_s], [ _h, -_h, -_s]], np.float32),  # -Z
    2: np.array([[ _s,  _h,  _h], [ _s,  _h, -_h], [ _s, -_h, -_h], [ _s, -_h,  _h]], np.float32),  # +X
    3: np.array([[-_s,  _h, -_h], [-_s,  _h,  _h], [-_s, -_h,  _h], [-_s, -_h, -_h]], np.float32),  # -X
    4: np.array([[-_h,  _s, -_h], [ _h,  _s, -_h], [ _h,  _s,  _h], [-_h,  _s,  _h]], np.float32),  # +Y
    5: np.array([[-_h, -_s,  _h], [ _h, -_s,  _h], [ _h, -_s, -_h], [-_h, -_s, -_h]], np.float32),  # -Y
}

# Rod geometry in cube-local space
ROD_BASE_IN_CUBE = np.array([0.0, 0.0,  _s],                   np.float32)
ROD_TIP_IN_CUBE  = np.array([0.0, 0.0,  _s + ROD_LENGTH_MM],   np.float32)

# ─── Model calibration slots (defined in model_config.json per model) ─────────
# These are the defaults for the bundled "default" model.
# cube_center_model: where the cube centre sits in model space when probe is seated
# cube_R_model: rotation matrix (3×3, flat row-major) of cube when seated
DEFAULT_SLOTS = [
    {
        "label": "Slot 1",
        "cube_center_model": [0.0, 0.0, _s],
        # Rod points in -Y direction (probe inserted from above, rod into hole)
        "cube_R_model": [
            1, 0, 0,
            0, 0, 1,
            0,-1, 0,
        ],
    },
    {
        "label": "Slot 2",
        "cube_center_model": [60.0, 0.0, _s],
        "cube_R_model": [
            1, 0, 0,
            0, 0, 1,
            0,-1, 0,
        ],
    },
]

# ─── Intrinsic calibration ────────────────────────────────────────────────────
CHECKERBOARD_COLS   = 9      # inner corners
CHECKERBOARD_ROWS   = 6
CHECKERBOARD_SQ_MM  = 25.0   # physical square size
INTRINSIC_CALIB_FRAMES = 20  # frames to collect before computing

# ─── Display ─────────────────────────────────────────────────────────────────
PREVIEW_W = 640
PREVIEW_H = 480
XRAY_DISPLAY_W = 480
XRAY_DISPLAY_H = 600

OVERLAY_COLOR_TIP    = (0, 255,   0)   # green dot at rod tip
OVERLAY_COLOR_SHAFT  = (0, 200, 255)   # yellow-cyan shaft line
OVERLAY_COLOR_SLOT   = (255, 165,  0)  # orange ring for slot detection
OVERLAY_THICKNESS    = 3
OVERLAY_TIP_RADIUS   = 10
OVERLAY_SHAFT_EXTEND = 80   # mm to extend shaft line beyond tip for visibility

# ─── Navigation ──────────────────────────────────────────────────────────────
NAV_UPDATE_MS   = 500    # milliseconds between overlay refreshes (real-time mode)
NAV_REALTIME    = True   # start in real-time mode

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
MODELS_DIR  = f"{DATA_DIR}/models"
CAMERAS_DIR = f"{DATA_DIR}/cameras"

# ─── Misc ─────────────────────────────────────────────────────────────────────
APP_TITLE   = "FluoroSim — Simulated Fluoroscopy Navigation"
APP_VERSION = "1.0.0"
MAX_CAMERAS_TO_SCAN = 8  # how many indices to probe for available cameras
