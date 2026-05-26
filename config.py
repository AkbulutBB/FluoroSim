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

Tool profiles define the tip and base positions in cube-local space.
For the standard probe the rod exits axially from face 0.
For custom tools (chisel, awl, etc.) the tip offset is measured from
the cube face-0 surface to the physical tip of the instrument,
as entered by the user in the navigation UI.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict


# ─── ArUco ────────────────────────────────────────────────────────────────────
ARUCO_DICT_ID   = cv2.aruco.DICT_4X4_50
ARUCO_DICT      = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)


# ─── Cube geometry ────────────────────────────────────────────────────────────
CUBE_SIDE_MM    = 40.0   # printed cube side length
MARKER_SIZE_MM  = 32.0   # printed marker size on each face  (= 0.8 × cube side)
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


# ─── Tool profile ─────────────────────────────────────────────────────────────

@dataclass
class ToolProfile:
    """
    Describes the geometry of a tracked tool in cube-local space.

    tip_in_cube  : 3-D position of the instrument tip  (mm, cube frame)
    base_in_cube : 3-D position of the shaft base where it exits the cube (mm, cube frame)

    Both points are expressed in the ArUco cube coordinate system defined above.
    For an axial tool the tip lies on the +Z axis beyond face 0.
    The user-visible 'tip_distance_mm' is the distance from the cube face-0
    surface (_s) to the physical tip.
    """
    name             : str
    tip_distance_mm  : float          # distance from cube face-0 surface → tip
    description      : str = ""

    @property
    def base_in_cube(self) -> np.ndarray:
        """Shaft exits face 0 at the face centre."""
        return np.array([0.0, 0.0, _s], np.float32)

    @property
    def tip_in_cube(self) -> np.ndarray:
        """Tip is tip_distance_mm beyond face 0 along +Z."""
        return np.array([0.0, 0.0, _s + self.tip_distance_mm], np.float32)


# ─── Built-in tool profiles ───────────────────────────────────────────────────

# Standard calibration probe: 100 mm hex rod, used only for slot calibration.
STANDARD_PROBE = ToolProfile(
    name            = "Standard Probe",
    tip_distance_mm = 100.0,
    description     = "Printed hex-rod calibration probe (100 mm). "
                      "Used for slot calibration only — do not use for navigation.",
)

# Default custom tool: represents an awl / chisel.
# The user edits tip_distance_mm via the navigation UI.
DEFAULT_CUSTOM_TOOL = ToolProfile(
    name            = "Chisel / Awl",
    tip_distance_mm = 150.0,
    description     = "Custom surgical instrument with detachable ArUco cube. "
                      "Set tip distance to measured shaft length before navigating.",
)

# Registry — keys are used in the UI dropdown and in saved session data.
TOOL_PROFILES: Dict[str, ToolProfile] = {
    "standard_probe" : STANDARD_PROBE,
    "custom_tool"    : DEFAULT_CUSTOM_TOOL,
}

# Active tool key — updated at runtime when the user selects a tool in the UI.
# tracker.py reads this via get_active_tool().
_active_tool_key: str = "standard_probe"


def get_active_tool() -> ToolProfile:
    """Return the currently selected ToolProfile."""
    return TOOL_PROFILES[_active_tool_key]


def set_active_tool(key: str) -> None:
    """Set the active tool by registry key. Raises KeyError for unknown keys."""
    if key not in TOOL_PROFILES:
        raise KeyError(f"Unknown tool key: '{key}'. Valid keys: {list(TOOL_PROFILES)}")
    global _active_tool_key
    _active_tool_key = key


def set_custom_tool_distance(distance_mm: float) -> None:
    """
    Update the tip distance of the custom tool at runtime.
    Called by the navigation UI when the user enters a new length.
    """
    if distance_mm <= 0:
        raise ValueError(f"Tip distance must be positive, got {distance_mm:.1f} mm.")
    TOOL_PROFILES["custom_tool"].tip_distance_mm = distance_mm


# ─── Legacy aliases (kept so existing imports in tracker.py don't break) ──────
# These always reflect the ACTIVE tool, not the standard probe specifically.
# tracker.py should be updated to call get_active_tool() directly.
@property
def _ROD_BASE_IN_CUBE_COMPAT() -> np.ndarray:  # noqa: N802
    return get_active_tool().base_in_cube

@property
def _ROD_TIP_IN_CUBE_COMPAT() -> np.ndarray:   # noqa: N802
    return get_active_tool().tip_in_cube

# Direct module-level names that old code may reference:
# These are intentionally left as the STANDARD PROBE defaults
# so that the calibration path (which always uses standard_probe) is unaffected.
ROD_LENGTH_MM    = STANDARD_PROBE.tip_distance_mm
ROD_BASE_IN_CUBE = STANDARD_PROBE.base_in_cube
ROD_TIP_IN_CUBE  = STANDARD_PROBE.tip_in_cube


# ─── Model calibration slots ──────────────────────────────────────────────────
DEFAULT_SLOTS = [
    {
        "label": "Slot 1",
        "cube_center_model": [0.0, 0.0, _s],
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
CHECKERBOARD_COLS       = 9
CHECKERBOARD_ROWS       = 6
CHECKERBOARD_SQ_MM      = 25.0
INTRINSIC_CALIB_FRAMES  = 20


# ─── Display ─────────────────────────────────────────────────────────────────
PREVIEW_W = 640
PREVIEW_H = 480
XRAY_DISPLAY_W = 480
XRAY_DISPLAY_H = 600

OVERLAY_COLOR_TIP    = (0, 255,   0)
OVERLAY_COLOR_SHAFT  = (0, 200, 255)
OVERLAY_COLOR_SLOT   = (255, 165,  0)
OVERLAY_THICKNESS    = 3
OVERLAY_TIP_RADIUS   = 10
OVERLAY_SHAFT_EXTEND = 80   # mm to extend shaft line beyond tip for visibility


# ─── Navigation ──────────────────────────────────────────────────────────────
NAV_UPDATE_MS   = 500
NAV_REALTIME    = True


# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
MODELS_DIR  = f"{DATA_DIR}/models"
CAMERAS_DIR = f"{DATA_DIR}/cameras"


# ─── Misc ────────────────────────────────────────────────────────────────────
APP_TITLE            = "FluoroSim — Simulated Fluoroscopy Navigation"
APP_VERSION          = "1.1.0"
MAX_CAMERAS_TO_SCAN  = 8
