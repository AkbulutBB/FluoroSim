"""
core/markers.py  —  Marker asset definitions (single source of truth)
=====================================================================

Builds the OpenCV objects for the platform ChArUco board and the ArUco probe
cube, and defines the 3-D corner geometry of every cube face.  Nothing else in
the codebase should construct dictionaries or boards directly — import from
here so the geometry stays consistent everywhere.

The cube-face object points below were verified corner-by-corner against the
printed sticker sheet: for every face both the "top-edge-toward" direction and
the corner winding (as seen from outside the cube, which is what the camera
sees) match the physical markers.  The detector always returns a marker's four
corners in the order [top-left, top-right, bottom-right, bottom-left] in the
marker's own canonical frame, so each face lists its corners in that same order.
"""

import cv2
import numpy as np

import config

# ── Dictionaries ────────────────────────────────────────────────────────────
CHARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(config.CHARUCO_DICT)
PROBE_DICTIONARY   = cv2.aruco.getPredefinedDictionary(config.PROBE_DICT)


def make_charuco_board() -> cv2.aruco.CharucoBoard:
    """The platform board, sized from config (7x4 landscape, 18 mm squares)."""
    return cv2.aruco.CharucoBoard(
        (config.CHARUCO_COLS, config.CHARUCO_ROWS),
        config.CHARUCO_SQUARE_MM,
        config.CHARUCO_MARKER_MM,
        CHARUCO_DICTIONARY,
    )


def make_charuco_detector() -> cv2.aruco.CharucoDetector:
    return cv2.aruco.CharucoDetector(make_charuco_board())


def make_aruco_detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corner refinement markedly improves pose stability at 40-50 cm.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(PROBE_DICTIONARY, params)


# ── Cube face geometry (cube-local coordinates, millimetres) ─────────────────
_h = config.PROBE_MARKER_MM / 2.0   # marker half-size  (16 mm)
_s = config.CUBE_SIDE_MM   / 2.0    # cube   half-size  (20 mm)

CUBE_FACE_OBJ_PTS = {
    # ID 0  +Z  FRONT  (rod exits here); marker top edge -> +Y
    0: np.array([[-_h,  _h,  _s], [ _h,  _h,  _s], [ _h, -_h,  _s], [-_h, -_h,  _s]]),
    # ID 1  -Z  BACK ; marker top edge -> +Y
    1: np.array([[ _h,  _h, -_s], [-_h,  _h, -_s], [-_h, -_h, -_s], [ _h, -_h, -_s]]),
    # ID 2  +X  RIGHT; marker top edge -> +Y
    2: np.array([[ _s,  _h,  _h], [ _s,  _h, -_h], [ _s, -_h, -_h], [ _s, -_h,  _h]]),
    # ID 3  -X  LEFT ; marker top edge -> +Y
    3: np.array([[-_s,  _h, -_h], [-_s,  _h,  _h], [-_s, -_h,  _h], [-_s, -_h, -_h]]),
    # ID 4  +Y  TOP  ; marker top edge -> -Z
    4: np.array([[-_h,  _s, -_h], [ _h,  _s, -_h], [ _h,  _s,  _h], [-_h,  _s,  _h]]),
    # ID 5  -Y  BOTTOM; marker top edge -> +Z
    5: np.array([[-_h, -_s,  _h], [ _h, -_s,  _h], [ _h, -_s, -_h], [-_h, -_s, -_h]]),
}
CUBE_FACE_OBJ_PTS = {k: v.astype(np.float64) for k, v in CUBE_FACE_OBJ_PTS.items()}

# Rod endpoints in cube-local space.
ROD_BASE_IN_CUBE = config.ROD_BASE_IN_CUBE
ROD_TIP_IN_CUBE  = config.ROD_TIP_IN_CUBE
