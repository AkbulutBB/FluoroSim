"""
core/markers.py  —  Marker asset definitions (single source of truth)
=====================================================================

Builds the OpenCV objects for the platform ChArUco board and the two-cube ArUco
probe, and defines the 3-D corner geometry of every cube face IN ONE SHARED
PROBE FRAME.  Nothing else in the codebase should construct dictionaries,
boards, or face geometry directly — import from here so the geometry stays
consistent everywhere.

Two-cube probe
--------------
The probe is two ArUco cubes rigidly joined along the probe axis (see config:
PROBE_BOTTOM_CENTER / PROBE_TOP_CENTER / PROBE_FACE_IDS).  Because the cubes are
rigid relative to each other, the whole probe is ONE rigid body: we express
every marker's corners in a single probe-local frame and hand them to the same
multi-view solver the single cube used.  The solved pose's +Y axis is the line
running through both cube centres — i.e. exactly "the line between the two
cubes" that points the K-wire.

Corner winding
--------------
The detector always returns a marker's four corners in the order
[top-left, top-right, bottom-right, bottom-left] in the marker's own canonical
frame, so each face template lists its corners in that same order, with the
marker's top edge pointing toward probe +Y (the handle side).  The four
side-face templates below are the corner-verified windings carried over from
the single-cube build (they were the FRONT/BACK/RIGHT/LEFT faces there); only
which printed ID sits on which face/cube is new, and that mapping lives in
config.PROBE_FACE_IDS.
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


# ── Cube side-face templates (one cube's own frame, millimetres) ─────────────
# Each template is the four corners [TL, TR, BR, BL] of a marker centred on one
# side face, with its top edge toward +Y.  _h is the marker half-size, _s is the
# cube half-size (the face stand-off from that cube's centre).
_h = config.PROBE_MARKER_MM / 2.0   # marker half-size  (16 mm)
_s = config.CUBE_SIDE_MM   / 2.0    # cube   half-size  (20 mm)

_FACE_TEMPLATES = {
    # +Z  FRONT ; top edge -> +Y
    "PZ": np.array([[-_h,  _h,  _s], [ _h,  _h,  _s], [ _h, -_h,  _s], [-_h, -_h,  _s]]),
    # -Z  BACK  ; top edge -> +Y
    "NZ": np.array([[ _h,  _h, -_s], [-_h,  _h, -_s], [-_h, -_h, -_s], [ _h, -_h, -_s]]),
    # +X  RIGHT ; top edge -> +Y
    "PX": np.array([[ _s,  _h,  _h], [ _s,  _h, -_h], [ _s, -_h, -_h], [ _s, -_h,  _h]]),
    # -X  LEFT  ; top edge -> +Y
    "NX": np.array([[-_s,  _h, -_h], [-_s,  _h,  _h], [-_s, -_h,  _h], [-_s, -_h, -_h]]),
}
_CUBE_CENTER = {
    "bottom": config.PROBE_BOTTOM_CENTER,
    "top":    config.PROBE_TOP_CENTER,
}

# ── Per-ID corner geometry in the SHARED probe frame ─────────────────────────
# Built from config.PROBE_FACE_IDS so the mapping lives in exactly one place.
CUBE_FACE_OBJ_PTS = {}
for _mid, (_cube, _face) in config.PROBE_FACE_IDS.items():
    if _face not in _FACE_TEMPLATES:
        raise ValueError(f"PROBE_FACE_IDS[{_mid}] uses unknown face '{_face}'")
    if _cube not in _CUBE_CENTER:
        raise ValueError(f"PROBE_FACE_IDS[{_mid}] uses unknown cube '{_cube}'")
    CUBE_FACE_OBJ_PTS[_mid] = (_FACE_TEMPLATES[_face] + _CUBE_CENTER[_cube]).astype(np.float64)

# Rod endpoints in the shared probe frame (collinear with +/-Y; from config).
ROD_BASE_IN_CUBE = config.ROD_BASE_IN_CUBE
ROD_TIP_IN_CUBE  = config.ROD_TIP_IN_CUBE
