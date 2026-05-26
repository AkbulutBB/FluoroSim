"""
core/tracker.py — ArUco detection and probe pose estimation.

Given a frame and camera intrinsics, detects any visible ArUco marker
on the probe cube and returns the full 6-DOF pose plus the rod tip
and base positions in camera space.

Tool geometry is read from the active ToolProfile at detection time,
so switching tools in the UI takes effect immediately without restart.
"""

import cv2
import numpy as np
from typing import Optional

from config import ARUCO_DICT, CUBE_FACE_OBJ_PTS, get_active_tool


class ProbeDetection:
    """
    Result of a single probe detection.

    All positions are in the camera coordinate frame, in millimetres.
    The tip and base are computed from whichever ToolProfile was active
    at the moment detect() was called.
    """
    __slots__ = (
        "rvec", "tvec", "R",
        "cube_center_cam",
        "rod_base_cam",
        "rod_tip_cam",
        "rod_dir_cam",
        "marker_id",
        "image_corners",
        "tool_name",
        "tip_distance_mm",
    )

    def __init__(
        self,
        rvec         : np.ndarray,
        tvec         : np.ndarray,
        R            : np.ndarray,
        marker_id    : int,
        image_corners: np.ndarray,
    ):
        self.rvec          = rvec           # (3,1) Rodrigues
        self.tvec          = tvec           # (3,1) translation mm
        self.R             = R             # (3,3) rotation matrix
        self.marker_id     = marker_id
        self.image_corners = image_corners  # (4,2) pixel positions

        # Snapshot the active tool at detection time so the overlay
        # renderer always uses consistent geometry for this frame.
        tool = get_active_tool()
        self.tool_name        = tool.name
        self.tip_distance_mm  = tool.tip_distance_mm

        # Transform tool geometry from cube-local → camera space
        def _xfm(pt_cube: np.ndarray) -> np.ndarray:
            return (R @ pt_cube.reshape(3, 1) + tvec).flatten()

        self.cube_center_cam = tvec.flatten()
        self.rod_base_cam    = _xfm(tool.base_in_cube)
        self.rod_tip_cam     = _xfm(tool.tip_in_cube)

        rod_vec              = self.rod_tip_cam - self.rod_base_cam
        norm                 = np.linalg.norm(rod_vec)
        self.rod_dir_cam     = rod_vec / norm if norm > 1e-6 else rod_vec


class ArucoTracker:
    """
    Detects ArUco markers and estimates the full probe cube pose.

    The tracker iterates through all detected markers and uses the first
    one whose ID belongs to the cube face table.  Multiple simultaneously
    visible faces are not fused (unnecessary for the accuracy target here).
    """

    def __init__(self):
        params = cv2.aruco.DetectorParameters()
        # Relax defaults slightly for printed markers under variable lighting
        params.adaptiveThreshWinSizeMin  = 3
        params.adaptiveThreshWinSizeMax  = 23
        params.adaptiveThreshWinSizeStep = 4
        params.minMarkerPerimeterRate    = 0.02
        self._detector = cv2.aruco.ArucoDetector(ARUCO_DICT, params)

    # ── Public API ─────────────────────────────────────────────────────────

    def detect(
        self,
        frame  : np.ndarray,
        cam_mtx: np.ndarray,
        dist   : np.ndarray,
    ) -> Optional[ProbeDetection]:
        """
        Detect the probe cube in frame and return a ProbeDetection,
        or None if the cube is not visible.

        Tip and base positions reflect the ToolProfile that is active
        at the moment this method is called.
        """
        gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None:
            return None

        for i, mid in enumerate(ids.flatten()):
            if mid not in CUBE_FACE_OBJ_PTS:
                continue

            obj_pts = CUBE_FACE_OBJ_PTS[mid]
            img_pts = corners[i][0].astype(np.float32)

            # IPPE_SQUARE is the most stable solver for near-planar targets
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, cam_mtx, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue

            R, _ = cv2.Rodrigues(rvec)
            return ProbeDetection(rvec, tvec, R, int(mid), img_pts)

        return None

    def annotate(
        self,
        frame     : np.ndarray,
        detection : Optional[ProbeDetection],
        cam_mtx   : np.ndarray,
        dist      : np.ndarray,
    ) -> np.ndarray:
        """
        Draw detection overlay on a copy of frame.
        Shows coordinate axes on the detected marker face, the rod shaft
        line, and a label with the active tool name + tip distance.
        """
        out = frame.copy()

        if detection is None:
            cv2.putText(out, "No probe detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
            return out

        # Coordinate axes on detected face
        cv2.drawFrameAxes(out, cam_mtx, dist,
                          detection.rvec, detection.tvec, 20.0)

        # Project shaft base and tip into image
        tool     = get_active_tool()
        pts_3d   = np.array(
            [tool.base_in_cube, tool.tip_in_cube], dtype=np.float32
        )
        proj_pts, _ = cv2.projectPoints(
            pts_3d, detection.rvec, detection.tvec, cam_mtx, dist,
        )
        proj_pts = proj_pts.reshape(-1, 2).astype(int)

        cv2.line(out, tuple(proj_pts[0]), tuple(proj_pts[1]), (0, 200, 255), 3)
        cv2.circle(out, tuple(proj_pts[1]), 8, (0, 255, 0), -1)

        # Tool label
        cx = int(detection.image_corners[:, 0].mean())
        cy = int(detection.image_corners[:, 1].mean())
        label = f"{detection.tool_name}  |  tip {detection.tip_distance_mm:.0f} mm"
        cv2.putText(out, label, (cx - 60, cy - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        return out
