"""
core/tracker.py — ArUco detection and probe pose estimation.

Given a frame and camera intrinsics, detects any visible ArUco marker
on the probe cube and returns the full 6-DOF pose plus the rod tip
and base positions in camera space.
"""

import cv2
import numpy as np
from typing import Optional
from config import ARUCO_DICT, CUBE_FACE_OBJ_PTS, ROD_TIP_IN_CUBE, ROD_BASE_IN_CUBE


class ProbeDetection:
    """
    Result of a single probe detection.

    All positions are in the camera coordinate frame, in millimetres.
    """
    __slots__ = (
        "rvec", "tvec", "R",
        "cube_center_cam",
        "rod_base_cam",
        "rod_tip_cam",
        "rod_dir_cam",
        "marker_id",
        "image_corners",
    )

    def __init__(self, rvec, tvec, R, marker_id, image_corners):
        self.rvec        = rvec                         # (3,1) Rodrigues
        self.tvec        = tvec                         # (3,1) translation mm
        self.R           = R                            # (3,3) rotation matrix
        self.marker_id   = marker_id
        self.image_corners = image_corners              # (4,2) pixel positions

        # Positions derived from the known cube geometry
        def _xfm(pt_cube: np.ndarray) -> np.ndarray:
            return (R @ pt_cube.reshape(3, 1) + tvec).flatten()

        self.cube_center_cam = tvec.flatten()
        self.rod_base_cam    = _xfm(ROD_BASE_IN_CUBE)
        self.rod_tip_cam     = _xfm(ROD_TIP_IN_CUBE)
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
        params           = cv2.aruco.DetectorParameters()
        # Relax defaults slightly for printed markers under variable lighting
        params.adaptiveThreshWinSizeMin  = 3
        params.adaptiveThreshWinSizeMax  = 23
        params.adaptiveThreshWinSizeStep = 4
        params.minMarkerPerimeterRate    = 0.02
        self._detector   = cv2.aruco.ArucoDetector(ARUCO_DICT, params)

    # ── Public API ─────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        cam_mtx: np.ndarray,
        dist: np.ndarray,
    ) -> Optional[ProbeDetection]:
        """
        Detect probe in frame and return pose, or None if not found.
        """
        gray             = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _  = self._detector.detectMarkers(gray)

        if ids is None:
            return None

        for i, mid in enumerate(ids.flatten()):
            if mid not in CUBE_FACE_OBJ_PTS:
                continue

            obj_pts  = CUBE_FACE_OBJ_PTS[mid]
            img_pts  = corners[i][0].astype(np.float32)

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
        frame: np.ndarray,
        detection: Optional[ProbeDetection],
        cam_mtx: np.ndarray,
        dist: np.ndarray,
    ) -> np.ndarray:
        """
        Draw detection overlay on a copy of frame.
        Shows axes on the detected marker face and the rod shaft line.
        """
        out = frame.copy()
        if detection is None:
            cv2.putText(out, "No probe detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
            return out

        # Draw coordinate axes on the detected face
        cv2.drawFrameAxes(out, cam_mtx, dist,
                          detection.rvec, detection.tvec, 20.0)

        # Project rod base and tip into image to show shaft
        shaft_pts = np.array([
            detection.rod_base_cam,
            detection.rod_tip_cam,
        ], dtype=np.float32).reshape(-1, 1, 3)

        # We already have the pose — project directly
        proj_pts, _ = cv2.projectPoints(
            np.array([ROD_BASE_IN_CUBE, ROD_TIP_IN_CUBE], dtype=np.float32),
            detection.rvec, detection.tvec, cam_mtx, dist,
        )
        proj_pts = proj_pts.reshape(-1, 2).astype(int)
        cv2.line(out, tuple(proj_pts[0]), tuple(proj_pts[1]), (0, 200, 255), 3)
        cv2.circle(out, tuple(proj_pts[1]), 8, (0, 255, 0), -1)

        # Marker ID label
        cx = int(detection.image_corners[:, 0].mean())
        cy = int(detection.image_corners[:, 1].mean())
        cv2.putText(out, f"ID {detection.marker_id}", (cx - 15, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return out
