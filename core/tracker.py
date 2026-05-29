"""
core/tracker.py — Probe cube ArUco detection and pose estimation.

Given a camera frame and its intrinsics, ArucoTracker detects whichever
face of the 6-faced probe cube is currently visible and returns the full
6-DOF pose of the cube plus the rod tip and base positions in camera space.

Only the PROBE_ARUCO_DICT (DICT_4X4_50) is searched here, so the platform
CharucoBoard markers (DICT_5X5_50) are never confused with probe faces.
"""

import cv2
import numpy as np
from typing import Optional

from config import PROBE_ARUCO_DICT, CUBE_FACE_OBJ_PTS, ROD_BASE_IN_CUBE, ROD_TIP_IN_CUBE


class ProbeDetection:
    """
    Result of a single probe detection.

    All positions are expressed in the camera coordinate frame (millimetres).
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

    def __init__(
        self,
        rvec:          np.ndarray,
        tvec:          np.ndarray,
        R:             np.ndarray,
        marker_id:     int,
        image_corners: np.ndarray,
    ):
        self.rvec          = rvec            # (3, 1) Rodrigues rotation
        self.tvec          = tvec            # (3, 1) translation in mm
        self.R             = R              # (3, 3) rotation matrix
        self.marker_id     = marker_id
        self.image_corners = image_corners   # (4, 2) pixel positions

        def _xfm(pt_cube: np.ndarray) -> np.ndarray:
            """Transform a cube-local point into camera space."""
            return (R @ pt_cube.reshape(3, 1) + tvec).flatten()

        self.cube_center_cam = tvec.flatten()
        self.rod_base_cam    = _xfm(ROD_BASE_IN_CUBE)
        self.rod_tip_cam     = _xfm(ROD_TIP_IN_CUBE)

        rod_vec           = self.rod_tip_cam - self.rod_base_cam
        norm              = np.linalg.norm(rod_vec)
        self.rod_dir_cam  = rod_vec / norm if norm > 1e-6 else rod_vec


class ArucoTracker:
    """
    Detects the probe cube and returns its 6-DOF pose.

    Iterates through all markers detected in the frame and uses the first
    one whose ID belongs to the cube face table (IDs 0–5).  A single
    visible face is sufficient for full pose estimation via solvePnP.
    """

    def __init__(self):
        params = cv2.aruco.DetectorParameters()
        # Slightly relaxed thresholds for variable lighting conditions
        params.adaptiveThreshWinSizeMin  = 3
        params.adaptiveThreshWinSizeMax  = 23
        params.adaptiveThreshWinSizeStep = 4
        params.minMarkerPerimeterRate    = 0.02
        self._detector = cv2.aruco.ArucoDetector(PROBE_ARUCO_DICT, params)

    def detect(
        self,
        frame:   np.ndarray,
        cam_mtx: np.ndarray,
        dist:    np.ndarray,
    ) -> Optional[ProbeDetection]:
        """
        Detect the probe cube in frame and return a ProbeDetection, or None.
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

            # IPPE_SQUARE is the most stable solver for near-planar square targets
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
        frame:   np.ndarray,
        det:     Optional[ProbeDetection],
        cam_mtx: np.ndarray,
        dist:    np.ndarray,
    ) -> np.ndarray:
        """
        Draw detection overlay on a copy of frame for live camera previews.
        Shows coordinate axes on the detected face and the rod shaft.
        """
        out = frame.copy()

        if det is None:
            cv2.putText(out, "Probe: not detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
            return out

        # Coordinate axes on the detected marker face
        cv2.drawFrameAxes(out, cam_mtx, dist, det.rvec, det.tvec, 20.0)

        # Rod shaft projected into the image
        proj, _ = cv2.projectPoints(
            np.array([ROD_BASE_IN_CUBE, ROD_TIP_IN_CUBE], dtype=np.float32),
            det.rvec, det.tvec, cam_mtx, dist,
        )
        p = proj.reshape(-1, 2).astype(int)
        cv2.line(out,   tuple(p[0]), tuple(p[1]), (0, 200, 255), 3)
        cv2.circle(out, tuple(p[1]), 8, (0, 255, 0), -1)

        # Face ID label
        cx = int(det.image_corners[:, 0].mean())
        cy = int(det.image_corners[:, 1].mean())
        cv2.putText(out, f"Face {det.marker_id}", (cx - 20, cy - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(out, "Probe: detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return out
