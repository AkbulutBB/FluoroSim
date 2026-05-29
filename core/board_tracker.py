"""
core/board_tracker.py — Platform CharucoBoard pose estimation.

PlatformBoardTracker detects the CharucoBoard mounted on the cranial face
of the training platform and returns a CameraModelTransform on every frame.

Because the board is rigidly fixed to the platform and the spine model sits
at a repeatable position via alignment pins, this transform IS the
camera-to-model transform.  No separate calibration step is needed — the
system self-registers the moment the board is visible.

Board coordinate system (= model coordinate system)
────────────────────────────────────────────────────
Origin : bottom-left corner of the CharucoBoard face
+X     : rightward across the board width
+Y     : upward along the board height
+Z     : out of the board face (toward the cameras)

All X-ray fiducial coordinates entered during OR Setup must be measured
from this origin in mm.
"""

import cv2
import numpy as np
from typing import Optional

from config import (
    BOARD_ARUCO_DICT,
    CHARUCO_COLS, CHARUCO_ROWS, CHARUCO_SQUARE_MM, CHARUCO_MARKER_MM,
    MIN_CHARUCO_CORNERS,
)
from core.transform import CameraModelTransform


class PlatformBoardTracker:
    """
    Detects the platform CharucoBoard and returns a CameraModelTransform.

    Call estimate_pose() on every frame from each camera.  If the board
    is visible, a valid transform is returned; otherwise None.
    """

    def __init__(self):
        self._board = cv2.aruco.CharucoBoard(
            (CHARUCO_COLS, CHARUCO_ROWS),
            CHARUCO_SQUARE_MM,
            CHARUCO_MARKER_MM,
            BOARD_ARUCO_DICT,
        )

        # CharucoDetector handles both marker detection and corner interpolation
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.minMarkers = 2   # minimum ArUco markers to attempt interpolation

        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.adaptiveThreshWinSizeMin  = 3
        aruco_params.adaptiveThreshWinSizeMax  = 23
        aruco_params.adaptiveThreshWinSizeStep = 4
        aruco_params.minMarkerPerimeterRate    = 0.015   # relaxed for distant view

        self._detector = cv2.aruco.CharucoDetector(
            self._board, charuco_params, aruco_params
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def estimate_pose(
        self,
        frame:   np.ndarray,
        cam_mtx: np.ndarray,
        dist:    np.ndarray,
    ) -> Optional[CameraModelTransform]:
        """
        Detect the CharucoBoard and return the camera-to-model transform.

        Returns None if the board is not visible or if fewer than
        MIN_CHARUCO_CORNERS inner corners are detected.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, _, _ = self._detector.detectBoard(gray)

        if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
            return None

        obj_pts, img_pts = self._board.matchImagePoints(charuco_corners, charuco_ids)

        if obj_pts is None or len(obj_pts) < 4:
            return None

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, cam_mtx, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return None

        R, _ = cv2.Rodrigues(rvec)
        return CameraModelTransform(R, tvec.flatten())

    def annotate(
        self,
        frame:   np.ndarray,
        cam_mtx: np.ndarray,
        dist:    np.ndarray,
    ) -> np.ndarray:
        """
        Detect and draw the board overlay on a copy of frame.
        Used in the camera preview during intrinsic calibration verification.
        """
        out  = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            self._detector.detectBoard(gray)

        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(out, marker_corners, marker_ids)

        if charuco_ids is not None and len(charuco_ids) >= MIN_CHARUCO_CORNERS:
            cv2.aruco.drawDetectedCornersCharuco(out, charuco_corners, charuco_ids)

            obj_pts, img_pts = self._board.matchImagePoints(charuco_corners, charuco_ids)
            if obj_pts is not None and len(obj_pts) >= 4:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, cam_mtx, dist,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok:
                    cv2.drawFrameAxes(out, cam_mtx, dist, rvec, tvec, 30.0)

            n = len(charuco_ids)
            cv2.putText(out, f"Board: {n} corners", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            n = len(charuco_ids) if charuco_ids is not None else 0
            cv2.putText(out, f"Board: searching ({n} corners)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

        return out

    @property
    def board(self) -> cv2.aruco.CharucoBoard:
        """Expose the underlying board object (used by the marker generator tool)."""
        return self._board
