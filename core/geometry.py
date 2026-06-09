"""
core/geometry.py  —  Rigid (camera <-> model) transforms
=========================================================

A RigidTransform holds a rotation + translation and applies it to 3-D points.
We always store it as the CAMERA -> MODEL direction, because the probe's rod
endpoints arrive in camera space and must be carried back into model space.

Key fact this module exists to get right:
    cv2.solvePnP returns the MODEL -> CAMERA transform
        X_cam = R @ X_model + t
    so to obtain CAMERA -> MODEL we invert it:
        R_cm = R.T ,  t_cm = -R.T @ t
Use RigidTransform.from_model_to_camera(rvec, tvec) and the inversion is done
for you.
"""

from __future__ import annotations
import cv2
import numpy as np


class RigidTransform:
    """Camera -> model rigid transform.  X_model = R @ X_cam + t."""

    __slots__ = ("R", "t")

    def __init__(self, R: np.ndarray, t: np.ndarray):
        self.R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(t, dtype=np.float64).reshape(3)

    @classmethod
    def from_model_to_camera(cls, rvec: np.ndarray, tvec: np.ndarray) -> "RigidTransform":
        """Build a CAMERA->MODEL transform from a solvePnP (MODEL->CAMERA) result."""
        R_mc, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        t_mc = np.asarray(tvec, dtype=np.float64).reshape(3)
        R_cm = R_mc.T
        t_cm = -R_mc.T @ t_mc
        return cls(R_cm, t_cm)

    def apply(self, pt_cam: np.ndarray) -> np.ndarray:
        """Map a single camera-space point (or N x 3 array) into model space."""
        pt = np.asarray(pt_cam, dtype=np.float64)
        if pt.ndim == 1:
            return self.R @ pt + self.t
        return (self.R @ pt.T).T + self.t

    @property
    def camera_origin_in_model(self) -> np.ndarray:
        """Where the camera centre sits in model space (handy for sanity checks)."""
        return self.t.copy()
