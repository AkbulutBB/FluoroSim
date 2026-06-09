"""
core/session.py  —  Shared application state
=============================================

One Session object is created at start-up and passed to every screen.  It
tracks which physical webcam is assigned to each role, the loaded lens
intrinsics, and the active model registration.  The two readiness flags drive
the home screen and gate the Simulation screen.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

import config
from core import paths
from core.intrinsics import CameraIntrinsics
from core.model_store import ModelRegistration


@dataclass
class Session:
    # role -> OpenCV device index chosen in the Cameras screen
    device_index: Dict[str, Optional[int]] = field(
        default_factory=lambda: {r: None for r in config.CAMERA_ROLES})
    # role -> loaded intrinsics (from disk or fresh calibration)
    intrinsics: Dict[str, Optional[CameraIntrinsics]] = field(
        default_factory=lambda: {r: None for r in config.CAMERA_ROLES})
    model: Optional[ModelRegistration] = None
    # result of the last calibration-hole verification, in mm (None until run)
    last_hole_error_mm: Optional[float] = None

    def __post_init__(self):
        paths.ensure_dirs()
        for role in config.CAMERA_ROLES:
            self.intrinsics[role] = CameraIntrinsics.load(role)

    # ---- readiness ---------------------------------------------------------
    @property
    def cameras_calibrated(self) -> bool:
        return all(self.intrinsics[r] is not None for r in config.CAMERA_ROLES)

    @property
    def cameras_assigned(self) -> bool:
        return all(self.device_index[r] is not None for r in config.CAMERA_ROLES)

    @property
    def cameras_ready(self) -> bool:
        """Lenses calibrated AND a verification has passed within tolerance."""
        return (self.cameras_calibrated
                and self.last_hole_error_mm is not None
                and self.last_hole_error_mm <= config.TIP_ERROR_TOLERANCE_MM)

    @property
    def model_ready(self) -> bool:
        return self.model is not None and self.model.is_complete

    @property
    def can_simulate(self) -> bool:
        return self.cameras_calibrated and self.model_ready
