"""
core/calibration.py — Slot-based camera-to-model registration.

The two calibration slots provide two known probe poses in model space.
For each slot, solvePnP gives us the cube pose in camera space.
Combining known-model-pose with known-camera-pose yields the rigid
transform T_cam_model (camera → model).

Two slots give two independent estimates; we validate consistency and
average them into a single transform.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import json

from core.tracker import ProbeDetection
from config import CAMERAS_DIR


# ─── Rigid transform helpers ──────────────────────────────────────────────────

def _make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return a 4×4 homogeneous transform from 3×3 R and 3-vector t."""
    T       = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = t.flatten()
    return T


def _invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a rigid-body 4×4 transform efficiently."""
    R    = T[:3, :3]
    t    = T[:3,  3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3,  3] = -(R.T @ t)
    return T_inv


def transform_point(T: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Apply a 4×4 transform to a 3-vector."""
    ph = np.append(pt.flatten(), 1.0)
    return (T @ ph)[:3]


def rotation_angle_deg(R: np.ndarray) -> float:
    """Return the magnitude of a rotation in degrees (axis-angle representation)."""
    trace  = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


# ─── Slot definitions ─────────────────────────────────────────────────────────

class SlotDefinition:
    """Known pose of the probe cube when seated in a calibration slot."""

    def __init__(self, cube_center_model: list, cube_R_model_flat: list, label: str = ""):
        self.label               = label
        self.cube_center_model   = np.array(cube_center_model, np.float64)
        self.cube_R_model        = np.array(cube_R_model_flat, np.float64).reshape(3, 3)
        self.T_cube_model        = _make_transform(self.cube_R_model, self.cube_center_model)

    @classmethod
    def from_dict(cls, d: dict) -> "SlotDefinition":
        return cls(
            cube_center_model  = d["cube_center_model"],
            cube_R_model_flat  = d["cube_R_model"],
            label              = d.get("label", ""),
        )


# ─── Camera-to-model transform ────────────────────────────────────────────────

class CameraModelTransform:
    """
    Rigid transform from camera space to model space for one camera.
    Stores the result of slot calibration.
    """

    def __init__(self, R_cam_to_model: np.ndarray, t_cam_to_model: np.ndarray):
        self._T = _make_transform(R_cam_to_model, t_cam_to_model)

    def point_cam_to_model(self, pt_cam: np.ndarray) -> np.ndarray:
        return transform_point(self._T, pt_cam)

    @property
    def T(self) -> np.ndarray:
        return self._T.copy()


# ─── Calibration engine ────────────────────────────────────────────────────────

class CalibrationEngine:
    """
    Computes the camera-to-model transform from two slot probe detections.

    Workflow:
        engine = CalibrationEngine([slot1_def, slot2_def])
        engine.record_slot(0, detection_from_slot_1)
        engine.record_slot(1, detection_from_slot_2)
        result = engine.compute()   # returns CameraModelTransform or raises
    """

    # Thresholds for cross-slot consistency check
    POSITION_TOLERANCE_MM  = 8.0
    ROTATION_TOLERANCE_DEG = 8.0

    def __init__(self, slot_definitions: list[SlotDefinition]):
        self._slots      = slot_definitions
        self._detections: dict[int, ProbeDetection] = {}

    @property
    def n_confirmed(self) -> int:
        return len(self._detections)

    @property
    def all_confirmed(self) -> bool:
        return self.n_confirmed == len(self._slots)

    def record_slot(self, slot_index: int, detection: ProbeDetection):
        """Store the probe detection for a given slot index."""
        if slot_index >= len(self._slots):
            raise ValueError(f"Slot index {slot_index} out of range.")
        self._detections[slot_index] = detection

    def compute(self) -> CameraModelTransform:
        """
        Compute the camera-to-model rigid transform.

        For each confirmed slot, we know:
          T_cube_cam  (from solvePnP → detection.R, detection.tvec)
          T_cube_model (from slot definition)

        Therefore:
          T_cam_model = T_cube_model @ inv(T_cube_cam)

        We compute one estimate per slot, validate consistency, then average
        the rotation (via Rodrigues mean) and translation.
        """
        if not self.all_confirmed:
            raise RuntimeError("Not all slots have been confirmed.")

        estimates: list[Tuple[np.ndarray, np.ndarray]] = []  # (R, t)

        for idx, det in self._detections.items():
            slot    = self._slots[idx]

            T_cube_cam   = _make_transform(det.R, det.tvec.flatten())
            T_cam_model  = slot.T_cube_model @ _invert_transform(T_cube_cam)

            estimates.append((T_cam_model[:3, :3].copy(), T_cam_model[:3, 3].copy()))

        # Consistency check between estimates
        if len(estimates) >= 2:
            dR = estimates[0][0].T @ estimates[1][0]
            dt = np.linalg.norm(estimates[0][1] - estimates[1][1])
            angle = rotation_angle_deg(dR)
            if angle > self.ROTATION_TOLERANCE_DEG or dt > self.POSITION_TOLERANCE_MM:
                raise ValueError(
                    f"Slot calibration inconsistency: "
                    f"rotation error {angle:.1f}° (limit {self.ROTATION_TOLERANCE_DEG}°), "
                    f"translation error {dt:.1f} mm (limit {self.POSITION_TOLERANCE_MM} mm). "
                    f"Re-seat the probe firmly and retry."
                )

        # Average translation directly; average rotation via Rodrigues
        t_mean = np.mean([e[1] for e in estimates], axis=0)
        rvecs  = [cv2.Rodrigues(e[0])[0] for e in estimates]
        rvec_mean = np.mean(rvecs, axis=0)
        R_mean, _ = cv2.Rodrigues(rvec_mean)

        return CameraModelTransform(R_mean, t_mean)


# ─── Persistence ───────────────────────────────────────────────────────────────

def save_cam_model_transform(camera_id: str, model_id: str, xfm: CameraModelTransform):
    path = Path(CAMERAS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path / f"cammodel_{camera_id}_{model_id}.npz"),
        T = xfm.T,
    )


def load_cam_model_transform(
    camera_id: str, model_id: str
) -> Optional[CameraModelTransform]:
    p = Path(CAMERAS_DIR) / f"cammodel_{camera_id}_{model_id}.npz"
    if not p.exists():
        return None
    T = np.load(str(p))["T"]
    return CameraModelTransform(T[:3, :3], T[:3, 3])
