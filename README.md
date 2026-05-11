# FluoroSim — Simulated Fluoroscopy Navigation System

Radiation-free pedicle screw placement training using pre-acquired fluoroscopy
images and real-time ArUco probe tracking.

---

## Requirements

```bash
pip install opencv-contrib-python numpy Pillow
```

Python 3.10+ required (uses `match` type hints).

---

## Launch

```bash
python main.py
```

---

## Workflow Overview

```
OR visit (once)          →  Training sessions (unlimited, no radiation)
─────────────────────────────────────────────────────────────────
Acquire AP + LAT X-rays  →  Step 1: Assign AP / LAT cameras
Mark fiducials in X-rays →  Step 2: Intrinsic calibration (saved)
Compute projection mats  →  Step 3: Select spine model
                         →  Step 4: Two-slot model calibration
                         →  Navigate (real-time or snapshot)
```

---

## Probe Design (for 3D printing)

Print a **40 mm cube** with:
- Each face bearing a unique ArUco marker (DICT_4X4_50, IDs 0–5)
- Marker size: 32 mm (80% of face)
- Face ID 0 on the **front face** (+Z) where the rod exits
- A **100 mm rod** attached to face 0, sized to fit your calibration slot holes

Marker layout:
| Face | ArUco ID | Position |
|------|----------|----------|
| 0    | 0        | +Z (rod exits here) |
| 1    | 1        | −Z (back) |
| 2    | 2        | +X (right) |
| 3    | 3        | −X (left) |
| 4    | 4        | +Y (top) |
| 5    | 5        | −Y (bottom) |

Print the markers from `cv2.aruco.generateImageMarker()` or any ArUco generator.
Affix printed paper markers to the cube faces with spray adhesive or double-sided tape.

---

## Calibration Slots on the Spine Model Platform

The 3D-printed model platform must include **two cylindrical slot holes**:
- Diameter matched to the rod diameter + 0.5 mm clearance
- Depth: rod length (100 mm) — rod should seat fully so cube rests on the surface
- Slot 1 at model origin (0, 0)
- Slot 2 offset 60 mm along X axis

When the probe is fully seated:
- The cube sits flat on the model surface
- The rod is fully inserted and vertical
- Both cameras should have clear sight lines to at least one cube face

---

## OR Visit Setup

1. Place ≥6 small metal ball bearings or radiopaque markers at **known 3D positions**
   on the model. Drill/print small holes at precisely measured locations and record
   their (X, Y, Z) coordinates relative to the model origin.
2. Take AP and lateral fluoroscopy shots with the model on the operating table.
3. Open **OR Setup** in FluoroSim, load both images, click each visible marker,
   enter the matching 3D coordinates, and click **Compute**.
4. Save. This is the only fluoroscopy ever required.

---

## Camera Recommendations

| Spec | Minimum | Recommended |
|------|---------|-------------|
| Resolution | 720p | 1080p |
| FPS | 15 | 30 |
| Focus | Fixed preferred | Fixed (disable autofocus) |
| Interface | USB 2.0 | USB 3.0 |
| Example | Logitech C270 | Logitech C920 |

Fixed focus is strongly preferred — autofocus invalidates the intrinsic calibration.
Most cameras support disabling autofocus via `v4l2-ctl` (Linux) or vendor software (Windows).

---

## Data Directory Structure

```
data/
├── cameras/
│   ├── intrinsics_0.npz          ← saved camera 0 intrinsics
│   └── intrinsics_1.npz          ← saved camera 1 intrinsics
└── models/
    └── default/
        ├── model_config.json     ← slot definitions
        ├── xray_ap.png           ← AP fluoroscopy image
        ├── xray_lat.png          ← LAT fluoroscopy image
        ├── P_ap.npy              ← AP projection matrix (3×4)
        ├── P_lat.npy             ← LAT projection matrix (3×4)
        └── fiducials.json        ← OR-visit correspondence points
```

---

## Accuracy Targets

| Parameter | Design target |
|-----------|---------------|
| Positional error | ≤ 5 mm |
| Angular error | ≤ 5° |
| Response time | ≤ 500 ms (real-time) / ≤ 30 s (snapshot) |

Achievable with fixed-focus cameras, stable probe seating, and ≥6 well-distributed fiducials.

---

## Distributing to Remote Sites

Package the following for each remote user:
- `model_config.json`
- `xray_ap.png` and `xray_lat.png`
- `P_ap.npy` and `P_lat.npy`
- `fiducials.json`
- STL files for the model, platform, and camera frame
- Printable ArUco marker PDFs

Remote user needs: Python + two USB webcams + 3D printer + inkjet printer.
No fluoroscopy access required at the remote site.
