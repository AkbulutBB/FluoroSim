# FluoroSim v2 — Simulated Fluoroscopy Training System

Radiation-free pedicle screw placement training using pre-acquired fluoroscopy
images and real-time ArUco probe tracking.  No repeated X-ray exposure after
the initial OR visit.

---

## Requirements

```bash
pip install opencv-contrib-python numpy Pillow
```

Python 3.10+ and OpenCV 4.7+ required.

---

## Quick start

```bash
python main.py
```

```bash
# Generate printable marker assets
python -m tools.generate_markers
```

---

## System overview

```
Hardware
────────
Platform      cranial face  : CharucoBoard (4×7, 18 mm squares) — laminated print
              top surface   : single cranial slot, 40 mm deep (probe rest position)
              alignment pins: ensure repeatable model placement

Probe         40 mm ArUco cube (IDs 0–5, DICT_4X4_50, 32 mm markers)
              K-wire exits the +Z face (ID 0, FRONT), 100 mm long

Cameras       Camera 1 — cranial, mounted straight down
              Camera 2 — cranial oblique, mounted at 45°
              Both cameras on the same cranial frame above the platform
              Recommended: Logitech C920 or equivalent 1080p fixed-focus USB webcam

Software flow
─────────────
OR visit (once)          →  Training sessions (unlimited, zero radiation)
──────────────────────────────────────────────────────────────────────────
Acquire AP + LAT X-rays  →  Step 1: Assign cameras
Mark ≥6 fiducials        →  Step 2: Intrinsic calibration (saved, skipped next time)
Compute projection mats  →  Step 3: Select spine model
Save                     →  Navigation: automatic registration, live overlay
```

---

## Coordinate systems

### Model / board space
Origin at the **bottom-left corner** of the CharucoBoard face (as seen from
the front of the cranial wall of the platform).

```
+X  rightward across board width
+Y  upward along board height
+Z  out of board face (toward cameras)
```

All X-ray fiducial 3D coordinates entered during OR Setup must be measured
from this origin in millimetres.

### Probe cube space
Origin at cube centre.

```
+Z  FRONT face — K-wire exits here (ID 0)
+Y  TOP face (ID 4) — cranial camera sees this during training
```

---

## Hardware build notes

### Platform CharucoBoard
- Print `output/charuco_board.png` at **100% scale** (no fit-to-page)
- Verify printed square size = 18 mm with a ruler before laminating
- Laminate (prevents warping, which causes registration drift)
- Glue or screw flat to the cranial face of the platform
- Board dimensions: 72 × 126 mm; platform face: 80 × 140 mm

### Probe cube net
- Print `output/cube_net.png` at **100% scale**
- Cut along the outer border
- Fold along the gap lines between faces
- Glue around a 40 mm cube blank (3D-printed or wooden)
- Attach a K-wire (or printed rod) to the front face (+Z, ID 0)
- Rod length from front face surface to tip: **100 mm**

### Calibration slot
- Single slot on the cranial wall of the platform, top-entry
- Slot depth: 40 mm
- Diameter: K-wire diameter + 0.5 mm clearance
- Add a shoulder collar on the rod to ensure consistent insertion depth

### Camera frame
- Both cameras mount to a single cranial post above the platform
- Camera 1 (cranial): looking straight down onto the model
- Camera 2 (oblique): angled at approximately 45° toward the platform
- Recommended mounting height: 35–50 cm above platform surface
- Disable autofocus on both cameras (use vendor software or v4l2-ctl on Linux)

---

## OR Setup (one time per model)

1. Embed ≥ 6 metal ball bearings or barium-impregnated spheres at **precisely
   known positions** in the 3D-printed spine model.  Record their (X, Y, Z)
   coordinates relative to the CharucoBoard bottom-left corner.

2. Place the model on the platform with alignment pins engaged.

3. Take AP and lateral fluoroscopy shots with a standard C-arm.

4. Export images as PNG from the C-arm workstation.

5. Open **OR Setup** in FluoroSim, load both images, click each visible
   fiducial marker in order, enter the matching 3D coordinates, click
   **Compute**, verify reprojection error (< 2 px = excellent), then **Save**.

6. This is the only fluoroscopy ever required for this model.

---

## Training session

1. Place the spine model on the platform (alignment pins snap it in).
2. Open FluoroSim.  Camera assignment and intrinsics are already saved.
3. Select the model — click **Start Training**.
4. The CharucoBoard is detected automatically.  No calibration step.
5. Insert the probe into the cranial slot as a starting position.
6. Begin the procedure.  The X-ray overlay updates in real time.

---

## Accuracy targets

| Parameter         | Design target |
|-------------------|---------------|
| Positional error  | ≤ 5 mm        |
| Angular error     | ≤ 5°          |
| Response latency  | ≤ 100 ms (real-time mode) / ≤ 30 s (snapshot) |

---

## File structure

```
fluorosim/
├── main.py
├── config.py
├── core/
│   ├── transform.py       — CameraModelTransform
│   ├── camera.py          — CameraCapture, IntrinsicCalibrator
│   ├── tracker.py         — ArucoTracker (probe cube)
│   ├── board_tracker.py   — PlatformBoardTracker (CharucoBoard → T_cam_model)
│   ├── pose_fusion.py     — fuse_poses() → FusedPose
│   ├── projection.py      — DLT, XRayOverlay
│   └── model_config.py    — ModelPackage
├── ui/
│   ├── widgets.py
│   ├── app.py             — AppState, FluoroSimApp
│   ├── camera_assign.py   — Step 1
│   ├── intrinsic_calib.py — Step 2
│   ├── model_select.py    — Step 3
│   ├── navigation.py      — Training view
│   └── or_setup.py        — OR visit
├── tools/
│   └── generate_markers.py
├── output/                — generated by generate_markers.py
└── data/
    ├── cameras/           — saved intrinsics
    └── models/
        └── <model_id>/
            ├── model_config.json
            ├── xray_ap.png
            ├── xray_lat.png
            ├── P_ap.npy
            ├── P_lat.npy
            └── fiducials.npz
```
