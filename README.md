# FluoroSim

Radiation-free **simulated fluoroscopy** for pedicle-screw training. Two webcams
watch a ChArUco-marked platform and an ArUco-marked probe; the system computes
where the probe is in model space and paints its trajectory onto pre-acquired
AP and lateral X-rays — no live radiation.

## Install

From the project folder:

```
pip install -r requirements.txt
```

(`tkinter` ships with standard Python / Anaconda — no install needed.)

## Run

```
python main.py
```

Everything the app saves goes into a `data/` folder created next to the code:

```
data/
  cameras/         camera_ap.json, camera_lat.json   (lens calibrations)
  models/          <name>.json + copied X-ray images (registrations)
  verification.json                                  (calibration-hole point)
```

## Workflow

1. **Cameras → Calibrate.** For each camera, pick its device, then show the
   ChArUco board from ~8+ angles, clicking *Capture view* each time, and press
   *Calibrate & save*. This is one-time per lens; moving the camera afterwards
   needs no recalibration.
2. **Cameras → Verify.** Put the probe tip in the platform calibration hole
   (set its known model coordinate in the box). The readout shows the measured
   tip vs the known point and the agreement between the two cameras. When the
   error is within tolerance, press *Mark cameras verified*. Re-run this any
   time the cameras get bumped.
3. **Model registration.** Enter the steel-bearing coordinates (one `X Y Z` per
   line), load the AP X-ray and click the bearings in that order, do the same
   for the lateral, then *Compute & save*. Aim for reprojection under ~2 px.
4. **Simulation.** Unlocks once cameras are calibrated and a model is complete.
   The probe trajectory is overlaid live on both X-rays.

## Coordinate convention

The ChArUco board defines model space: origin at its bottom-left inner corner,
**+X** along the long 126 mm edge, **+Y** along the short 72 mm edge, **+Z** out
of the face toward the cameras, millimetres throughout. All fiducial coordinates
and the calibration-hole coordinate must be measured in this frame.

## Hardware constants

Defined in `config.py` — match your printed assets, do not change unless you
reprint:

- ChArUco board: `DICT_5X5_50`, 7×4 landscape, 18 mm squares, 14.4 mm markers
- Probe cube: `DICT_4X4_50`, IDs 0–5, 32 mm markers on a 40 mm white PETG cube
- Rod: 100 mm from the front face (ID 0); tip at `(0, 0, 120)` in cube space
