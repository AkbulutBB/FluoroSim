# FluoroSim 2.0 — gVXR Synthetic Fluoroscopy Navigation

Radiation-free pedicle screw placement training. Synthetic AP + lateral X-ray
images are generated directly from CAD/STL using **gVXR**, with a two-cube
ArUco probe tracked by two webcams and overlaid in real time.

No real fluoroscopy. No DLT calibration. No image-intensifier distortion. The
entire coordinate chain is defined in CAD, so every projection is exact.

---

## Why this architecture

The previous DLT-from-real-fluoroscopy approach was floored at 11–15 px
reprojection error by image-intensifier distortion (pincushion + S-distortion)
that cannot be estimated from 8 sparse fiducials. gVXR sidesteps this entirely:
the X-ray is **generated** from the STL with exact pinhole geometry, so the
3D→2D projection is analytically known and invertible. The steel bearings now
appear in the synthetic image at their exact CAD positions as a built-in
visual sanity check.

---

## Coordinate chain (zero physical calibration)

```
Camera  ──[ChArUco detect]──►  Board frame
Board   ──[BOARD_TO_WORLD]──►  gVXR world = CAD platform origin
World   ──[hard-stop slide]──► Spine STL origin
```

Every transform is exact by design. The only value you measure is
`BOARD_TO_WORLD` — and that comes from Fusion, not from a calibration session.

---

## Install

```bash
pip install opencv-contrib-python==4.9.0.80 numpy Pillow gvxr
```

`opencv-contrib-python` is pinned to 4.9.0.80 for numpy<2 compatibility in
the Anaconda base environment. gVXR provides `gvxrPython3`.

---

## First-run setup

### 1. Export STLs from Fusion 360
File → 3D Print → STL, units = **millimetres**:
- `models/spine.stl`
- `models/platform.stl` (optional)

### 2. Fill in CAD values in `config.py`
Four things, all read from your Fusion assembly:

| Value | What it is |
|-------|------------|
| `BEARING_POSITIONS` | (X,Y,Z) of each steel bearing centre in platform space |
| `SPINE_ORIGIN_IN_WORLD` | spine origin at its seated hard-stop position |
| `BOARD_TO_WORLD` | 4×4 rigid transform: ChArUco board origin → platform space |
| `SPINE_STL_PATH` / `PLATFORM_STL_PATH` | STL file locations |

### 3. Verify gVXR
```bash
python tools/test_gvxr.py
```
Should print `ALL TESTS PASSED` and produce `test_ap.png`, `test_lat.png`.

### 4. Calibrate both cameras (checkerboard intrinsics)
```bash
python tools/calibrate_intrinsics.py --device 0 --out data/intrinsics/cam0.npz
python tools/calibrate_intrinsics.py --device 1 --out data/intrinsics/cam1.npz
```

### 5. Print the probe markers
```bash
python tools/make_probe_sheet.py --out probe_sheet.pdf --dpi 300
```
Print at **100% scale** (no fit-to-page). Verify with the 50 mm scale bar.

---

## Run

```bash
python main.py
```

1. **Render X-ray Background** — generates AP + LAT from the STL (once, ~1–3 s)
2. **Start** — begins dual-camera tracking
3. Move the probe; its trajectory overlays live on both views (~5 Hz, 3 ms overlay)
4. **Photorealistic Snapshot** — re-renders with the K-wire as an actual
   tungsten/steel cylinder casting a true X-ray shadow (~2–10 s)

---

## Probe design (two stacked cubes)

- Two **40 mm cubes**, 5 mm spacer, 45 mm centre-to-centre pitch
- **32 mm** ArUco markers (DICT_4X4_50), one per side face, 4 faces per cube
- Bottom cube: IDs 0–3. Top cube: IDs 4–7.
- **100 mm K-wire** collinear with the probe axis, exiting the bottom cube's
  −Z face, tip 120 mm below the bottom-cube centre.

**Gluing rule:** Hold the probe handle-up / tip-down. The bottom edge of every
marker faces the K-wire tip. Both cubes oriented identically. Print cubes in
white PETG so the cube face itself serves as the ArUco quiet zone.

Fusing all visible faces from both cubes in a single solvePnP (plus VVS
refinement) gives far stronger angular constraints than any single face. Two
cameras at ~45° oblique (corner-on, each seeing two adjacent faces per cube)
constrain the pose best.

---

## Module map

```
config.py                 All geometry, camera, gVXR, and CAD parameters
main.py                   Entry point

core/
  camera.py               Threaded webcam capture + intrinsic I/O
  markers.py              BoardTracker + ProbeTracker (ChArUco + two-cube)
  navigation.py           Dual-camera pose fusion → world-space tip/base
  xray_sim.py             gVXR wrapper: render + analytic overlay + snapshot

ui/
  app.py                  Tkinter app: previews + AP/LAT panels + live overlay

tools/
  test_gvxr.py            gVXR install + render verification
  calibrate_intrinsics.py Checkerboard intrinsic calibration
  make_probe_sheet.py     A4 marker sheet with round-trip self-check

models/                   spine.stl, platform.stl  (your CAD exports)
data/
  intrinsics/             cam0.npz, cam1.npz  (saved calibration)
```

---

## gVXR 2.1 initialisation order (important)

gVXR is strict about call order — deviating produces silently all-zero images
with no error. The required sequence (handled inside `xray_sim.py`):

1. `createNewContext`
2. Create meshes / assign materials / position them
3. `setSourcePosition` → `usePointSource`
4. `resetBeamSpectrum` → `addEnergyBinToSpectrumPerPixelAtSDD`
   (`setMonoChromatic` and `addEnergyBinToSpectrum` are deprecated)
5. `setDetectorPosition` → `setDetectorUpVector`
6. `setDetectorNumberOfPixels` → `setDetectorPixelSize`
7. `addPolygonMeshAsOuterSurface` (register meshes — **after** steps 3–6)
8. `computeXRayImage`

To switch views, redo 3–6 and recompute; registered meshes persist.

On Windows/Anaconda set `GVXR_CONTEXT = "OPENGL"`; on a headless Linux box use
`"EGL"`.

---

## Error budget

| Source | Contribution |
|--------|-------------|
| Spine hard-stop play | ≤ 1 mm (consistent, could be zeroed in CAD) |
| Probe reprojection (two-cube, dual-cam) | target < 15 px → ~ sub-mm tip |
| CAD-defined transforms | exact |
| **Total target** | **< 5 mm, < 5°** |

---

*Reference for marker-based AR neuronavigation methods: Yavas et al.,
Neurosurg Focus 2021.*
