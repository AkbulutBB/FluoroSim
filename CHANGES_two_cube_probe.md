# FluoroSim — two-cube probe build

This build replaces the single ArUco cube with the **two-cube probe**. Only two
source files changed; the tracker, solver, fusion, and all UI screens are
untouched (they already solve a rigid body of N markers in one shared frame).

## What changed
- **`config.py`** — new probe section:
  - 40 mm cubes, 32 mm markers, 5 mm spacer -> `CUBE_PITCH_MM = 45 mm` centre-to-centre (CAD-verified).
  - Probe-local frame: origin = bottom-cube centre; +Y = handle; -Y = K-wire tip.
  - `PROBE_FACE_IDS` — the 8 marker IDs mapped to (cube, face). **Single source of truth.**
  - K-wire collinear with the axis; tip `100 mm` from the bottom-cube centre (no lateral offset).
- **`core/markers.py`** — the 8 face object-points are now built programmatically
  from `PROBE_FACE_IDS` into the shared probe frame. The four side-face windings
  are the corner-verified ones carried over from the single-cube build.
- **`tools/make_probe_sheet.py`** (new) — renders a print-scale A4 ArUco sheet
  straight from `PROBE_FACE_IDS`, so the printed stickers can't disagree with the
  tracker. Includes a generate->detect self-check and a 50 mm scale bar.

## Verified before shipping
- All modules compile; core imports clean; 8 faces registered (IDs 0-7).
- Synthetic solvePnP round-trip: 0.0 mm tip recovery.
- **End-to-end joint solve** (two virtual cameras ~45 deg apart, 0.3 px detector
  noise): joint reprojection 0.37 px, **tip error 0.20 mm**, direction exact.
- Sheet self-check: all 8 markers detect back to their own ID; page renders A4.

## Run
```
python main.py                  # the app
python tools/make_probe_sheet.py    # regenerate the sticker sheet (-> probe_sheet.pdf)
```

## Assembly rule (uniform for all 8 markers)
Hold the probe handle-up / tip-down. Each marker is glued **facing outward with
its bottom edge toward the K-wire tip**. Face the "Front (+Z)" marker toward you:
Right (+X) on your right hand, Left (-X) on your left, Back (-Z) on the far side.
Orient **both cubes identically**. The red-rod overlay on the Cameras/Sim screen
is the truth test — if it lands on the physical wire, the markers are right.

## Robust X-ray registration (8-bearing build)
Image-intensifier images are geometrically distorted, so a single DLT floors at
~11-15 px on 8 bearings. Worse, a couple of misclicked bearings silently tilt the
whole projection and corrupt the trajectory ANGLE. This build adds:
- `dlt.ProjectionMatrix.from_correspondences_robust()` — greedy outlier rejection
  (drops the worst-reprojecting bearing while it exceeds threshold, keeps >=6),
  plus `per_point_errors()` for QA.
- `model_store.compute_view(robust=True)` — uses it, stores per-bearing residuals
  and an inlier mask (saved/loaded).
- Registration screen now reports per-bearing error and names which bearings to
  re-click; Save reports dropped bearings.
On model_01 this took reprojection 14.9->1.1 px (AP) and 11.1->0.11 px (lateral),
and shifted the lateral trajectory ~15 deg by excluding two bad bearings.
Note: dropping to 6 bearings is the DLT minimum (no redundancy) — re-click the
flagged bearings to restore a clean 8. Residual II distortion (~2-3 mm at the tip,
within the 5 mm budget) remains; tightening it needs a distortion grid or more beads.
