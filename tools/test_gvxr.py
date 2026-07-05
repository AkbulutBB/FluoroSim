"""
tools/test_gvxr.py — gVXR installation and render verification.

Run this BEFORE anything else. If it produces two PNG images and prints
"ALL TESTS PASSED", gVXR is correctly installed and working.

Usage
-----
    python tools/test_gvxr.py

On Windows / Anaconda
---------------------
    conda activate base
    pip install gvxr
    python tools/test_gvxr.py

The script creates test_ap.png and test_lat.png in the current directory.
"""

import sys, time
import numpy as np
import cv2

print("=" * 60)
print("FluoroSim — gVXR Installation Test")
print("=" * 60)

# ── 1. Import ──────────────────────────────────────────────────────────────
print("\n[1/5] Importing gvxrPython3 ... ", end="", flush=True)
try:
    from gvxrPython3 import gvxr
    print("OK  (version:", gvxr.getVersionOfCoreGVXR(), ")")
except ImportError as e:
    print("\nFAILED:", e)
    print("\nFix:  pip install gvxr   (or activate your Anaconda env first)")
    sys.exit(1)

# ── 2. Create OpenGL context ───────────────────────────────────────────────
print("[2/5] Creating OpenGL context ... ", end="", flush=True)
try:
    # Windows: use "OPENGL"   (needs a display — always available on desktop)
    # Linux headless: use "EGL"
    # Call with backend ONLY — gVXR negotiates the OpenGL version itself.
    # Extra positional args break version negotiation on Windows GLFW.
    context = "EGL" if sys.platform.startswith("linux") else "OPENGL"
    gvxr.createNewContext(context)
    print(f"OK  (context={context}, renderer={gvxr.getOpenGlRenderer()})")
except Exception as e:
    print("\nFAILED:", e)
    print("On Windows, ensure you have an active desktop session (not RDP minimal).")
    sys.exit(1)

# ── 3. Configure detector + source ────────────────────────────────────────
print("[3/5] Building synthetic scene ... ", end="", flush=True)
try:
    # 80 keV mono-energetic beam (typical C-arm)
    # Note: setMonoChromatic is deprecated in gVXR 2.x; use addEnergyBinToSpectrum
    gvxr.resetBeamSpectrum()
    gvxr.addEnergyBinToSpectrum(0.08, "MeV", 1000)
    gvxr.usePointSource()
    gvxr.setDetectorNumberOfPixels(512, 512)
    gvxr.setDetectorPixelSize(0.5, 0.5, "mm")

    # ── Spine-like cuboid (bone density / Ca) ──────────────────────────
    gvxr.makeCuboid("spine", 80, 60, 40, "mm")
    gvxr.setElement ("spine", "Ca")
    gvxr.setDensity ("spine", 1.92, "g/cm3")
    gvxr.addPolygonMeshAsOuterSurface("spine")
    gvxr.moveToCentre("spine")

    # ── 8 steel ball bearings at known CAD positions ─────────────────
    BEARING_POS = [
        ( 30,  5,  20), (-30,  5,  20), (  0,  5,  20),
        ( 30,  5, -20), (-30,  5, -20), (  0,  5, -20),
        ( 15,  5,   0), (  0, 20,   0),   # 7th surface, 8th = probe tip
    ]
    for i, (x, y, z) in enumerate(BEARING_POS):
        lbl = f"bearing_{i}"
        gvxr.makeSphere(lbl, 20, 20, 1.5, "mm")
        gvxr.setElement (lbl, "Fe")
        gvxr.setDensity (lbl, 7.87, "g/cm3")
        gvxr.addPolygonMeshAsOuterSurface(lbl)
        gvxr.translateNode(lbl, float(x), float(y), float(z), "mm")

    # ── K-wire cylinder (probe test) ──────────────────────────────────
    gvxr.makeCylinder("kwire", 20, 80.0, 1.0, "mm")
    gvxr.setElement  ("kwire", "Fe")
    gvxr.setDensity  ("kwire", 7.87, "g/cm3")
    gvxr.addPolygonMeshAsOuterSurface("kwire")
    # Place at 20° angle to simulate an angled pedicle screw insertion
    gvxr.rotateNode    ("kwire", 20.0, 0.0, 0.0, 1.0)
    gvxr.translateNode ("kwire", 10.0, 10.0, 0.0, "mm")

    print("OK")
except Exception as e:
    print("\nFAILED:", e)
    gvxr.destroyAllWindows()
    sys.exit(1)

# ── 4. Render AP + lateral ─────────────────────────────────────────────────

def render_view(src_pos, det_pos, det_up, label):
    """Render one view and return display image + timing."""
    gvxr.setSourcePosition  (*src_pos, "mm")
    gvxr.setDetectorPosition(*det_pos, "mm")
    gvxr.setDetectorUpVector(*det_up)
    t0 = time.perf_counter()
    gvxr.computeXRayImage()
    elapsed = time.perf_counter() - t0
    raw = np.array(gvxr.getLastXRayImage())
    # Log-compress → invert (film convention: anatomy bright)
    safe = np.clip(raw, 1e-10, None)
    norm = cv2.normalize(np.log(safe), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp = cv2.bitwise_not(norm)
    # Add view label
    cv2.putText(disp, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 128, 2, cv2.LINE_AA)
    print(f"  {label:<8}  {raw.shape}  range [{raw.min():.2f}, {raw.max():.2f}]"
          f"  rendered in {elapsed*1000:.0f} ms")
    return disp

print("[4/5] Rendering AP and lateral views:")

#  AP : source 1 m above, detector 50 mm below
ap_img = render_view(
    src_pos=(0.0, 1000.0, 0.0),
    det_pos=(0.0,  -50.0, 0.0),
    det_up =(0.0,    0.0,-1.0),
    label  ="AP",
)
# LAT: source 1 m to the right, detector 50 mm to the left
lat_img = render_view(
    src_pos=( 1000.0, 0.0, 0.0),
    det_pos=(  -50.0, 0.0, 0.0),
    det_up =(    0.0, 1.0, 0.0),
    label  ="LAT",
)

# ── 5. Save and verify ────────────────────────────────────────────────────
print("[5/5] Saving images ... ", end="", flush=True)
try:
    cv2.imwrite("test_ap.png",  ap_img)
    cv2.imwrite("test_lat.png", lat_img)
    print("OK")
except Exception as e:
    print("\nFAILED to save:", e)
    gvxr.destroyAllWindows()
    sys.exit(1)

# ── Analytic projection sanity check ──────────────────────────────────────
print("\n[Bonus] Testing analytic projection math:")
# A bearing at (30, 5, 20) should project near centre-right in AP view.
# Source at (0, 1000, 0), detector at (0, -50, 0), up=(0,0,-1)
S = np.array([0.0, 1000.0, 0.0])
D = np.array([0.0,  -50.0, 0.0])
P = np.array([30.0,   5.0, 20.0])   # first bearing

normal = (D - S) / np.linalg.norm(D - S)
right  = np.cross(normal, [0.0, 0.0, -1.0])
right /= np.linalg.norm(right)
up    = np.cross(right, normal)

d     = P - S
t     = np.dot(D - S, normal) / np.dot(d, normal)
hit   = S + t * d
delta = hit - D
u_mm  = np.dot(delta, right)
v_mm  = np.dot(delta, up)
col   = int(round(256 + u_mm / 0.5))
row   = int(round(256 - v_mm / 0.5))
print(f"  Bearing_0 (30, 5, 20) → pixel ({col}, {row})")
assert 100 < col < 400 and 100 < row < 400, \
    f"Projection sanity check FAILED: ({col}, {row}) out of range"
print("  Projection sanity check: PASSED")

# ── Cleanup ───────────────────────────────────────────────────────────────
gvxr.destroyAllWindows()

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("Output: test_ap.png, test_lat.png")
print("=" * 60)
print("\nNext step: configure config.py with your STL paths and")
print("BOARD_TO_WORLD matrix from Fusion, then run main.py.")
