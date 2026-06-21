"""
tools/make_probe_sheet.py  —  Print-scale ArUco sheet for the two-cube probe
=============================================================================

Renders every probe marker defined in config.PROBE_FACE_IDS onto an A4 sheet at
TRUE physical size, each cell labelled with its ID, cube, face, and a "toward
the K-wire tip" arrow.  Because it reads the SAME table the tracker reads, the
printed sheet can never disagree with core/markers.py: edit PROBE_FACE_IDS and
re-run this, and the stickers update with it.

Gluing rule (uniform for all 8 markers)
---------------------------------------
Each marker is drawn upright, exactly as it must sit on the cube:

        +Y  (handle)  =  TOP edge of the printed marker
        -Y  (tip)     =  BOTTOM edge   <-- the arrow points here
        marker faces OUTWARD on its cube face

Stick them on in ID order.  Keep every marker's bottom edge pointing along the
probe toward the K-wire tip and you cannot mis-orient one.

Print
-----
Print at "Actual size" / 100% (NOT "fit to page").  Verify with the 50 mm scale
bar at the top — if it doesn't measure 50 mm with a ruler, the scale is wrong;
fix the print dialog before cutting anything.

Run from the repo root:
    python tools/make_probe_sheet.py
    python tools/make_probe_sheet.py --out probe_sheet.pdf --dpi 600 --buffer 2.0
"""

from __future__ import annotations
import os
import sys
import argparse

# Allow "import config" / "from core import ..." when run from tools/ or root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

import config
from core import markers as M


# ── Human-readable labels ────────────────────────────────────────────────────
CUBE_DISPLAY = {
    "bottom": 'BOTTOM cube  —  "Aruco 2"  (K-wire exits here)',
    "top":    'TOP cube  —  "Aruco 1"  (handle side)',
}
FACE_DISPLAY = {
    "PZ": "Front  (+Z)",
    "NZ": "Back  (-Z)",
    "PX": "Right  (+X)",
    "NX": "Left  (-X)",
}
CUBE_ORDER = ["bottom", "top"]   # which cube section prints first


# ── Marker raster ─────────────────────────────────────────────────────────────
def marker_image(dictionary, marker_id: int, side_mm: float, dpi: int) -> np.ndarray:
    """Crisp marker raster sized so on-page modules land on whole device pixels."""
    modules = dictionary.markerSize + 2                      # data + 1 black border ring
    device_px = side_mm / 25.4 * dpi
    px_per_mod = max(1, int(np.ceil(device_px / modules)))
    side_px = px_per_mod * modules                          # exact multiple of modules
    return cv2.aruco.generateImageMarker(dictionary, int(marker_id), side_px)


# ── Sheet ────────────────────────────────────────────────────────────────────
def build_sheet(out_path: str, dpi: int = 600, buffer_mm: float = 2.0,
                page_mm=(210.0, 297.0)) -> str:
    dictionary = M.PROBE_DICTIONARY
    side_mm = float(config.PROBE_MARKER_MM)
    cut_mm = side_mm + 2.0 * buffer_mm                      # white paper buffer around marker
    PW, PH = page_mm

    # group marker IDs by cube, in ID order within each cube
    by_cube: dict[str, list[int]] = {c: [] for c in CUBE_ORDER}
    for mid, (cube, _face) in config.PROBE_FACE_IDS.items():
        by_cube.setdefault(cube, []).append(mid)
    for c in by_cube:
        by_cube[c].sort()

    fig = plt.figure(figsize=(PW / 25.4, PH / 25.4))        # exact A4 inches
    ax = fig.add_axes([0, 0, 1, 1])                          # axes fills the page
    ax.set_xlim(0, PW); ax.set_ylim(0, PH)                   # 1 data unit == 1 mm
    ax.set_aspect("equal")                                   # ratios already match -> no distortion
    ax.axis("off")

    ink = "#111111"
    y = PH - 12.0

    # ── title + instructions ────────────────────────────────────────────────
    ax.text(12, y, f"{config.APP_NAME}  probe markers  "
                   f"({_dict_name(config.PROBE_DICT)}, {side_mm:.0f} mm)",
            fontsize=12, fontweight="bold", color=ink, va="top")
    y -= 6.5
    ax.text(12, y, "Print at ACTUAL SIZE / 100%.  Stick in ID order. "
                   "Every marker's BOTTOM edge (arrow) points along the probe "
                   "toward the K-wire tip; marker faces outward.",
            fontsize=7.5, color=ink, va="top", wrap=True)
    y -= 9.0

    # ── 50 mm scale-verification bar ─────────────────────────────────────────
    bar = 50.0; x0 = 12.0
    ax.plot([x0, x0 + bar], [y, y], color=ink, lw=1.3, solid_capstyle="butt")
    for xx in (x0, x0 + bar):
        ax.plot([xx, xx], [y - 1.6, y + 1.6], color=ink, lw=1.3)
    ax.text(x0 + bar + 4, y, "= 50 mm  (verify with a ruler before cutting)",
            fontsize=7.5, color=ink, va="center")
    y -= 10.0

    # ── marker cells, one section per cube ───────────────────────────────────
    n_cols = max(len(v) for v in by_cube.values()) if by_cube else 4
    col_w = (PW - 24.0) / n_cols
    cell_top_pad = 7.0          # space for the cube header
    label_above = 5.5           # ID line above the cut box
    label_below = 9.0           # face + arrow below the cut box
    row_h = cell_top_pad + label_above + cut_mm + label_below + 4.0

    for cube in CUBE_ORDER:
        ids = by_cube.get(cube, [])
        if not ids:
            continue
        # section header
        ax.text(12, y, CUBE_DISPLAY.get(cube, cube), fontsize=9,
                fontweight="bold", color=ink, va="top")
        row_y_top = y - cell_top_pad

        for i, mid in enumerate(ids):
            cube_i, face = config.PROBE_FACE_IDS[mid]
            cx = 12.0 + col_w * (i + 0.5)               # cell centre x
            # ID label
            ax.text(cx, row_y_top, f"ID {mid}", fontsize=9, fontweight="bold",
                    color=ink, ha="center", va="top")
            # cut box (dashed) + centred marker
            box_top = row_y_top - label_above
            bx = cx - cut_mm / 2.0
            by = box_top - cut_mm
            ax.add_patch(Rectangle((bx, by), cut_mm, cut_mm, fill=False,
                                   edgecolor="#999999", lw=0.6, linestyle=(0, (4, 3))))
            mx = cx - side_mm / 2.0
            my = box_top - buffer_mm - side_mm
            img = marker_image(dictionary, mid, side_mm, dpi)
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, origin="upper",
                      interpolation="nearest",
                      extent=(mx, mx + side_mm, my, my + side_mm), zorder=3)
            # face label + tip arrow below
            fy = by - 2.0
            ax.text(cx, fy, FACE_DISPLAY.get(face, face), fontsize=8,
                    color=ink, ha="center", va="top")
            ay_top = fy - 3.2
            ax.add_patch(FancyArrow(cx, ay_top, 0, -3.4, width=0.0,
                                    head_width=2.6, head_length=2.0,
                                    length_includes_head=True, color=ink, zorder=4))
            ax.text(cx + 3.4, ay_top - 1.7, "tip", fontsize=6.5, color=ink,
                    ha="left", va="center")

        y = row_y_top - (label_above + cut_mm + label_below) - 6.0

    # ── footer ───────────────────────────────────────────────────────────────
    ax.text(12, 8, "Verify after assembly: on the Cameras/Sim screen the red rod "
                   "overlay must land on the physical K-wire. If it points off, a "
                   "marker is mis-placed or mis-oriented.",
            fontsize=6.8, color="#555555", va="bottom")

    fig.savefig(out_path, dpi=dpi)            # NO bbox_inches='tight' -> scale preserved
    # PNG preview alongside the PDF
    png_path = os.path.splitext(out_path)[0] + ".png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return out_path


def _dict_name(dict_id: int) -> str:
    for name in dir(cv2.aruco):
        if name.startswith("DICT_") and getattr(cv2.aruco, name) == dict_id:
            return name
    return str(dict_id)


def _self_check():
    """Generate -> detect each marker -> confirm the ID round-trips."""
    dictionary = M.PROBE_DICTIONARY
    det = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    bad = []
    for mid in config.PROBE_IDS:
        img = marker_image(dictionary, mid, config.PROBE_MARKER_MM, 300)
        pad = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
        corners, ids, _ = det.detectMarkers(pad)
        ok = ids is not None and int(ids.flatten()[0]) == mid
        if not ok:
            bad.append(mid)
    return bad


def main():
    ap = argparse.ArgumentParser(description="Render the two-cube probe ArUco sheet.")
    ap.add_argument("--out", default="probe_sheet.pdf", help="output PDF path")
    ap.add_argument("--dpi", type=int, default=600, help="render DPI (>=600 recommended)")
    ap.add_argument("--buffer", type=float, default=2.0,
                    help="white paper buffer (mm) left around each marker")
    args = ap.parse_args()

    bad = _self_check()
    if bad:
        print(f"WARNING: markers failed detect round-trip: {bad}")
    else:
        print(f"Self-check OK: all {len(config.PROBE_IDS)} markers detect to their own ID.")

    out = build_sheet(args.out, dpi=args.dpi, buffer_mm=args.buffer)
    print("Wrote:", out)
    print("Preview:", os.path.splitext(out)[0] + ".png")
    print(f"Dictionary {_dict_name(config.PROBE_DICT)}, marker {config.PROBE_MARKER_MM:.0f} mm, "
          f"{len(config.PROBE_IDS)} markers.")


if __name__ == "__main__":
    main()
