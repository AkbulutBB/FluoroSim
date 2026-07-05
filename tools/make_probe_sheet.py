"""
tools/make_probe_sheet.py — Print-scale ArUco marker sheet for the two-cube probe.

Generates an A4 PNG/PDF containing the 8 cube-face markers (IDs 0–7) at the
exact printed size from config.py (MARKER_SIZE_MM), each with its ID label and
a quiet-zone border. Includes a 50 mm scale bar so you can verify your printer
is not rescaling.

A round-trip self-check re-detects every generated marker to confirm IDs are
correct before you commit ink to paper.

Usage
-----
    python tools/make_probe_sheet.py --out probe_sheet.png
    python tools/make_probe_sheet.py --out probe_sheet.pdf --dpi 300

Gluing rule (matches config geometry)
-------------------------------------
    Hold probe handle-up / tip-down.
    Bottom edge of every marker faces the K-wire tip.
    Both cubes oriented identically.
    Bottom cube = IDs 0–3, top cube = IDs 4–7.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg

# A4 at given DPI
A4_MM = (210.0, 297.0)


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def generate_marker(marker_id: int, size_px: int) -> np.ndarray:
    img = cv2.aruco.generateImageMarker(cfg.ARUCO_PROBE_DICT, marker_id, size_px)
    return img


def build_sheet(dpi: int) -> np.ndarray:
    W = mm_to_px(A4_MM[0], dpi)
    H = mm_to_px(A4_MM[1], dpi)
    sheet = np.full((H, W), 255, np.uint8)

    marker_px = mm_to_px(cfg.MARKER_SIZE_MM, dpi)
    quiet_px  = mm_to_px(4.0, dpi)            # 4 mm quiet zone
    cell      = marker_px + 2 * quiet_px
    label_px  = mm_to_px(8.0, dpi)

    margin_x = mm_to_px(15.0, dpi)
    margin_y = mm_to_px(20.0, dpi)
    gap      = mm_to_px(10.0, dpi)

    cols = 2
    face_names = {0:"BOT +X",1:"BOT -X",2:"BOT +Y",3:"BOT -Y",
                  4:"TOP +X",5:"TOP -X",6:"TOP +Y",7:"TOP -Y"}

    for mid in range(8):
        r = mid // cols
        c = mid % cols
        x0 = margin_x + c * (cell + gap)
        y0 = margin_y + r * (cell + label_px + gap)

        # Quiet-zone box
        cv2.rectangle(sheet, (x0, y0), (x0 + cell, y0 + cell), 0, 1)
        marker = generate_marker(mid, marker_px)
        sheet[y0 + quiet_px : y0 + quiet_px + marker_px,
              x0 + quiet_px : x0 + quiet_px + marker_px] = marker

        # Label
        cv2.putText(sheet, f"ID {mid}  ({face_names[mid]})",
                    (x0, y0 + cell + mm_to_px(6.0, dpi)),
                    cv2.FONT_HERSHEY_SIMPLEX, dpi / 300.0 * 0.7, 0, 2, cv2.LINE_AA)

    # 50 mm scale bar at the bottom
    bar_len = mm_to_px(50.0, dpi)
    bx = margin_x
    by = H - mm_to_px(15.0, dpi)
    cv2.line(sheet, (bx, by), (bx + bar_len, by), 0, 3)
    for tick_mm in (0, 10, 20, 30, 40, 50):
        tx = bx + mm_to_px(tick_mm, dpi)
        cv2.line(sheet, (tx, by - 8), (tx, by + 8), 0, 2)
    cv2.putText(sheet, "50 mm scale bar — measure to verify 100% print scale",
                (bx, by - mm_to_px(4.0, dpi)),
                cv2.FONT_HERSHEY_SIMPLEX, dpi / 300.0 * 0.6, 0, 2, cv2.LINE_AA)

    # Header
    cv2.putText(sheet,
                f"FluoroSim probe markers — {cfg.MARKER_SIZE_MM:.0f}mm on "
                f"{cfg.CUBE_SIDE_MM:.0f}mm cubes (DICT_4X4_50)",
                (margin_x, mm_to_px(12.0, dpi)),
                cv2.FONT_HERSHEY_SIMPLEX, dpi / 300.0 * 0.65, 0, 2, cv2.LINE_AA)

    return sheet


def self_check(dpi: int) -> bool:
    """Re-detect every generated marker to confirm IDs round-trip correctly."""
    params   = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(cfg.ARUCO_PROBE_DICT, params)
    marker_px = mm_to_px(cfg.MARKER_SIZE_MM, dpi)
    all_ok = True
    for mid in range(8):
        img = generate_marker(mid, marker_px)
        padded = cv2.copyMakeBorder(img, 40, 40, 40, 40,
                                    cv2.BORDER_CONSTANT, value=255)
        corners, ids, _ = detector.detectMarkers(padded)
        ok = ids is not None and int(ids.flatten()[0]) == mid
        all_ok &= ok
        print(f"  ID {mid}: {'OK' if ok else 'FAILED ROUND-TRIP'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="probe_sheet.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    print("Self-check (round-trip detection):")
    if not self_check(args.dpi):
        print("WARNING: round-trip detection failed for one or more markers.")
    else:
        print("  All 8 markers detected correctly.")

    sheet = build_sheet(args.dpi)

    out = Path(args.out)
    if out.suffix.lower() == ".pdf":
        # Save via PIL for PDF
        from PIL import Image
        Image.fromarray(sheet).save(str(out), "PDF", resolution=args.dpi)
    else:
        cv2.imwrite(str(out), sheet)
    print(f"Saved {out}  ({args.dpi} DPI). Print at 100% scale (no fit-to-page).")


if __name__ == "__main__":
    main()
