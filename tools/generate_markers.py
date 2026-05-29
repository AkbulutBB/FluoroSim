"""
tools/generate_markers.py — Generate all printable marker assets.

Run this script once to produce two files ready for printing:

  output/charuco_board.png  — the platform CharucoBoard (72 × 126 mm)
                               Print at 100% scale and laminate.
                               Glue or screw to the cranial face of the platform.

  output/cube_net.png       — the probe cube unfolding (6 faces as a cross)
                               Print at 100% scale.
                               Cut out, fold, and glue around the 40 mm cube blank.

Usage
-----
    python -m tools.generate_markers

Both images are saved as high-resolution PNGs (300 DPI).  Open in any
image viewer, set print scale to 100% (no "fit to page"), and print.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    BOARD_ARUCO_DICT, PROBE_ARUCO_DICT,
    CHARUCO_COLS, CHARUCO_ROWS, CHARUCO_SQUARE_MM, CHARUCO_MARKER_MM,
    CUBE_SIDE_MM, MARKER_SIZE_MM,
)

OUTPUT_DIR = Path("output")
DPI        = 300
PPM        = DPI / 25.4      # pixels per millimetre at 300 DPI  ≈ 11.81 px/mm
MARGIN_MM  = 5.0             # white border around each asset


def mm(v: float) -> int:
    """Convert mm to pixels at the target DPI."""
    return round(v * PPM)


# ── CharucoBoard ───────────────────────────────────────────────────────────────

def generate_charuco_board() -> np.ndarray:
    """
    Render the platform CharucoBoard at 300 DPI.

    Board dimensions:
      CHARUCO_COLS × CHARUCO_ROWS squares × CHARUCO_SQUARE_MM mm per square
      = 4 × 7 × 18 mm = 72 × 126 mm (fits inside 80 × 140 mm platform face).
    """
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_COLS, CHARUCO_ROWS),
        CHARUCO_SQUARE_MM,
        CHARUCO_MARKER_MM,
        BOARD_ARUCO_DICT,
    )

    board_w_mm = CHARUCO_COLS * CHARUCO_SQUARE_MM
    board_h_mm = CHARUCO_ROWS * CHARUCO_SQUARE_MM
    img_w = mm(board_w_mm + 2 * MARGIN_MM)
    img_h = mm(board_h_mm + 2 * MARGIN_MM)

    img = board.generateImage(
        (img_w, img_h),
        marginSize = mm(MARGIN_MM),
        borderBits = 1,
    )
    return img


# ── Cube net ───────────────────────────────────────────────────────────────────

# Net layout (each cell is one cube face):
#
#         [ 4 +Y TOP ]
#  [3 -X] [ 0 +Z FRT ] [2 +X] [1 -Z BCK]
#         [ 5 -Y BOT ]
#
# Column indices: 0=left, 1=centre, 2=right, 3=far-right
# Row    indices: 0=top,  1=middle, 2=bottom

_NET_POSITIONS = {
    #  face_id : (col, row)
    4: (1, 0),   # TOP
    3: (0, 1),   # LEFT
    0: (1, 1),   # FRONT  (rod exits, most critical)
    2: (2, 1),   # RIGHT
    1: (3, 1),   # BACK
    5: (1, 2),   # BOTTOM
}

# Rotation (in 90° steps, counter-clockwise) to apply to each face marker
# so that when the net is folded, every marker's "top" edge aligns correctly.
# Face 4 (TOP) top edge must point toward face 0 (down in net → rotate 180°).
# Face 5 (BOT) top edge must point toward face 0 (up in net → no rotation).
# All other faces: top edge points toward +Y (up in net → no rotation).
_NET_ROTATIONS = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 2,   # 180° — top of ID4 marker points down toward face 0
    5: 0,
}

# Labels printed below each face for assembly guidance
_FACE_LABELS = {
    0: "ID0  +Z  FRONT\n(rod exits)",
    1: "ID1  -Z  BACK",
    2: "ID2  +X  RIGHT",
    3: "ID3  -X  LEFT",
    4: "ID4  +Y  TOP",
    5: "ID5  -Y  BOTTOM",
}


def _generate_face(face_id: int) -> np.ndarray:
    """
    Render a single cube face: white square with the ArUco marker centred,
    rotation applied, and a small text label below.
    Returns a square image of size mm(CUBE_SIDE_MM) pixels.
    """
    face_px   = mm(CUBE_SIDE_MM)
    marker_px = mm(MARKER_SIZE_MM)
    margin_px = (face_px - marker_px) // 2

    # Generate the raw ArUco marker
    marker_img = cv2.aruco.generateImageMarker(PROBE_ARUCO_DICT, face_id, marker_px)

    # Apply rotation
    rot = _NET_ROTATIONS[face_id]
    for _ in range(rot):
        marker_img = cv2.rotate(marker_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Place on white face background
    face = np.full((face_px, face_px), 255, dtype=np.uint8)
    face[margin_px:margin_px + marker_px, margin_px:margin_px + marker_px] = marker_img

    return face


def generate_cube_net() -> np.ndarray:
    """
    Render the full cube net as a single image at 300 DPI.

    Net size: 4 faces wide × 3 faces tall = 160 × 120 mm.
    A 2 mm gap is left between faces for fold lines.
    """
    face_px = mm(CUBE_SIDE_MM)
    gap_px  = mm(2.0)
    margin  = mm(MARGIN_MM)

    net_cols = 4
    net_rows = 3

    total_w = margin + net_cols * face_px + (net_cols - 1) * gap_px + margin
    total_h = margin + net_rows * face_px + (net_rows - 1) * gap_px + margin

    canvas = np.full((total_h, total_w), 255, dtype=np.uint8)

    for face_id, (col, row) in _NET_POSITIONS.items():
        x0 = margin + col * (face_px + gap_px)
        y0 = margin + row * (face_px + gap_px)

        face_img = _generate_face(face_id)
        canvas[y0:y0 + face_px, x0:x0 + face_px] = face_img

        # Draw a thin border around each face
        cv2.rectangle(canvas, (x0, y0), (x0 + face_px - 1, y0 + face_px - 1), 180, 1)

        # Label
        label    = _FACE_LABELS[face_id]
        font_sz  = 0.28
        y_label  = y0 + face_px + gap_px - 2
        for i, line in enumerate(label.splitlines()):
            cv2.putText(canvas, line,
                        (x0 + 2, y_label + i * 12),
                        cv2.FONT_HERSHEY_SIMPLEX, font_sz, 80, 1)

    # Add fold-line legend
    cv2.putText(canvas, "Fold along gap lines.  Glue tabs if printed with tabs.",
                (margin, total_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, 100, 1)

    return canvas


# ── Header / scale bar ────────────────────────────────────────────────────────

def _add_header(img: np.ndarray, title: str, w_mm: float, h_mm: float) -> np.ndarray:
    """Prepend a header strip with title and scale reference."""
    header_h = mm(12.0)
    header   = np.full((header_h, img.shape[1]), 240, dtype=np.uint8)

    cv2.putText(header, title,
                (mm(MARGIN_MM), mm(8.0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
    cv2.putText(header,
                f"Print at 100% scale (no fit-to-page).  "
                f"Verify dimensions: {w_mm:.0f} x {h_mm:.0f} mm.",
                (mm(MARGIN_MM), mm(11.0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, 60, 1)

    return np.vstack([header, img])


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # CharucoBoard
    print("Generating CharucoBoard…")
    board_img = generate_charuco_board()
    board_w   = CHARUCO_COLS * CHARUCO_SQUARE_MM
    board_h   = CHARUCO_ROWS * CHARUCO_SQUARE_MM
    board_img = _add_header(board_img,
                            f"FluoroSim — Platform CharucoBoard  "
                            f"({CHARUCO_COLS}x{CHARUCO_ROWS} squares, "
                            f"{CHARUCO_SQUARE_MM:.0f} mm each, DICT_5X5_50)",
                            board_w, board_h)
    out_board = str(OUTPUT_DIR / "charuco_board.png")
    cv2.imwrite(out_board, board_img)
    print(f"  Saved: {out_board}  ({board_w:.0f} x {board_h:.0f} mm @ {DPI} DPI)")

    # Cube net
    print("Generating cube net…")
    net_img = generate_cube_net()
    net_w   = 4 * CUBE_SIDE_MM
    net_h   = 3 * CUBE_SIDE_MM
    net_img = _add_header(net_img,
                          f"FluoroSim — Probe Cube Net  "
                          f"(40 mm cube, 32 mm markers, DICT_4X4_50, IDs 0–5)",
                          net_w, net_h)
    out_net = str(OUTPUT_DIR / "cube_net.png")
    cv2.imwrite(out_net, net_img)
    print(f"  Saved: {out_net}  ({net_w:.0f} x {net_h:.0f} mm @ {DPI} DPI)")

    print("\nDone.  Print both files at 100% scale (no fit-to-page).")
    print("Verify printed CharucoBoard square size = 18 mm before laminating.")


if __name__ == "__main__":
    main()
