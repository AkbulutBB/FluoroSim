"""
main.py — FluoroSim entry point.

Opens the centralized setup launcher. Everything is driven from there:
gVXR verification, camera detection, calibration, probe-sheet generation,
and finally launching the navigation window. No command-line flags.

Launch:
    python main.py

Requirements:
    pip install opencv-contrib-python==4.9.0.80 "numpy<2" Pillow gvxr

Before first run:
    Fill in the CAD values in config.py (BEARING_POSITIONS,
    SPINE_TO_WORLD, BOARD_TO_WORLD, STL paths). Everything else
    is handled by the on-screen setup steps.
"""

import logging
import tkinter as tk

from ui.launcher import LauncherWindow


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    root = tk.Tk()
    LauncherWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
