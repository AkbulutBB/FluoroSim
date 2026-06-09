"""
FluoroSim — entry point
========================

Run from the project root:

    python main.py

Requires: opencv-contrib-python, numpy, pillow  (see requirements.txt)
"""

from core import paths
from ui.app import run


def main():
    paths.ensure_dirs()      # create data/ folders on first launch
    run()


if __name__ == "__main__":
    main()
