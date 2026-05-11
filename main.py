"""
main.py — FluoroSim entry point.

Launch with:
    python main.py

Requirements:
    pip install opencv-contrib-python numpy Pillow
"""

import sys
import tkinter as tk
from pathlib import Path

# Ensure project root is on the path regardless of where the script is invoked from
sys.path.insert(0, str(Path(__file__).parent))

from ui.app import FluoroSimApp
from config import APP_TITLE


def main():
    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("1440x900")
    root.minsize(1200, 780)

    app = FluoroSimApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
