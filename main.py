"""
main.py — FluoroSim entry point.

Usage:
    python main.py
"""

import tkinter as tk
from ui.app import FluoroSimApp


def main():
    root = tk.Tk()
    app  = FluoroSimApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: _on_close(root, app))
    root.mainloop()


def _on_close(root: tk.Tk, app: FluoroSimApp):
    app.state.stop_cameras()
    root.destroy()


if __name__ == "__main__":
    main()
