"""
main.py — FluoroSim entry point.

Usage:
    python main.py

Note for Spyder / IPython users
────────────────────────────────
Tkinter does not release its root window when a script finishes inside an
IPython kernel.  Running the script a second time without restarting the
kernel leaves the old window open and causes PhotoImage context errors.

The _cleanup() call below destroys any lingering root before creating a
fresh one.  If errors persist, restart the Spyder kernel (Ctrl+.) between
runs, or launch from a plain terminal:  python main.py
"""

import tkinter as tk
from ui.app import FluoroSimApp


def _cleanup():
    """Destroy any Tk root left over from a previous IPython run."""
    try:
        if tk._default_root is not None:
            tk._default_root.destroy()
    except Exception:
        pass


def main():
    _cleanup()

    root = tk.Tk()
    app  = FluoroSimApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: _on_close(root, app))
    root.mainloop()


def _on_close(root: tk.Tk, app: FluoroSimApp):
    app.state.stop_cameras()
    root.destroy()


if __name__ == "__main__":
    main()
