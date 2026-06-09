"""
ui/app.py  —  Application shell and screen router
==================================================

Creates the root window, holds the single Session, and switches between the
four screens (Home, Cameras, Model, Simulation).  Each screen is a Frame with
optional on_show()/on_hide() hooks so it can start and stop its camera loop
cleanly when navigated to or away from.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk

import config
from core.session import Session
from ui import widgets as W


class Screen(ttk.Frame):
    """Base class: override on_show / on_hide as needed."""
    def on_show(self):  ...
    def on_hide(self):  ...


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(config.APP_NAME)
        self.geometry("1180x760")
        self.minsize(980, 680)
        self.configure(bg=W.BG)
        self.session = Session()

        self._init_style()

        self._container = ttk.Frame(self, style="App.TFrame")
        self._container.pack(fill="both", expand=True)

        # screens are created lazily to avoid opening cameras until needed
        self._screens: dict = {}
        self._current: Screen | None = None

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.show("home")

    # ---- styling -----------------------------------------------------------
    def _init_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background=W.BG)
        style.configure("Panel.TFrame", background=W.PANEL)
        style.configure("TFrame", background=W.BG)
        style.configure("TLabel", background=W.BG, foreground=W.FG, font=("Segoe UI", 11))
        style.configure("Panel.TLabel", background=W.PANEL, foreground=W.FG, font=("Segoe UI", 11))
        style.configure("Muted.TLabel", background=W.BG, foreground=W.MUTED, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=W.BG, foreground=W.FG, font=("Segoe UI", 22, "bold"))
        style.configure("H2.TLabel", background=W.BG, foreground=W.FG, font=("Segoe UI", 14, "bold"))
        style.configure("TButton", font=("Segoe UI", 11), padding=8)
        style.configure("Accent.TButton", font=("Segoe UI", 12, "bold"), padding=10)
        style.map("Accent.TButton",
                  background=[("active", "#3b87e0"), ("!disabled", W.ACCENT)],
                  foreground=[("!disabled", "#04122b")])
        style.configure("TEntry", fieldbackground=W.PANEL, foreground=W.FG)
        style.configure("Treeview", background=W.PANEL, fieldbackground=W.PANEL,
                        foreground=W.FG, rowheight=24)
        style.configure("TCombobox", fieldbackground=W.PANEL, foreground=W.FG)

    # ---- navigation --------------------------------------------------------
    def show(self, name: str):
        if self._current is not None:
            self._current.on_hide()
            self._current.pack_forget()

        if name not in self._screens:
            self._screens[name] = self._build(name)
        screen = self._screens[name]
        screen.pack(fill="both", expand=True)
        self._current = screen
        screen.on_show()

    def _build(self, name: str) -> Screen:
        # imported here to avoid circular imports at module load
        from ui.screen_home import HomeScreen
        from ui.screen_cameras import CamerasScreen
        from ui.screen_model import ModelScreen
        from ui.screen_sim import SimulationScreen
        return {
            "home":    HomeScreen,
            "cameras": CamerasScreen,
            "model":   ModelScreen,
            "sim":     SimulationScreen,
        }[name](self._container, self)

    def _on_close(self):
        if self._current is not None:
            self._current.on_hide()
        self.destroy()


def run():
    App().mainloop()
