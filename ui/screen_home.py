"""
ui/screen_home.py  —  Home / hub
================================

The landing screen.  Shows the two readiness gates (Cameras, Model) and routes
to each setup screen.  The "Start Simulation" button only enables when both the
lenses are calibrated and a model registration is loaded and complete.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk

import config
from ui.app import Screen
from ui import widgets as W


class HomeScreen(Screen):
    def __init__(self, master, app):
        super().__init__(master, style="App.TFrame")
        self.app = app

        wrap = ttk.Frame(self, style="App.TFrame")
        wrap.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(wrap, text=config.APP_NAME, style="Title.TLabel").pack(pady=(0, 4))
        ttk.Label(wrap, text="Radiation-free simulated fluoroscopy for pedicle-screw training",
                  style="Muted.TLabel").pack(pady=(0, 24))

        # status gates
        gates = ttk.Frame(wrap, style="App.TFrame")
        gates.pack(pady=(0, 24))
        self.chip_cam   = W.StatusChip(gates, "Cameras")
        self.chip_model = W.StatusChip(gates, "Model")
        self.chip_cam.grid(row=0, column=0, padx=18, sticky="w")
        self.chip_model.grid(row=0, column=1, padx=18, sticky="w")

        # navigation buttons
        btns = ttk.Frame(wrap, style="App.TFrame")
        btns.pack()
        ttk.Button(btns, text="1.  Cameras  —  calibrate & verify",
                   width=34, command=lambda: self.app.show("cameras")).pack(pady=6)
        ttk.Button(btns, text="2.  Model registration  —  X-ray fiducials",
                   width=34, command=lambda: self.app.show("model")).pack(pady=6)
        self.btn_sim = ttk.Button(btns, text="3.  Start simulation",
                                  width=34, style="Accent.TButton",
                                  command=lambda: self.app.show("sim"))
        self.btn_sim.pack(pady=(14, 6))

        self.hint = ttk.Label(wrap, text="", style="Muted.TLabel")
        self.hint.pack(pady=(10, 0))

    def on_show(self):
        s = self.app.session
        self.chip_cam.set_state(
            s.cameras_calibrated,
            "Cameras  —  " + ("calibrated" if s.cameras_calibrated else "not calibrated"))
        self.chip_model.set_state(
            s.model_ready,
            "Model  —  " + (s.model.name if s.model_ready else "none loaded"))

        if s.can_simulate:
            self.btn_sim.state(["!disabled"])
            verified = (s.last_hole_error_mm is not None
                        and s.last_hole_error_mm <= config.TIP_ERROR_TOLERANCE_MM)
            self.hint.configure(
                text="Ready." if verified else
                "Ready. Tip: run the calibration-hole check on the Cameras screen "
                "to confirm accuracy before a session.")
        else:
            self.btn_sim.state(["disabled"])
            need = []
            if not s.cameras_calibrated: need.append("calibrate both cameras")
            if not s.model_ready:        need.append("complete a model registration")
            self.hint.configure(text="To simulate: " + " and ".join(need) + ".")
