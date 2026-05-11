"""
ui/model_select.py — Step 3: Choose a spine model package.
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import shutil
from pathlib import Path

from ui.widgets import (
    DarkFrame, primary_btn, success_btn, section_label, info_label,
    BG, BG2, BG3, FG, FG_MUTED, FG_SUCCESS, FG_ERR, FONT_BODY, FONT_LABEL,
)
from core.model_config import load_model, list_models, ModelPackage
from config import MODELS_DIR


class ModelSelectView(DarkFrame):

    def __init__(self, parent, app, state, **kwargs):
        super().__init__(parent, **kwargs)
        self._app   = app
        self._state = state
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=BG2)
        hdr.pack(fill=tk.X, pady=(0, 8))
        section_label(hdr, "Step 3 — Select Spine Model").pack(pady=10)
        info_label(
            hdr,
            "Choose the model package that matches your 3D-printed spine. "
            "Each package contains pre-acquired X-ray images and projection data "
            "from the OR setup visit.",
            color=FG_MUTED,
        ).pack(pady=(0, 8))

        # Model list
        list_frame = tk.Frame(self, bg=BG, padx=20)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self._listbox_var = tk.StringVar()
        self._listbox = tk.Listbox(
            list_frame,
            listvariable=self._listbox_var,
            font=FONT_BODY, fg=FG, bg=BG3,
            selectbackground="#0f3460", selectforeground=FG,
            relief=tk.FLAT, height=10, width=40,
        )
        self._listbox.pack(side=tk.LEFT, pady=8)
        self._listbox.bind("<<ListboxSelect>>", self._on_select)

        # Info panel
        info_panel = tk.Frame(list_frame, bg=BG2, padx=14, pady=10)
        info_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12)

        self._info_name  = tk.Label(info_panel, text="", font=("Segoe UI", 11, "bold"),
                                     fg=FG, bg=BG2)
        self._info_name.pack(anchor=tk.W)
        self._info_slots = tk.Label(info_panel, text="", font=FONT_LABEL, fg=FG_MUTED, bg=BG2)
        self._info_slots.pack(anchor=tk.W, pady=2)
        self._info_xray  = tk.Label(info_panel, text="", font=FONT_LABEL, fg=FG_MUTED, bg=BG2)
        self._info_xray.pack(anchor=tk.W, pady=2)
        self._info_proj  = tk.Label(info_panel, text="", font=FONT_LABEL, fg=FG_MUTED, bg=BG2)
        self._info_proj.pack(anchor=tk.W, pady=2)

        # Buttons
        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(pady=10)
        primary_btn(btn_row, "↻ Refresh", command=self._refresh, width=12).pack(side=tk.LEFT, padx=6)
        self._next_btn = success_btn(btn_row, "Load model →", command=self._load, width=14)
        self._next_btn.configure(state=tk.DISABLED)
        self._next_btn.pack(side=tk.LEFT, padx=6)

        self._selected_id: str | None = None
        self._refresh()

    def on_show(self, **kwargs):
        self._refresh()

    def _refresh(self):
        models = list_models()
        self._listbox.delete(0, tk.END)
        for m in models:
            self._listbox.insert(tk.END, m)
        if not models:
            self._listbox.insert(tk.END, "(no models found — run OR Setup first)")

    def _on_select(self, event):
        sel = self._listbox.curselection()
        if not sel:
            return
        model_id = self._listbox.get(sel[0])
        if model_id.startswith("("):
            return
        self._selected_id = model_id
        try:
            pkg = load_model(model_id)
            self._info_name.configure(text=pkg.name)
            self._info_slots.configure(
                text=f"Calibration slots: {len(pkg.slots)}"
            )
            self._info_xray.configure(
                text=f"X-ray images: {'Yes' if pkg.has_xrays else 'Missing — run OR Setup'}",
                fg=FG_SUCCESS if pkg.has_xrays else FG_ERR,
            )
            self._info_proj.configure(
                text=f"Projection matrices: {'Yes' if pkg.has_projection else 'Missing — run OR Setup'}",
                fg=FG_SUCCESS if pkg.has_projection else FG_ERR,
            )
            self._next_btn.configure(state=tk.NORMAL if pkg.has_xrays else tk.DISABLED)
        except Exception as exc:
            self._info_name.configure(text=str(exc))
            self._next_btn.configure(state=tk.DISABLED)

    def _load(self):
        if not self._selected_id:
            return
        try:
            pkg = load_model(self._selected_id)
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            return
        self._state.model = pkg
        self._app.proceed_after_model_select()
