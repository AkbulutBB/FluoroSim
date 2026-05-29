"""
ui/model_select.py — Step 3: Select or create a spine model package.

Lists all available model packages and lets the user select one.
A "New model" button opens a simple dialog to create a new package entry.
After selection the user proceeds directly to navigation — no calibration
step is needed because the board handles registration automatically.
"""

import tkinter as tk
from tkinter import messagebox, simpledialog

from ui.widgets import (
    DarkFrame, primary_btn, success_btn, danger_btn,
    info_label, section_label,
    BG, BG2, BG3, FG, FG_MUTED, FG_SUCCESS, FG_ERR, FG_WARN,
    FONT_BODY, FONT_LABEL, FONT_TITLE,
)
from core.model_config import list_models, load_model, create_model


class ModelSelectView(DarkFrame):

    def __init__(self, parent, app, state, **kwargs):
        super().__init__(parent, **kwargs)
        self._app   = app
        self._state = state
        self._build()

    # ── Build ────────────────────────────────────────────────────────────────

    def _build(self):
        hdr = tk.Frame(self, bg=BG2, pady=10)
        hdr.pack(fill=tk.X)
        section_label(hdr, "Step 3 — Select Spine Model").pack(pady=6)
        info_label(
            hdr,
            "Select the spine model currently mounted on the training platform.  "
            "If this model has not yet had its X-rays set up, run OR Setup after selecting it.  "
            "Camera-to-model registration is automatic — no slot calibration required.",
            color=FG_MUTED,
        ).pack(padx=20, pady=(0, 6))

        # Model list
        list_frame = tk.Frame(self, bg=BG2, padx=20, pady=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=10)

        tk.Label(list_frame, text="Available models",
                 font=FONT_LABEL, fg=FG_MUTED, bg=BG2).pack(anchor=tk.W)

        self._listbox = tk.Listbox(
            list_frame,
            bg=BG3, fg=FG, selectbackground="#1f6feb",
            font=FONT_BODY, relief=tk.FLAT, height=12,
            activestyle="none",
        )
        self._listbox.pack(fill=tk.BOTH, expand=True, pady=4)
        self._listbox.bind("<<ListboxSelect>>", self._on_select)
        self._listbox.bind("<Double-Button-1>", lambda e: self._proceed())

        # Status / description
        self._desc_var = tk.StringVar(value="")
        tk.Label(list_frame, textvariable=self._desc_var,
                 font=FONT_LABEL, fg=FG_MUTED, bg=BG2, wraplength=600).pack(pady=4)

        # Buttons
        btns = tk.Frame(self, bg=BG)
        btns.pack(pady=12)
        primary_btn(btns, "↩ Back",       command=self._app.go_home, width=12).pack(side=tk.LEFT, padx=6)
        primary_btn(btns, "+ New model",  command=self._new_model,   width=14).pack(side=tk.LEFT, padx=6)
        primary_btn(btns, "OR Setup",     command=self._or_setup,    width=14).pack(side=tk.LEFT, padx=6)
        self._next_btn = success_btn(btns, "Start Training →",
                                     command=self._proceed, width=18)
        self._next_btn.configure(state=tk.DISABLED)
        self._next_btn.pack(side=tk.LEFT, padx=6)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_show(self, **kwargs):
        self._refresh_list()

    # ── List management ──────────────────────────────────────────────────────

    def _refresh_list(self):
        self._listbox.delete(0, tk.END)
        self._models = list_models()
        for m in self._models:
            try:
                pkg = load_model(m)
                status = "✓ ready" if pkg.is_ready else "⚠ needs OR Setup"
                self._listbox.insert(tk.END, f"  {pkg.name}   [{status}]")
            except Exception:
                self._listbox.insert(tk.END, f"  {m}   [error loading]")
        self._next_btn.configure(state=tk.DISABLED)

    def _on_select(self, _event=None):
        sel = self._listbox.curselection()
        if not sel:
            return
        model_id = self._models[sel[0]]
        try:
            pkg = load_model(model_id)
            self._state.model = pkg
            status = "Ready for training." if pkg.is_ready else \
                     "X-rays not yet configured — run OR Setup before training."
            self._desc_var.set(f"{pkg.name}: {status}")
            self._next_btn.configure(
                state=tk.NORMAL if pkg.is_ready else tk.DISABLED
            )
        except Exception as e:
            self._desc_var.set(f"Error loading model: {e}")

    # ── Actions ──────────────────────────────────────────────────────────────

    def _new_model(self):
        name = simpledialog.askstring("New Model", "Enter a name for the new spine model:",
                                      parent=self)
        if not name or not name.strip():
            return
        model_id = name.strip().lower().replace(" ", "_")
        try:
            create_model(model_id, name.strip())
        except Exception as e:
            messagebox.showerror("Error", f"Could not create model: {e}")
            return
        self._refresh_list()
        messagebox.showinfo("Model Created",
                            f"Model '{name}' created.\n\nRun OR Setup to add X-rays.")

    def _or_setup(self):
        if self._state.model is None:
            messagebox.showwarning("No Model", "Select a model first.")
            return
        self._app.go_to_or_setup()

    def _proceed(self):
        if self._state.model is None:
            messagebox.showwarning("No Model", "Please select a model.")
            return
        if not self._state.model.is_ready:
            messagebox.showwarning(
                "Not Ready",
                "This model needs OR Setup before training.\n"
                "Click OR Setup to configure the X-ray projection."
            )
            return
        self._app.proceed_after_model_select()
