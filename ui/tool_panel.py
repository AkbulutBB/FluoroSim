"""
ui/tool_panel.py — Tool selector and tip-distance control for the navigation view.

Rendered as a compact sidebar panel.  The user can:
  • Choose between Standard Probe and Custom Tool (chisel / awl).
  • Edit the tip distance (mm) for the custom tool via a spinbox.
  • See a live description of the active tool.

Changes take effect immediately — config.set_active_tool() and
config.set_custom_tool_distance() update the global state read by
tracker.py on every detect() call, so no restart is required.
"""

import tkinter as tk
from tkinter import messagebox

from ui.widgets import (
    DarkFrame,
    BG, BG2, BG3, ACCENT, FG, FG_MUTED, FG_SUCCESS, FG_ERR, FG_WARN,
    FONT_BODY, FONT_LABEL,
)
import config


# ── Constants ──────────────────────────────────────────────────────────────────
_TIP_MIN_MM  =  10.0    # sanity lower bound
_TIP_MAX_MM  = 400.0    # sanity upper bound
_TIP_STEP_MM =   5.0    # spinbox increment


class ToolPanel(DarkFrame):
    """
    Compact tool-selector widget.  Drop into the navigation sidebar with:

        self._tool_panel = ToolPanel(sidebar, on_change=self._on_tool_changed)
        self._tool_panel.pack(fill=tk.X, padx=8, pady=6)
    """

    def __init__(self, parent, on_change=None, **kwargs):
        super().__init__(parent, bg=BG2, **kwargs)
        self._on_change = on_change   # optional callback(tool_key: str)
        self._build()
        # Initialise UI to whatever is already active in config
        active = config._active_tool_key
        self._tool_var.set(active)
        self._sync_to_key(active)

    # ── Build ──────────────────────────────────────────────────────────────

    def _build(self):
        # ── Section header ──────────────────────────────────────────────
        tk.Label(
            self, text="ACTIVE TOOL",
            font=("Segoe UI", 8, "bold"), fg=FG_MUTED, bg=BG2,
        ).pack(anchor=tk.W, padx=8, pady=(8, 2))

        # ── Tool selector radio buttons ──────────────────────────────────
        self._tool_var = tk.StringVar(value="standard_probe")

        radio_frame = tk.Frame(self, bg=BG2)
        radio_frame.pack(fill=tk.X, padx=8)

        for key, profile in config.TOOL_PROFILES.items():
            tk.Radiobutton(
                radio_frame,
                text=profile.name,
                variable=self._tool_var,
                value=key,
                command=self._on_radio,
                font=FONT_LABEL,
                fg=FG, bg=BG2,
                selectcolor=BG3,
                activebackground=BG2,
                activeforeground=FG,
            ).pack(anchor=tk.W, pady=1)

        # ── Tip distance control (custom tool only) ──────────────────────
        self._dist_frame = tk.Frame(self, bg=BG2)
        self._dist_frame.pack(fill=tk.X, padx=8, pady=(6, 2))

        tk.Label(
            self._dist_frame,
            text="Tip distance (face → tip, mm):",
            font=FONT_LABEL, fg=FG_MUTED, bg=BG2,
        ).pack(anchor=tk.W)

        spinbox_row = tk.Frame(self._dist_frame, bg=BG2)
        spinbox_row.pack(anchor=tk.W, pady=2)

        self._dist_var = tk.DoubleVar(
            value=config.TOOL_PROFILES["custom_tool"].tip_distance_mm
        )
        self._spinbox = tk.Spinbox(
            spinbox_row,
            from_=_TIP_MIN_MM,
            to=_TIP_MAX_MM,
            increment=_TIP_STEP_MM,
            textvariable=self._dist_var,
            width=7,
            font=FONT_BODY,
            fg=FG, bg=BG3,
            buttonbackground=BG3,
            insertbackground=FG,
            relief=tk.FLAT,
            command=self._on_spinbox,
        )
        self._spinbox.pack(side=tk.LEFT)

        tk.Label(
            spinbox_row, text="mm",
            font=FONT_LABEL, fg=FG_MUTED, bg=BG2,
        ).pack(side=tk.LEFT, padx=(4, 0))

        # Bind manual entry (Return key or focus-out)
        self._spinbox.bind("<Return>",    lambda _: self._on_spinbox())
        self._spinbox.bind("<FocusOut>",  lambda _: self._on_spinbox())

        # ── Description label ────────────────────────────────────────────
        self._desc_var = tk.StringVar()
        tk.Label(
            self,
            textvariable=self._desc_var,
            font=("Segoe UI", 8), fg=FG_MUTED, bg=BG2,
            wraplength=200, justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=8, pady=(4, 8))

    # ── Event handlers ─────────────────────────────────────────────────────

    def _on_radio(self):
        key = self._tool_var.get()
        self._sync_to_key(key)

    def _on_spinbox(self):
        """Apply the spinbox value to the custom tool profile."""
        try:
            val = float(self._dist_var.get())
        except (tk.TclError, ValueError):
            # Reset to current stored value if entry is garbage
            self._dist_var.set(config.TOOL_PROFILES["custom_tool"].tip_distance_mm)
            return

        if not (_TIP_MIN_MM <= val <= _TIP_MAX_MM):
            messagebox.showwarning(
                "Invalid distance",
                f"Tip distance must be between {_TIP_MIN_MM:.0f} and "
                f"{_TIP_MAX_MM:.0f} mm.",
            )
            self._dist_var.set(config.TOOL_PROFILES["custom_tool"].tip_distance_mm)
            return

        config.set_custom_tool_distance(val)
        self._refresh_desc()

        if self._on_change:
            self._on_change(config._active_tool_key)

    def _sync_to_key(self, key: str):
        """
        Update config active tool, show/hide distance control,
        and refresh the description label.
        """
        config.set_active_tool(key)

        # Show distance spinbox only for the custom tool
        if key == "custom_tool":
            self._dist_frame.pack(fill=tk.X, padx=8, pady=(6, 2))
            # Sync spinbox to stored value in case it was changed externally
            self._dist_var.set(config.TOOL_PROFILES["custom_tool"].tip_distance_mm)
        else:
            self._dist_frame.pack_forget()

        self._refresh_desc()

        if self._on_change:
            self._on_change(key)

    def _refresh_desc(self):
        tool = config.get_active_tool()
        self._desc_var.set(tool.description)

    # ── Public API ─────────────────────────────────────────────────────────

    def set_tool(self, key: str) -> None:
        """Programmatically select a tool (e.g. to restore a saved session)."""
        if key in config.TOOL_PROFILES:
            self._tool_var.set(key)
            self._sync_to_key(key)

    def get_tool_key(self) -> str:
        return self._tool_var.get()
