"""
ui/app.py — Application state and view state machine.

FluoroSimApp owns the Tkinter root window, swaps views in and out of a
single container frame, and holds the shared AppState.

State machine
─────────────
CAMERA_ASSIGN  → INTRINSIC_CALIB  → MODEL_SELECT  → NAVIGATION
                                                   ↑
                                              OR_SETUP (accessible any time
                                                        via navigation toolbar)

There is no MODEL_CALIB step.  Camera-to-model registration is handled
automatically by the PlatformBoardTracker on every frame.
"""

import tkinter as tk
from tkinter import messagebox
import threading
from typing import Optional

import numpy as np

from core.camera       import CameraCapture, load_intrinsics, list_available_cameras
from core.tracker      import ArucoTracker
from core.board_tracker import PlatformBoardTracker
from core.projection   import XRayOverlay
from core.model_config import ModelPackage
from config            import APP_TITLE, APP_VERSION


# ── Shared application state ───────────────────────────────────────────────────

class AppState:
    """
    Mutable state shared between all views.

    Camera 1 = cranial (straight down).
    Camera 2 = cranial oblique (45°).
    Both cameras are top-mounted on the same cranial frame.
    """

    def __init__(self):
        # Camera indices (assigned in CameraAssignView)
        self.cam1_idx: Optional[int] = None   # cranial
        self.cam2_idx: Optional[int] = None   # oblique 45°

        # Active capture objects
        self.cap1: Optional[CameraCapture] = None
        self.cap2: Optional[CameraCapture] = None

        # Intrinsic calibration
        self.mtx1:  Optional[np.ndarray] = None
        self.dist1: Optional[np.ndarray] = None
        self.mtx2:  Optional[np.ndarray] = None
        self.dist2: Optional[np.ndarray] = None

        # Active spine model
        self.model: Optional[ModelPackage] = None

        # Trackers — created once, reused across sessions
        self.probe_tracker: ArucoTracker         = ArucoTracker()
        self.board_tracker: PlatformBoardTracker = PlatformBoardTracker()

        # X-ray overlay renderers (built after OR Setup)
        self.overlay_ap:  Optional[XRayOverlay] = None
        self.overlay_lat: Optional[XRayOverlay] = None

    def stop_cameras(self):
        """Release all camera captures safely (handles shared-index case)."""
        stopped = set()
        for cap in (self.cap1, self.cap2):
            if cap is not None and id(cap) not in stopped:
                cap.stop()
                stopped.add(id(cap))
        self.cap1 = None
        self.cap2 = None

    def build_overlays(self):
        """Construct XRayOverlay objects from the loaded model's X-rays and projection matrices."""
        if self.model and self.model.has_xrays and self.model.has_projection:
            self.overlay_ap  = XRayOverlay(self.model.xray_ap,  self.model.P_ap)
            self.overlay_lat = XRayOverlay(self.model.xray_lat, self.model.P_lat)


# ── Main application window ────────────────────────────────────────────────────

class FluoroSimApp:
    """
    Hosts the Tkinter window and manages transitions between views.

    Views are DarkFrame subclasses that get swapped into self._container.
    Only one view is visible at a time.
    """

    def __init__(self, root: tk.Tk):
        self.root  = root
        self.state = AppState()

        root.title(f"{APP_TITLE}  v{APP_VERSION}")
        root.configure(bg="#0d1117")
        root.geometry("1280x820")
        root.minsize(1024, 700)

        self._container  = tk.Frame(root, bg="#0d1117")
        self._container.pack(fill=tk.BOTH, expand=True)

        self._views: dict[str, object] = {}
        self._current_view: Optional[str] = None

        self._register_views()
        self.show_view("camera_assign")

    # ── View registry ────────────────────────────────────────────────────────

    def _register_views(self):
        # Import here to avoid circular imports at module level
        from ui.camera_assign  import CameraAssignView
        from ui.intrinsic_calib import IntrinsicCalibView
        from ui.model_select   import ModelSelectView
        from ui.navigation     import NavigationView
        from ui.or_setup       import ORSetupView

        for key, cls in (
            ("camera_assign",   CameraAssignView),
            ("intrinsic_calib", IntrinsicCalibView),
            ("model_select",    ModelSelectView),
            ("navigation",      NavigationView),
            ("or_setup",        ORSetupView),
        ):
            view = cls(self._container, app=self, state=self.state)
            view.place(relx=0, rely=0, relwidth=1, relheight=1)
            self._views[key] = view

    def show_view(self, name: str, **kwargs):
        """Swap to the named view, calling on_hide / on_show hooks."""
        if self._current_view:
            self._views[self._current_view].on_hide()

        view = self._views[name]
        view.lift()
        view.on_show(**kwargs)
        self._current_view = name

    # ── Transition callbacks (called by individual views) ────────────────────

    def proceed_after_camera_assign(self):
        """Open cameras and advance to intrinsic calibration or model select."""
        ap_idx  = self.state.cam1_idx
        lat_idx = self.state.cam2_idx

        loading = self._show_loading("Opening cameras…")

        def _open():
            errors = []
            cap1 = CameraCapture(ap_idx)
            if not cap1.start():
                errors.append(f"Cannot open camera {ap_idx} (cranial).")
                self.root.after(0, lambda: _done(None, None, errors))
                return

            cap2 = CameraCapture(lat_idx) if lat_idx != ap_idx else cap1
            if cap2 is not cap1 and not cap2.start():
                cap1.stop()
                errors.append(f"Cannot open camera {lat_idx} (oblique).")
                self.root.after(0, lambda: _done(None, None, errors))
                return

            self.root.after(0, lambda: _done(cap1, cap2, errors))

        def _done(cap1, cap2, errors):
            self._hide_loading(loading)
            if errors:
                messagebox.showerror("Camera Error", "\n".join(errors))
                return

            self.state.cap1 = cap1
            self.state.cap2 = cap2

            # Try to load saved intrinsics for both cameras
            for idx, attr in ((ap_idx, "1"), (lat_idx, "2")):
                saved = load_intrinsics(str(idx))
                if saved:
                    setattr(self.state, f"mtx{attr}",  saved[0])
                    setattr(self.state, f"dist{attr}", saved[1])

            if self.state.mtx1 is not None and self.state.mtx2 is not None:
                self.show_view("model_select")
            else:
                self.show_view("intrinsic_calib")

        threading.Thread(target=_open, daemon=True).start()

    def proceed_after_intrinsic_calib(self):
        self.show_view("model_select")

    def proceed_after_model_select(self):
        self.state.build_overlays()
        self.show_view("navigation")

    def go_to_or_setup(self):
        self.show_view("or_setup")

    def go_to_navigation(self):
        self.state.build_overlays()
        self.show_view("navigation")

    def go_home(self):
        self.show_view("camera_assign")

    # ── Loading overlay ──────────────────────────────────────────────────────

    def _show_loading(self, message: str) -> tk.Toplevel:
        w = tk.Toplevel(self.root)
        w.title("")
        w.geometry("320x90")
        w.resizable(False, False)
        w.configure(bg="#161b22")
        w.transient(self.root)
        w.grab_set()
        tk.Label(w, text=message, font=("Segoe UI", 10),
                 fg="#e6edf3", bg="#161b22").pack(expand=True, pady=24)
        self.root.update()
        return w

    def _hide_loading(self, w: tk.Toplevel):
        try:
            w.grab_release()
            w.destroy()
        except Exception:
            pass
        self.root.update_idletasks()
