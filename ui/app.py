"""
ui/app.py — Main application window and state machine.

States:
    CAMERA_ASSIGN    → user assigns AP and LAT cameras
    INTRINSIC_CALIB  → checkerboard calibration per camera
    MODEL_SELECT     → choose a spine model package
    MODEL_CALIB      → two-slot probe registration
    NAVIGATION       → live fluoroscopy simulation
    OR_SETUP         → one-time OR-visit fiducial setup (accessible any time)
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

import numpy as np

from core.camera      import CameraCapture, list_available_cameras, load_intrinsics
from core.tracker     import ArucoTracker
from core.calibration import CalibrationEngine, SlotDefinition, save_cam_model_transform, load_cam_model_transform
from core.projection  import XRayOverlay
from core.model_config import ModelPackage

from config import APP_TITLE, APP_VERSION, CAMERAS_DIR


# ── Application state keys ─────────────────────────────────────────────────────

class AppState:
    """Shared mutable state passed between views."""

    def __init__(self):
        # Camera assignment
        self.cam_ap_idx : Optional[int] = None
        self.cam_lat_idx: Optional[int] = None

        # Active capture objects
        self.cap_ap : Optional[CameraCapture] = None
        self.cap_lat: Optional[CameraCapture] = None

        # Intrinsics
        self.mtx_ap : Optional[np.ndarray] = None
        self.dist_ap: Optional[np.ndarray] = None
        self.mtx_lat : Optional[np.ndarray] = None
        self.dist_lat: Optional[np.ndarray] = None

        # Model
        self.model  : Optional[ModelPackage] = None

        # Calibration engines (created after model is selected)
        self.cal_engine_ap : Optional[CalibrationEngine] = None
        self.cal_engine_lat: Optional[CalibrationEngine] = None

        # Camera-to-model transforms
        self.xfm_ap : Optional[object] = None   # CameraModelTransform
        self.xfm_lat: Optional[object] = None

        # Overlay renderers
        self.overlay_ap : Optional[XRayOverlay] = None
        self.overlay_lat: Optional[XRayOverlay] = None

        # Tracker (shared between views)
        self.tracker = ArucoTracker()

    def stop_cameras(self):
        stopped = set()
        for cap in (self.cap_ap, self.cap_lat):
            if cap is not None and id(cap) not in stopped:
                cap.stop()
                stopped.add(id(cap))


# ── Main application ───────────────────────────────────────────────────────────

class FluoroSimApp:
    """
    Hosts the Tkinter window, manages view transitions, and owns shared state.
    Views are Frames that get swapped into self._container.
    """

    VIEWS = (
        "camera_assign",
        "intrinsic_calib",
        "model_select",
        "model_calib",
        "navigation",
        "or_setup",
    )

    def __init__(self, root: tk.Tk):
        self.root  = root
        self.state = AppState()
        self._current_view = None
        self._view_instances: dict[str, tk.Frame] = {}

        self._build_layout()
        self.show_view("camera_assign")

    # ── Layout ─────────────────────────────────────────────────────────────

    def _build_layout(self):
        self.root.configure(bg="#1a1a2e")

        # Top bar
        top = tk.Frame(self.root, bg="#16213e", height=48)
        top.pack(fill=tk.X, side=tk.TOP)
        top.pack_propagate(False)

        tk.Label(
            top, text=f"  {APP_TITLE}  v{APP_VERSION}",
            font=("Segoe UI", 11, "bold"), fg="#e0e0e0", bg="#16213e",
        ).pack(side=tk.LEFT, padx=8)

        # OR Setup button always accessible
        tk.Button(
            top, text="OR Setup",
            font=("Segoe UI", 9), fg="#ccc", bg="#0f3460",
            relief=tk.FLAT, padx=10,
            command=lambda: self.show_view("or_setup"),
        ).pack(side=tk.RIGHT, padx=8, pady=8)

        # Home button — returns to Step 1 from anywhere
        tk.Button(
            top, text="← Home",
            font=("Segoe UI", 9), fg="#ccc", bg="#0f3460",
            relief=tk.FLAT, padx=10,
            command=lambda: self.show_view("camera_assign"),
        ).pack(side=tk.RIGHT, padx=4, pady=8)

        # Step indicator
        self._step_var = tk.StringVar(value="")
        tk.Label(
            top, textvariable=self._step_var,
            font=("Segoe UI", 9), fg="#a0a0c0", bg="#16213e",
        ).pack(side=tk.RIGHT, padx=16)

        # Main content area
        self._container = tk.Frame(self.root, bg="#1a1a2e")
        self._container.pack(fill=tk.BOTH, expand=True)

    # ── View management ────────────────────────────────────────────────────

    def show_view(self, name: str, **kwargs):
        """Switch to the named view, instantiating it if necessary."""
        assert name in self.VIEWS, f"Unknown view: {name}"
        print(f"[DEBUG] show_view → {name}")
        try:
            if self._current_view:
                self._current_view.pack_forget()
                if hasattr(self._current_view, "on_hide"):
                    self._current_view.on_hide()

            if name not in self._view_instances:
                print(f"[DEBUG]   Instantiating {name}…")
                view_cls = self._get_view_class(name)
                frame    = view_cls(self._container, app=self, state=self.state)
                self._view_instances[name] = frame
                print(f"[DEBUG]   {name} instantiated")

            view = self._view_instances[name]
            if hasattr(view, "on_show"):
                print(f"[DEBUG]   Calling on_show for {name}…")
                view.on_show(**kwargs)
                print(f"[DEBUG]   on_show complete for {name}")

            view.pack(fill=tk.BOTH, expand=True)
            self._current_view = view
            self._update_step_indicator(name)
            self.root.update_idletasks()  # force repaint
            print(f"[DEBUG] show_view complete → {name}")
        except Exception as exc:
            import traceback
            print(f"[DEBUG] ✘ show_view CRASHED on {name}:")
            traceback.print_exc()
            messagebox.showerror("View Error", f"Failed to show '{name}':\n\n{exc}")

    def _get_view_class(self, name: str):
        # Imported here to avoid circular imports
        from ui.camera_assign   import CameraAssignView
        from ui.intrinsic_calib import IntrinsicCalibView
        from ui.model_select    import ModelSelectView
        from ui.model_calib     import ModelCalibView
        from ui.navigation      import NavigationView
        from ui.or_setup        import ORSetupView

        return {
            "camera_assign"  : CameraAssignView,
            "intrinsic_calib": IntrinsicCalibView,
            "model_select"   : ModelSelectView,
            "model_calib"    : ModelCalibView,
            "navigation"     : NavigationView,
            "or_setup"       : ORSetupView,
        }[name]

    def _update_step_indicator(self, name: str):
        steps = {
            "camera_assign"  : "Step 1 — Assign cameras",
            "intrinsic_calib": "Step 2 — Intrinsic calibration",
            "model_select"   : "Step 3 — Select model",
            "model_calib"    : "Step 4 — Model calibration",
            "navigation"     : "Navigation",
            "or_setup"       : "OR Setup (one-time)",
        }
        self._step_var.set(steps.get(name, ""))

    # ── Navigation helpers ─────────────────────────────────────────────────

    def proceed_after_camera_assign(self):
        """Called by CameraAssignView once both cameras are labelled."""
        ap_idx  = self.state.cam_ap_idx
        lat_idx = self.state.cam_lat_idx

        # Show a loading overlay while cameras open (can take several seconds)
        loading = tk.Toplevel(self.root)
        loading.title("Opening cameras…")
        loading.geometry("320x100")
        loading.resizable(False, False)
        loading.configure(bg="#16213e")
        loading.transient(self.root)
        loading.grab_set()
        tk.Label(
            loading, text="Opening cameras, please wait…",
            font=("Segoe UI", 10), fg="#e0e0e0", bg="#16213e"
        ).pack(expand=True, pady=20)
        self.root.update()

        def _open_cameras():
            errors = []
            print(f"[DEBUG] Opening AP camera (index {ap_idx})…")
            cap_ap = CameraCapture(ap_idx)
            if not cap_ap.start():
                errors.append(f"Cannot open camera index {ap_idx} (AP).")
                print(f"[DEBUG] ✘ AP camera failed")
                self.root.after(0, lambda: _done(None, None, errors))
                return
            print(f"[DEBUG] ✔ AP camera open")

            if lat_idx == ap_idx:
                print(f"[DEBUG] LAT = AP (single camera mode)")
                cap_lat = cap_ap
            else:
                print(f"[DEBUG] Opening LAT camera (index {lat_idx})…")
                cap_lat = CameraCapture(lat_idx)
                if not cap_lat.start():
                    cap_ap.stop()
                    errors.append(f"Cannot open camera index {lat_idx} (LAT).")
                    print(f"[DEBUG] ✘ LAT camera failed")
                    self.root.after(0, lambda: _done(None, None, errors))
                    return
                print(f"[DEBUG] ✔ LAT camera open")

            print(f"[DEBUG] Both cameras ready — proceeding")
            self.root.after(0, lambda: _done(cap_ap, cap_lat, errors))

        def _done(cap_ap, cap_lat, errors):
            try:
                loading.grab_release()
                loading.destroy()
            except Exception:
                pass
            self.root.update_idletasks()

            if errors:
                messagebox.showerror("Camera Error", "\n".join(errors))
                return

            print(f"[DEBUG] _done called — assigning captures to state")
            self.state.cap_ap  = cap_ap
            self.state.cap_lat = cap_lat

            # Load saved intrinsics for each camera
            for idx, cam_id in ((ap_idx, "ap"), (lat_idx, "lat")):
                saved = load_intrinsics(str(idx))
                if saved:
                    print(f"[DEBUG] Loaded saved intrinsics for camera {idx}")
                    mtx, dist = saved
                    setattr(self.state, f"mtx_{cam_id}",  mtx)
                    setattr(self.state, f"dist_{cam_id}", dist)
                else:
                    print(f"[DEBUG] No saved intrinsics for camera {idx}")

            if self.state.mtx_ap is not None and self.state.mtx_lat is not None:
                print(f"[DEBUG] Intrinsics found — going to model_select")
                self.show_view("model_select")
            else:
                print(f"[DEBUG] No intrinsics — going to intrinsic_calib")
                self.show_view("intrinsic_calib")

        import threading
        threading.Thread(target=_open_cameras, daemon=True).start()

    def proceed_after_intrinsic_calib(self):
        self.show_view("model_select")

    def proceed_after_model_select(self):
        # Build calibration engines from slot definitions
        slots = self.state.model.slots
        slot_defs = slots  # already SlotDefinition instances
        self.state.cal_engine_ap  = CalibrationEngine(slot_defs)
        self.state.cal_engine_lat = CalibrationEngine(slot_defs)
        self.show_view("model_calib")

    def proceed_after_model_calib(self):
        model = self.state.model
        # Save transforms for this camera+model combination
        save_cam_model_transform(str(self.state.cam_ap_idx),  model.model_id, self.state.xfm_ap)
        save_cam_model_transform(str(self.state.cam_lat_idx), model.model_id, self.state.xfm_lat)
        self._build_overlays()
        self.show_view("navigation")

    def _build_overlays(self):
        model = self.state.model
        if model.P_ap is not None and model.xray_ap is not None:
            self.state.overlay_ap  = XRayOverlay(model.xray_ap,  model.P_ap)
            self.state.overlay_lat = XRayOverlay(model.xray_lat, model.P_lat)

    def go_to_navigation_direct(self, model_id: str):
        """
        Called from OR Setup's 'Start Training' button.
        Loads everything from disk and jumps straight to navigation
        if cameras, intrinsics, and model calibration are all available.
        """
        from core.model_config import load_model
        from core.camera import load_intrinsics
        from core.calibration import load_cam_model_transform

        # Load model
        try:
            self.state.model = load_model(model_id)
        except Exception as exc:
            messagebox.showerror("Model Error", str(exc))
            return

        # Check cameras assigned
        if self.state.cam_ap_idx is None or self.state.cam_lat_idx is None:
            messagebox.showinfo(
                "Cameras not assigned",
                "Please assign your cameras first via ← Home, then return here.",
            )
            return

        # Load intrinsics
        for cam_id_attr, mtx_attr, dist_attr in (
            (self.state.cam_ap_idx,  "mtx_ap",  "dist_ap"),
            (self.state.cam_lat_idx, "mtx_lat", "dist_lat"),
        ):
            saved = load_intrinsics(str(cam_id_attr))
            if saved is None:
                messagebox.showinfo(
                    "Intrinsics missing",
                    f"Camera {cam_id_attr} has not been calibrated yet.\n"
                    "Go through the full setup flow first.",
                )
                return
            mtx, dist = saved
            setattr(self.state, mtx_attr,  mtx)
            setattr(self.state, dist_attr, dist)

        # Load saved camera-model transforms
        xfm_ap  = load_cam_model_transform(str(self.state.cam_ap_idx),  model_id)
        xfm_lat = load_cam_model_transform(str(self.state.cam_lat_idx), model_id)

        if xfm_ap is None or xfm_lat is None:
            messagebox.showinfo(
                "Slot calibration required",
                "No saved slot calibration found for this camera+model combination.\n"
                "Click ← Home and complete Step 4 (model calibration) first.",
            )
            return

        self.state.xfm_ap  = xfm_ap
        self.state.xfm_lat = xfm_lat
        self._build_overlays()
        self.show_view("navigation")

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def on_close(self):
        self.state.stop_cameras()
        self.root.destroy()
