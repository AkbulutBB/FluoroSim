"""
diagnose_context.py — Find the correct gVXR context-creation call on Windows.

Run from the Anaconda Prompt in the FluoroSim folder:
    python diagnose_context.py

It prints the function signature, then tries several documented forms of
context creation and reports which one succeeds. Copy the line that prints
"*** THIS ONE WORKS ***" back to Claude.
"""

import inspect
from gvxrPython3 import gvxr

print("=" * 64)
print("gVXR context diagnostic")
print("=" * 64)
print("Core version:", gvxr.getVersionOfCoreGVXR())
print()

# ── 1. Show the true signature ───────────────────────────────────────────
print("[ Signature of createNewContext ]")
try:
    print("   ", inspect.signature(gvxr.createNewContext))
except (TypeError, ValueError):
    print("    (C++ binding — signature not introspectable; see help below)")
print()
print("[ help(createNewContext) — first lines ]")
try:
    doc = gvxr.createNewContext.__doc__ or ""
    for line in doc.splitlines()[:12]:
        print("   ", line)
except Exception as e:
    print("    (no docstring)", e)
print()

# ── 2. List all context-related functions available ─────────────────────
print("[ Available context/window functions ]")
for name in sorted(dir(gvxr)):
    low = name.lower()
    if any(k in low for k in ("context", "window", "opengl", "egl")) \
            and not name.startswith("_"):
        print("   ", name)
print()

# ── 3. Try several creation forms, report the first that works ───────────
print("[ Trying context-creation forms ]")

def try_form(description, fn):
    try:
        fn()
        # If we got here, it worked — verify renderer is real
        try:
            r = gvxr.getOpenGlRenderer()
        except Exception:
            r = "(unknown)"
        print(f"   OK   {description}")
        print(f"        renderer: {r}")
        print(f"        *** THIS ONE WORKS — copy this line to Claude ***")
        return True
    except Exception as e:
        msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        print(f"   fail {description}   ({msg})")
        return False
    finally:
        try:
            gvxr.destroyAllWindows()
        except Exception:
            pass

forms = [
    ('createNewContext("OPENGL")',
     lambda: gvxr.createNewContext("OPENGL")),

    ('createNewContext("OPENGL", -1, 0, 4, 5)',
     lambda: gvxr.createNewContext("OPENGL", -1, 0, 4, 5)),

    ('createNewContext("OPENGL", -1, 4, 5, False)',
     lambda: gvxr.createNewContext("OPENGL", -1, 4, 5, False)),

    ('createWindow(-1, 0, "OPENGL", 4, 5)',
     lambda: gvxr.createWindow(-1, 0, "OPENGL", 4, 5)
        if hasattr(gvxr, "createWindow") else (_ for _ in ()).throw(
            AttributeError("createWindow not present"))),

    ('createOpenGLContext()',
     lambda: gvxr.createOpenGLContext()
        if hasattr(gvxr, "createOpenGLContext") else (_ for _ in ()).throw(
            AttributeError("createOpenGLContext not present"))),

    ('createNewContext("OPENGL", -1, 0)',
     lambda: gvxr.createNewContext("OPENGL", -1, 0)),
]

any_worked = False
for desc, fn in forms:
    if try_form(desc, fn):
        any_worked = True
        break   # stop at first success

print()
if any_worked:
    print("SUCCESS — at least one form works. Send Claude the line marked ***.")
else:
    print("No form worked. Likely a GPU/driver OpenGL issue rather than an")
    print("argument problem. Send Claude this entire output.")
    print()
    print("Quick driver check: update your graphics driver, and if this is a")
    print("laptop with switchable graphics, ensure Python is using the discrete")
    print("GPU (NVIDIA Control Panel → Manage 3D settings → python.exe → High-")
    print("performance processor).")
