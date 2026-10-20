"""Single source of truth for the packaged app's artifacts + shortcuts.

The PyInstaller specs, the Inno Setup script, and the build script all agree with
this, and a CI-ish test (tests/test_packaging_manifest.py) enforces that they do —
so a rename can't silently desync the installer from the built exes.
"""

APP_NAME = "MEDO"
PUBLISHER = "MEDO"
INSTALL_DIR_NAME = "MEDO"                 # -> C:\Program Files\MEDO
SHORTCUT_NAME = "MEDO"                     # Start-menu + desktop

#: The three frozen executables. The LAUNCHER is what the shortcut runs; it
#: supervises the other two (and ensures Ollama). The vision exe is a SEPARATE
#: bundle on purpose (mediapipe pins numpy<2 — it cannot share the engine's env).
EXES = {
    "launcher": "MEDO.exe",
    "engine": "medo-engine.exe",
    "vision": "medo-vision.exe",
}

SPECS = {
    "launcher": "medo-launcher.spec",
    "engine": "medo-engine.spec",
    "vision": "medo-vision.spec",
}

#: What the wizard downloads on first run (NOT shipped in the installer, to keep
#: it small): the Ollama models per hardware profile + the whisper model.
FIRST_RUN_DOWNLOADS = "ollama models (per hardware profile) + the whisper model"
