# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the MEDO launcher/tray (app/launcher.py) — the exe the
# Start-menu/desktop shortcut runs. It supervises medo-engine + medo-vision and
# ensures Ollama. Small bundle (tray + http probes). Build from the repo root:
#   pyinstaller winsetup/medo-launcher.spec

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("pystray") + ["PIL.Image", "PIL.ImageDraw",
                                                 "httpx"]

a = Analysis(
    ["../app/launcher.py"],
    pathex=[".."],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    excludes=["mediapipe", "cv2", "torch", "faster_whisper"],
    noarchive=False,
)
pyz = PYZ(a.pure)
# name MEDO.exe -> this is the user-facing app the shortcut points at.
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="MEDO",
          console=False, disable_windowed_traceback=True)
coll = COLLECT(exe, a.binaries, a.datas, name="MEDO")
