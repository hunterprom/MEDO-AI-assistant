# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the MEDO engine (main.py). Build from the repo root:
#   pyinstaller winsetup/medo-engine.spec
#
# Bundles the engine + HUD assets + small local models (wakeword, confirm-word
# banks). The LLM and whisper models are NOT bundled — the first-run wizard pulls
# them (keeps the installer small). The vision sidecar is a SEPARATE bundle
# (medo-vision.spec) because mediapipe pins numpy<2.

import os

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

ICON = os.path.join(SPECPATH, "medo.ico")   # the MEDO logo, embedded in the exe

hiddenimports = []
# faster-whisper + onnxruntime + ctranslate2 hide their backends from the analyzer.
hiddenimports += collect_submodules("onnxruntime")
hiddenimports += collect_submodules("faster_whisper")
hiddenimports += ["sounddevice", "_sounddevice_data"]

datas = []
# The web HUD (served by aiohttp) and its static assets.
datas += [("ui", "ui")]
# Yes/no banks + any language data the router loads at startup.
datas += [("lang", "lang")]
# Small on-device models shipped with the app (wakeword; piper voice is ~60MB and
# may instead be a first-run download — see docs/Packaging.md).
datas += [("models/wakeword", "models/wakeword")]
# onnxruntime / ctranslate2 native libs.
datas += collect_data_files("onnxruntime")

a = Analysis(
    ["../main.py"],
    pathex=[".."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["mediapipe", "cv2"],   # vision lives in its OWN bundle
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="medo-engine",
          console=False, disable_windowed_traceback=True, icon=ICON)
coll = COLLECT(exe, a.binaries, a.datas, name="medo-engine")
