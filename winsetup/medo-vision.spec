# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the MEDO vision sidecar (vision/run.py). Build from repo
# root, IN THE VISION VENV (numpy<2), separately from the engine:
#   .venv-vision\Scripts\pyinstaller winsetup/medo-vision.spec
#
# WHY SEPARATE: mediapipe pins numpy<2 while the engine uses newer numpy — they
# cannot coexist in one environment or one bundle. This is the known packaging
# fight; keeping vision its own exe is the whole reason it's a separate process.
#
# mediapipe hides its .binarypb graph/model files and native modules from the
# analyzer, so collect them explicitly or the frozen sidecar fails at runtime
# with "graph not found".

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules("mediapipe")
datas = collect_data_files("mediapipe", include_py_files=True)  # incl. .binarypb
datas += collect_data_files("cv2")

a = Analysis(
    ["../vision/run.py"],
    pathex=[".."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["torch", "onnxruntime"],   # not needed in the vision bundle
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="medo-vision",
          console=False, disable_windowed_traceback=True)
coll = COLLECT(exe, a.binaries, a.datas, name="medo-vision")
