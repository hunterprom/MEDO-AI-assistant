"""Build the three PyInstaller bundles + the Inno Setup installer.

Run from the repo root:  python winsetup/build.py
Requires: PyInstaller in BOTH venvs, and Inno Setup's `iscc` on PATH.

This orchestrates the build; it does not itself get unit-tested (it shells out to
real tools). The KEY subtlety it encodes: the vision sidecar is frozen with the
VISION venv's PyInstaller (numpy<2), everything else with the main venv's.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "packaging"


def _pyinstaller(venv: str, spec: str) -> None:
    exe = ROOT / venv / ("Scripts/pyinstaller.exe" if os.name == "nt"
                         else "bin/pyinstaller")
    print(f"[build] {venv} -> {spec}")
    subprocess.run([str(exe), "--noconfirm", str(PKG / spec)], cwd=ROOT, check=True)


def main() -> int:
    # Engine + launcher in the MAIN venv; vision in the VISION venv (numpy<2).
    _pyinstaller(".venv", "medo-engine.spec")
    _pyinstaller(".venv", "medo-launcher.spec")
    _pyinstaller(".venv-vision", "medo-vision.spec")

    iscc = "iscc"
    print("[build] installer -> iscc winsetup/installer.iss")
    try:
        subprocess.run([iscc, str(PKG / "installer.iss")], cwd=ROOT, check=True)
    except FileNotFoundError:
        print("[build] Inno Setup 'iscc' not found on PATH — install it and re-run,"
              " or open winsetup/installer.iss in the Inno Setup IDE.")
        return 1
    print("[build] done — see dist\\MEDO-Setup.exe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
