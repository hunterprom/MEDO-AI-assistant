"""Tiny cross-platform helpers.

MEDO targets Windows but is developed on macOS, so every OS-specific action goes
through here (or a per-platform table in config.yaml) rather than scattering
``sys.platform`` checks across the skills.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


def current_os() -> str:
    """Return ``"windows"``, ``"darwin"`` (macOS), or ``"linux"``."""
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


IS_WINDOWS = current_os() == "windows"
IS_MACOS = current_os() == "darwin"


def pick_for_os(table: dict[str, str]) -> str | None:
    """Select the value for the current OS from a ``{os: command}`` table."""
    return table.get(current_os())


def run_detached(command: str) -> None:
    """Launch a shell command without waiting for it (e.g. opening an app).

    On Windows the command is run through the shell so builtins like ``start``
    work; elsewhere it's tokenized and run directly.
    """
    if IS_WINDOWS:
        subprocess.Popen(command, shell=True)
    else:
        subprocess.Popen(shlex.split(command))


def open_path(path: Path) -> None:
    """Open a file or folder with the OS default handler."""
    if IS_WINDOWS:
        import os

        os.startfile(str(path))  # type: ignore[attr-defined]  # Windows-only
    elif IS_MACOS:
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
