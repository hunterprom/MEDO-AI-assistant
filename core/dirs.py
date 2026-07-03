"""User directory shortcuts for the HUD sphere dots.

The web HUD maps dots on the orb to real folders: the centre dot is the system
drive, the surrounding dots are the standard user folders. Hovering a dot shows
its path; clicking it opens the folder in the OS file explorer via the companion
API's ``POST /open`` (see remote/server.py).

Kept tiny and dependency-light (stdlib only) so the shortcut list is computed the
same way on the server and is easy to unit-test. Only folders that actually exist
on this machine are offered, and ``resolve()`` is the *only* way a key becomes a
path — the API never opens an arbitrary caller-supplied path.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _system_root() -> tuple[str, Path]:
    """(label, path) of the 'centre' dot — the system drive on Windows, / elsewhere."""
    if os.name == "nt":
        drive = os.environ.get("SystemDrive", "C:")
        return drive.rstrip("\\") + "\\", Path(drive + "\\")
    return "/", Path("/")


def user_dirs() -> list[dict]:
    """Ordered folder shortcuts as ``{"key", "label", "path", "center"}`` dicts.

    Exactly one entry (the system drive/root) has ``center=True``; the HUD binds
    it to the big middle dot. Non-existent folders are dropped so no dead dots
    appear. Paths are absolute strings, ready for a tooltip.
    """
    home = Path.home()
    root_label, root_path = _system_root()
    out: list[dict] = [
        {"key": "root", "label": root_label, "path": str(root_path), "center": True},
    ]
    candidates = [
        ("home", "HOME", home),
        ("desktop", "DESKTOP", home / "Desktop"),
        ("downloads", "DOWNLOADS", home / "Downloads"),
        ("documents", "DOCUMENTS", home / "Documents"),
        ("pictures", "PICTURES", home / "Pictures"),
        ("music", "MUSIC", home / "Music"),
        ("videos", "VIDEOS", home / "Videos"),
    ]
    for key, label, path in candidates:
        if path.exists():
            out.append({"key": key, "label": label, "path": str(path), "center": False})
    return out


def resolve(key: str) -> Path | None:
    """Absolute path for a shortcut key, or ``None`` if unknown/missing.

    The whitelist gate: a client can only open folders that appear in
    :func:`user_dirs`, never an arbitrary path.
    """
    for entry in user_dirs():
        if entry["key"] == key:
            return Path(entry["path"])
    return None


def open_path(path: Path) -> bool:
    """Open ``path`` in the OS file explorer. Returns True on a clean launch.

    Windows uses ``explorer`` (``os.startfile`` also works but returns nothing to
    check); macOS uses ``open``; Linux uses ``xdg-open``. Never raises.
    """
    try:
        if not path.exists():
            return False
        if sys.platform.startswith("win"):
            # explorer returns 1 even on success for some paths, so trust the
            # spawn rather than the exit code.
            subprocess.Popen(["explorer", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return True
    except Exception:
        logger.exception("could not open %s in file explorer", path)
        return False
