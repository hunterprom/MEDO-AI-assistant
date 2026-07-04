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


# Cache of the last sphere_dirs() result so /open can validate a clicked key
# against the exact set the HUD was given (populated when the page renders).
_SPHERE_INDEX: dict[str, str] = {}


def _scan_children(root: Path, out: list[dict], seen: set[str], limit: int) -> None:
    """Append immediate sub-directories of ``root`` to ``out`` (best-effort)."""
    if len(out) >= limit:
        return
    try:
        with os.scandir(root) as it:
            for entry in it:
                if len(out) >= limit:
                    return
                name = entry.name
                if name.startswith(".") or name.startswith("$"):
                    continue  # hidden / system junctions (e.g. $Recycle.Bin)
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                p = os.path.normpath(entry.path)
                if p in seen:
                    continue
                seen.add(p)
                out.append({"key": p, "label": name.upper()[:20], "path": p,
                            "center": False, "primary": False})
    except (PermissionError, OSError, NotADirectoryError):
        return


def sphere_dirs(limit: int = 300) -> list[dict]:
    """Up to ``limit`` real folders to scatter across the orb (hundreds on a PC).

    Starts from the curated :func:`user_dirs` (drive + standard user folders, the
    only ones with ``primary=True`` so the HUD labels just those), then a
    breadth-first descent through their sub-directories, level by level, until the
    cap. Extra dots carry their absolute path as ``key``. The result is cached so
    :func:`resolve` can whitelist exactly these paths for ``/open``.
    """
    curated = [{**d, "primary": True} for d in user_dirs()]
    out: list[dict] = list(curated)
    seen = {os.path.normpath(d["path"]) for d in out}

    # BFS: a queue of folders to expand, seeded with the curated set. Each pop
    # scans one folder's children (appending new dots) and enqueues them, so the
    # orb fills evenly outward instead of exhausting one branch first.
    queue: list[Path] = [Path(d["path"]) for d in curated]
    while queue and len(out) < limit:
        parent = queue.pop(0)
        before = len(out)
        _scan_children(parent, out, seen, limit)
        for entry in out[before:]:
            queue.append(Path(entry["path"]))

    out = out[:limit]
    _SPHERE_INDEX.clear()
    _SPHERE_INDEX.update({d["key"]: d["path"] for d in out})
    return out


def _search_roots() -> list[Path]:
    """Trees that file search walks — the user's folders (fast, and what they mean)."""
    home = Path.home()
    roots = [home]
    for sub in ("Desktop", "Downloads", "Documents", "Pictures", "Music", "Videos"):
        p = home / sub
        if p.exists():
            roots.append(p)
    return roots


def search_files(query: str, limit: int = 40, max_scanned: int = 60000) -> list[dict]:
    """Filename/-folder substring search under the user's folders.

    Bounded DFS (``max_scanned`` entries) so a huge tree can't hang the request;
    returns ``[{"name", "path", "is_dir"}]`` for names containing ``query``.
    Hidden/system entries are skipped. Never raises.
    """
    q = query.strip().lower()
    if not q:
        return []
    out: list[dict] = []
    scanned = 0
    stack: list[str] = [str(r) for r in _search_roots()]
    seen_dirs: set[str] = set()
    while stack and len(out) < limit and scanned < max_scanned:
        d = stack.pop()
        if d in seen_dirs:
            continue
        seen_dirs.add(d)
        try:
            with os.scandir(d) as it:
                for entry in it:
                    scanned += 1
                    name = entry.name
                    if name.startswith(".") or name.startswith("$"):
                        continue
                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        continue
                    if q in name.lower():
                        out.append({"name": name, "path": os.path.normpath(entry.path),
                                    "is_dir": is_dir})
                        if len(out) >= limit:
                            break
                    if is_dir:
                        stack.append(entry.path)
        except (PermissionError, OSError, NotADirectoryError):
            continue
    return out


def resolve_path(path_str: str) -> Path | None:
    """Validate a caller-supplied path for ``/open`` — else ``None``.

    Allowed only when the path exists AND lives under the user's home tree or was
    one of the sphere dots. Keeps ``/open`` from launching anything arbitrary even
    though the API is LAN-only.
    """
    try:
        rp = Path(path_str).resolve()
    except (OSError, ValueError):
        return None
    if not rp.exists():
        return None
    home = Path.home().resolve()
    if rp == home or home in rp.parents:
        return rp
    if os.path.normpath(str(rp)) in _SPHERE_INDEX:
        return rp
    return None


def resolve(key: str) -> Path | None:
    """Absolute path for a shortcut key, or ``None`` if not whitelisted/missing.

    The whitelist gate for ``/open``: a client may only open folders that appear
    in :func:`user_dirs` (curated keys) or the last :func:`sphere_dirs` set —
    never an arbitrary caller-supplied path.
    """
    for entry in user_dirs():
        if entry["key"] == key:
            return Path(entry["path"])
    if not _SPHERE_INDEX:
        sphere_dirs()  # cold /open before the page rendered — populate the cache
    path = _SPHERE_INDEX.get(key)
    if path and Path(path).exists():
        return Path(path)
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
