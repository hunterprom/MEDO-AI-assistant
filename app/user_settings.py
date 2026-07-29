"""Per-user app settings — the file a non-technical user's choices live in.

Separate from the dev ``config.yaml`` (which keeps working for the CLI). Lives at
``%APPDATA%\\MEDO\\settings.json`` on Windows (``~/.config/MEDO`` elsewhere) so a
normal user never edits YAML. Holds the chosen hardware profile (auto or manual
override) plus the plain-language explanation to show them. All functions take an
optional explicit ``path`` so tests never touch the real user directory.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def user_dir() -> Path:
    base = os.environ.get("APPDATA")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "MEDO"


def settings_path() -> Path:
    return user_dir() / "settings.json"


def _path(path: Optional[Path]) -> Path:
    return Path(path) if path is not None else settings_path()


def load(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the user settings; {} on missing/corrupt (never raises)."""
    p = _path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(data: Dict[str, Any], path: Optional[Path] = None) -> None:
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, p)          # atomic swap so a crash can't truncate settings


def get_profile(path: Optional[Path] = None) -> Optional[str]:
    return load(path).get("profile")


def is_override(path: Optional[Path] = None) -> bool:
    return bool(load(path).get("profile_override", False))


def set_profile(name: str, *, override: bool, explanation: str = "",
                path: Optional[Path] = None) -> None:
    """Persist the chosen profile. ``override=True`` marks it a MANUAL choice, so
    a later auto-detect won't quietly change it out from under the user."""
    data = load(path)
    data["profile"] = name
    data["profile_override"] = bool(override)
    if explanation:
        data["profile_explanation"] = explanation
    save(data, path)
