"""Learned UI maps for local apps — what MEDO knows about how to use software.

MEDO can *scan* an app it controls (``software/ui_scan.py``) and remember where
that app's controls live, so later it knows where to go for "find the effects
search in CapCut". Each app's map is a small JSON file on THIS machine (learned
per install, so git-ignored). This module is pure data + fuzzy lookup + on-disk
persistence; the OS-touching scan lives in ``ui_scan.py``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Per-install, machine-specific, so git-ignored (see .gitignore).
DEFAULT_MAPS_DIR = PROJECT_ROOT / "data" / "software_maps"

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


@dataclass
class UIElement:
    """One control MEDO found in an app, and how to talk about / reach it."""

    name: str
    role: str = ""                  # UIA control type: Button, MenuItem, Edit…
    path: Tuple[str, ...] = ()      # control names from the window down to this one
    clickable: bool = False
    rect: Optional[Tuple[int, int, int, int]] = None  # left, top, right, bottom
    source: str = "uia"             # uia | menu | vision
    keywords: Tuple[str, ...] = ()  # extra search terms (menu name, synonyms)

    def haystack(self) -> str:
        return " ".join([self.name, self.role, " ".join(self.path),
                         " ".join(self.keywords)])

    def location(self) -> str:
        """A short, speakable 'where it lives' hint."""
        if self.source == "menu" and self.path:
            return f"in the {self.path[0]} menu"
        if self.path:
            return f"under {self.path[-1]}" if self.path[-1] else "in the window"
        return "in the window"


@dataclass
class AppMap:
    """Everything MEDO learned about one app in a single scan."""

    app_id: str
    display_name: str = ""
    scanned_at: float = 0.0
    elements: List[UIElement] = field(default_factory=list)
    menus: dict = field(default_factory=dict)   # menu name -> [item names]
    notes: str = ""

    def element_count(self) -> int:
        return len(self.elements)


def map_path(app_id: str, base_dir: Path = DEFAULT_MAPS_DIR) -> Path:
    safe = re.sub(r"[^a-z0-9_-]+", "_", (app_id or "app").lower()).strip("_") or "app"
    return Path(base_dir) / f"{safe}.json"


def save_map(app_map: AppMap, base_dir: Path = DEFAULT_MAPS_DIR) -> Path:
    p = map_path(app_map.app_id, base_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(app_map), indent=2), encoding="utf-8")
    return p


def load_map(app_id: str, base_dir: Path = DEFAULT_MAPS_DIR) -> Optional[AppMap]:
    p = map_path(app_id, base_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        elems = [
            UIElement(
                name=e["name"], role=e.get("role", ""),
                path=tuple(e.get("path") or ()),
                clickable=bool(e.get("clickable")),
                rect=tuple(e["rect"]) if e.get("rect") else None,
                source=e.get("source", "uia"),
                keywords=tuple(e.get("keywords") or ()),
            )
            for e in data.get("elements", []) if e.get("name")
        ]
        return AppMap(
            app_id=data["app_id"], display_name=data.get("display_name", ""),
            scanned_at=float(data.get("scanned_at", 0.0)), elements=elems,
            menus={k: list(v) for k, v in (data.get("menus") or {}).items()},
            notes=data.get("notes", ""))
    except Exception:
        logger.warning("could not load app map %r", app_id, exc_info=True)
        return None


def forget_map(app_id: str, base_dir: Path = DEFAULT_MAPS_DIR) -> bool:
    p = map_path(app_id, base_dir)
    try:
        if p.exists():
            p.unlink()
            return True
    except Exception:
        logger.warning("could not delete app map %r", app_id, exc_info=True)
    return False


def list_maps(base_dir: Path = DEFAULT_MAPS_DIR) -> List[dict]:
    """A light summary of every learned app (for the HUD), newest first."""
    out: List[dict] = []
    try:
        for p in Path(base_dir).glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                out.append({"app_id": d.get("app_id", p.stem),
                            "display_name": d.get("display_name", p.stem),
                            "count": len(d.get("elements", [])),
                            "scanned_at": float(d.get("scanned_at", 0.0))})
            except Exception:
                continue
    except Exception:
        return []
    out.sort(key=lambda m: m["scanned_at"], reverse=True)
    return out


def search(app_map: AppMap, query: str,
           limit: int = 5) -> List[Tuple[UIElement, float]]:
    """Fuzzy-rank the map's controls against a natural query. Token overlap plus
    a phrase/substring bonus, nudged toward clickable + shallow controls.
    Returns the best ``limit`` ``(element, score)`` pairs (score > 0)."""
    qset = set(_tokens(query))
    ql = (query or "").strip().lower()
    if not qset:
        return []
    scored: List[Tuple[UIElement, float]] = []
    for el in app_map.elements:
        hay = el.haystack().lower()
        overlap = len(qset & set(_tokens(hay)))
        if overlap == 0 and ql not in hay:
            continue
        score = float(overlap)
        if ql and ql in el.name.lower():
            score += 3.0
        elif ql and ql in hay:
            score += 1.0
        if el.clickable:
            score += 0.5
        score -= 0.05 * len(el.path)
        scored.append((el, score))
    scored.sort(key=lambda t: (t[1], t[0].clickable), reverse=True)
    return [(el, s) for el, s in scored[:limit] if s > 0]


def clock() -> float:                     # injection seam for deterministic tests
    return time.time()
