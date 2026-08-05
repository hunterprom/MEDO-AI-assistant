"""Scan a local app's UI so MEDO learns where its controls are (the foundation).

Two seams keep this testable AND safe:

* ``scan_app(walker, …)`` is a PURE transform over whatever a *walker* yields —
  unit-tested with a fake walker, no desktop needed.
* ``UiaWalker`` is the real Windows implementation (``uiautomation``), used only
  at runtime and written defensively (depth/count/time caps, try/except
  everywhere), exactly like ``software/accessibility.py``.

Exploration is **"reveal, then back out"**: it reads the control tree and OPENS
menus to enumerate their items, pressing Escape after — it never invokes a
normal (potentially destructive) control. Vision is the backup: when UIA yields
very little, ``scan_app`` records that a vision pass is warranted (the pass
itself is a follow-on; the seam and the flag are here).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from software.knowledge import AppMap, UIElement, clock

logger = logging.getLogger(__name__)

#: UIA control types worth remembering as things MEDO can go to / act on.
CLICKABLE_ROLES = frozenset({
    "Button", "Hyperlink", "CheckBox", "RadioButton", "MenuItem", "TabItem",
    "ListItem", "ComboBox", "SplitButton", "TreeItem", "Edit",
})

#: Below this many UIA controls, an app (often Electron/custom, like CapCut)
#: is under-exposing its tree — flag it so a vision pass can fill in later.
_VISION_HINT_BELOW = 8


@dataclass
class RawElement:
    name: str
    role: str
    path: Tuple[str, ...] = ()
    rect: Optional[Tuple[int, int, int, int]] = None
    source: str = "uia"


class Walker(Protocol):
    """What ``scan_app`` needs from the OS. The real one walks uiautomation;
    tests inject a fake."""

    def elements(self, window_hint: str, *,
                 reveal_menus: bool) -> Sequence[RawElement]: ...

    def menus(self, window_hint: str) -> Dict[str, List[str]]: ...


def _clickable(role: str) -> bool:
    return role in CLICKABLE_ROLES


def scan_app(walker: Walker, app_id: str, display_name: str, *,
             reveal_menus: bool = True, window_hint: str = "",
             now=clock) -> AppMap:
    """Turn a walker's raw enumeration into a stored :class:`AppMap` (pure).

    Deduplicates by (name, role), turns menu items into clickable elements
    tagged with their menu, and notes when the tree was too thin to trust."""
    hint = window_hint or display_name
    try:
        raws = list(walker.elements(window_hint=hint, reveal_menus=reveal_menus))
    except Exception:
        logger.warning("walker.elements failed for %r", hint, exc_info=True)
        raws = []
    try:
        menus = dict(walker.menus(window_hint=hint)) if reveal_menus else {}
    except Exception:
        logger.warning("walker.menus failed for %r", hint, exc_info=True)
        menus = {}

    seen: set = set()
    elements: List[UIElement] = []
    for r in raws:
        name = (r.name or "").strip()
        if not name:
            continue
        key = (name.lower(), r.role)
        if key in seen:
            continue
        seen.add(key)
        elements.append(UIElement(
            name=name, role=r.role, path=tuple(r.path or ()),
            clickable=_clickable(r.role), rect=r.rect, source=r.source))

    uia_count = len(elements)
    for menu, items in menus.items():
        for item in items:
            item = (item or "").strip()
            if not item:
                continue
            key = (item.lower(), "MenuItem")
            if key in seen:
                continue
            seen.add(key)
            elements.append(UIElement(
                name=item, role="MenuItem", path=(menu,), clickable=True,
                rect=None, source="menu", keywords=(menu,)))

    notes = ""
    if uia_count < _VISION_HINT_BELOW:
        notes = ("thin UIA tree — a vision pass would help map this app "
                 "(likely a custom/Electron UI)")
    return AppMap(app_id=app_id, display_name=display_name,
                  scanned_at=float(now()), elements=elements, menus=menus,
                  notes=notes)


def is_thin(app_map: AppMap, below: int = _VISION_HINT_BELOW) -> bool:
    """The UIA scan saw too little to trust — a vision pass is warranted. Counts
    only genuinely UIA-sourced controls (menu items are separate)."""
    return sum(1 for e in app_map.elements if e.source == "uia") < below


def vision_augment(app_map: AppMap, vision_elements) -> AppMap:
    """Merge vision-derived controls into a map, de-duplicating by name (pure).
    Updates ``notes`` to reflect what vision added."""
    have = {e.name.lower() for e in app_map.elements}
    added = 0
    for el in vision_elements:
        if not el.name or el.name.lower() in have:
            continue
        have.add(el.name.lower())
        app_map.elements.append(el)
        added += 1
    app_map.notes = (f"vision added {added} controls the UIA tree didn't expose"
                     if added else app_map.notes)
    return app_map


def window_point(norm_xy: Tuple[int, int],
                 rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """A vision control's 0-1000 window point -> absolute screen pixels against
    a live window ``rect`` (left, top, right, bottom). Pure — so a moved window
    still clicks right as long as its layout hasn't changed."""
    x, y = norm_xy
    left, top, right, bottom = rect
    w, h = max(1, right - left), max(1, bottom - top)
    px = left + round(max(0, min(1000, x)) / 1000.0 * w)
    py = top + round(max(0, min(1000, y)) / 1000.0 * h)
    return int(px), int(py)


class UiaWalker:
    """The real Windows walker (``uiautomation``). Defensive: any failure yields
    *less*, never an exception. It only ever OPENS menus (expand / click a menu-
    bar item) and Escapes afterward — it never invokes another control."""

    def __init__(self, *, max_elements: int = 400, max_depth: int = 12,
                 max_seconds: float = 8.0) -> None:
        self._max_elements = max_elements
        self._max_depth = max_depth
        self._max_seconds = max_seconds

    def _window(self, auto, window_hint):
        win = auto.WindowControl(searchDepth=1, SubName=window_hint)
        if win.Exists(maxSearchSeconds=1.5):
            return win
        # Electron/custom apps often expose a top-level Pane, not a Window.
        pane = auto.PaneControl(searchDepth=1, SubName=window_hint)
        return pane if pane.Exists(maxSearchSeconds=0.5) else None

    def elements(self, window_hint: str, *,
                 reveal_menus: bool = True) -> List[RawElement]:
        try:
            import uiautomation as auto
        except Exception:
            return []
        out: List[RawElement] = []
        try:
            win = self._window(auto, window_hint)
            if win is None:
                return []
            start = time.monotonic()
            stack = [(win, (), 0)]
            while stack and len(out) < self._max_elements:
                if time.monotonic() - start > self._max_seconds:
                    break
                node, path, depth = stack.pop()
                try:
                    children = node.GetChildren()
                except Exception:
                    children = []
                for ch in children:
                    try:
                        name = (ch.Name or "").strip()
                        role = ch.ControlTypeName.replace("Control", "")
                        rect = None
                        try:
                            r = ch.BoundingRectangle
                            rect = (r.left, r.top, r.right, r.bottom)
                        except Exception:
                            rect = None
                        if name:
                            out.append(RawElement(name=name, role=role,
                                                  path=path, rect=rect))
                        if depth < self._max_depth and len(out) < self._max_elements:
                            nxt = path + (name,) if name else path
                            stack.append((ch, nxt, depth + 1))
                    except Exception:
                        continue
        except Exception:
            logger.warning("UIA element walk failed for %r", window_hint,
                           exc_info=True)
        return out

    def menus(self, window_hint: str) -> Dict[str, List[str]]:
        try:
            import uiautomation as auto
        except Exception:
            return {}
        menus: Dict[str, List[str]] = {}
        try:
            win = self._window(auto, window_hint)
            if win is None:
                return {}
            bar = win.MenuBarControl(searchDepth=6)
            if not bar.Exists(maxSearchSeconds=1.0):
                return {}
            for top in bar.GetChildren():
                try:
                    label = (top.Name or "").strip()
                    if not label:
                        continue
                    items = self._open_and_read(auto, top)
                    if items:
                        menus[label] = items
                except Exception:
                    continue
        except Exception:
            logger.warning("UIA menu reveal failed for %r", window_hint,
                           exc_info=True)
        return menus

    def _open_and_read(self, auto, top) -> List[str]:
        """Open one menu, read its item names, then ALWAYS Escape back out."""
        items: List[str] = []
        try:
            try:
                top.GetExpandCollapsePattern().Expand()
            except Exception:
                top.Click(simulateMove=False)
            time.sleep(0.15)
            menu = auto.MenuControl(searchDepth=8)
            if menu.Exists(maxSearchSeconds=0.6):
                for mi in menu.GetChildren():
                    n = (mi.Name or "").strip()
                    if n:
                        items.append(n)
        except Exception:
            pass
        finally:
            try:                       # back out, twice, no matter what happened
                auto.SendKeys("{Esc}")
                auto.SendKeys("{Esc}")
            except Exception:
                pass
        return items
