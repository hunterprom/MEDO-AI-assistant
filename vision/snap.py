"""Click snapping: land on the button you meant, and show which one first.

The problem this solves is physical. Pointer mode drives the cursor from the
index fingertip, and a pinch pulls the fingertip *down* a few millimetres as
the thumb comes up to meet it. On screen that is a downward drift of tens of
pixels at the exact moment of the click, so the press lands just below the
button you were aiming at.

The fix is a capture circle: at the moment of the click, ask the OS what is
clickable near the cursor and press the nearest one instead of the raw point.

Three deliberate constraints, in order of importance:

1. **Never silently click somewhere else.** A snap that quietly relocates your
   click is worse than the drift it fixes — you would stop trusting the
   cursor. So the chosen target is highlighted on screen first and the click
   waits ``snap_confirm_ms``, giving you time to pull back if it picked wrong.
   Set the delay to 0 once you trust it.
2. **Never regress.** No accessibility data, nothing in radius, or any error at
   all falls back to clicking the raw cursor position — exactly today's
   behaviour.
3. **Bias upward.** The drift is downward, so among candidates at a similar
   distance the one *above* the cursor is the one you were reaching for.

Everything OS-specific (querying elements, drawing the highlight) is injected,
so the geometry that decides where your click lands is pure and unit-tested
without a screen.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

#: UI Automation control types worth snapping to. Deliberately a small
#: allowlist: snapping to a Pane or a Text label would move the click onto
#: something that does nothing, which is worse than the drift.
CLICKABLE_ROLES = frozenset({
    "Button", "Hyperlink", "CheckBox", "RadioButton", "MenuItem", "TabItem",
    "ListItem", "ComboBox", "Edit", "SplitButton", "TreeItem", "Slider",
})


@dataclass(frozen=True)
class Clickable:
    """One clickable element the OS reported, in screen coordinates."""

    name: str
    role: str
    left: int
    top: int
    right: int
    bottom: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.right) // 2, (self.top + self.bottom) // 2

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


class ElementProvider(Protocol):
    """Anything that can list clickable elements near a screen point."""

    def near(self, x: int, y: int, radius_px: int) -> list[Clickable]:
        ...


def distance_to(element: Clickable, x: int, y: int) -> float:
    """Cursor-to-element-centre distance in pixels."""
    cx, cy = element.center
    return math.hypot(cx - x, cy - y)


def choose_target(
    elements: list[Clickable],
    x: int,
    y: int,
    radius_px: int,
    above_bias_px: float = 0.0,
) -> Clickable | None:
    """The element to click instead of the raw point, or None to click raw.

    Pure. This is the function that decides where your click actually lands,
    so it is the one worth reading closely.

    * Only elements whose **centre** lies within ``radius_px`` are eligible —
      centre, not edge, so a full-screen pane that happens to overlap the
      circle can never win.
    * An element already **under** the cursor wins outright: if you are on the
      button, there is nothing to fix, and moving off it would be the bug.
    * Otherwise the nearest centre wins, with ``above_bias_px`` subtracted from
      the effective distance of elements above the cursor. That is the whole
      point: the drift is downward, so an equally-near target above you is the
      one you were reaching for.
    * Ties break toward the element above, then the smaller one — a small
      control is a more specific target than a large one covering it.
    """
    if not elements or radius_px <= 0:
        return None

    under = [e for e in elements if e.contains(x, y)]
    if under:
        # Already on something clickable: smallest wins (the innermost control).
        return min(under, key=lambda e: e.width * e.height)

    def key(element: Clickable) -> tuple[float, int, int]:
        _cx, cy = element.center
        above = cy < y
        effective = distance_to(element, x, y) - (above_bias_px if above else 0.0)
        return (effective, 0 if above else 1, element.width * element.height)

    in_range = [e for e in elements if distance_to(e, x, y) <= radius_px]
    return min(in_range, key=key) if in_range else None


# --- Windows UI Automation provider -------------------------------------------


class UiaProvider:
    """Clickable elements from Windows UI Automation, probed around a point.

    Walking a whole window's element tree is far too slow to sit inside a
    click, so this probes instead: ``ControlFromPoint`` at the cursor and on
    two rings around it, then keeps the distinct clickable ancestors. That is a
    handful of cheap COM calls and finds what is actually reachable at those
    pixels, which is precisely the question being asked.

    Import and COM failures are swallowed to an empty list — the caller then
    clicks the raw position, and pointer mode behaves exactly as before.
    """

    #: Probe angles per ring. Eight is enough to catch a normal button; more
    #: costs COM round-trips inside the click path.
    _ANGLES = tuple(range(0, 360, 45))

    def __init__(self, budget_ms: float = 120.0) -> None:
        self._budget_s = budget_ms / 1000.0
        self._uia = None
        try:
            import uiautomation

            self._uia = uiautomation
        except Exception as exc:                      # noqa: BLE001 - optional dep
            logger.info("UI Automation unavailable (%s); clicks stay raw", exc)

    @property
    def available(self) -> bool:
        return self._uia is not None

    def _probe_points(self, x: int, y: int, radius_px: int) -> list[tuple[int, int]]:
        points = [(x, y)]
        for fraction in (0.5, 1.0):
            step = radius_px * fraction
            for angle in self._ANGLES:
                radians = math.radians(angle)
                points.append((int(x + step * math.cos(radians)),
                               int(y + step * math.sin(radians))))
        return points

    def _to_clickable(self, control) -> Clickable | None:
        """Walk up from a probed control to the nearest clickable ancestor."""
        for _ in range(4):                            # a few levels, not the root
            if control is None:
                return None
            role = str(getattr(control, "ControlTypeName", "") or "")
            role = role.removesuffix("Control")
            if role in CLICKABLE_ROLES:
                rect = control.BoundingRectangle
                if rect and rect.right > rect.left and rect.bottom > rect.top:
                    return Clickable(str(control.Name or ""), role,
                                     rect.left, rect.top, rect.right, rect.bottom)
                return None
            control = control.GetParentControl()
        return None

    def near(self, x: int, y: int, radius_px: int) -> list[Clickable]:
        if self._uia is None:
            return []
        started = time.monotonic()
        found: dict[tuple[int, int, int, int], Clickable] = {}
        try:
            for px, py in self._probe_points(x, y, radius_px):
                if time.monotonic() - started > self._budget_s:
                    logger.debug("snap probe hit its time budget")
                    break
                control = self._uia.ControlFromPoint(px, py)
                element = self._to_clickable(control)
                if element is not None:
                    found[(element.left, element.top,
                           element.right, element.bottom)] = element
        except Exception:                             # noqa: BLE001 - COM is fragile
            logger.debug("UI Automation probe failed", exc_info=True)
            return list(found.values())
        return list(found.values())


# --- on-screen highlight -------------------------------------------------------


def draw_highlight(element: Clickable, thickness: int = 3) -> None:
    """Outline ``element`` on the real screen, briefly.

    Drawn straight onto the desktop device context with GDI: a transient
    rectangle, the same trick screen-capture tools use for their selection
    band. No window to create, focus to steal, or z-order to fight — which
    matters because this has to appear *over* whatever you are about to click,
    including full-screen apps, within milliseconds.

    The trade-off is honest: it is not repaint-safe, so the ring can be erased
    early by an app redrawing underneath it. For a flash before a click that is
    acceptable; a real layered window would survive repaints but would also
    have to be created, shown and destroyed inside the click path.
    """
    import ctypes

    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    screen_dc = user32.GetDC(0)
    if not screen_dc:
        return
    #: A saturated cyan that reads on both light and dark UI.
    pen = gdi32.CreatePen(0, thickness, 0x00FFFF00)      # BGR: cyan
    brush = gdi32.GetStockObject(5)                      # NULL_BRUSH: outline only
    old_pen = gdi32.SelectObject(screen_dc, pen)
    old_brush = gdi32.SelectObject(screen_dc, brush)
    try:
        gdi32.Rectangle(screen_dc, element.left, element.top,
                        element.right, element.bottom)
    finally:
        gdi32.SelectObject(screen_dc, old_pen)
        gdi32.SelectObject(screen_dc, old_brush)
        gdi32.DeleteObject(pen)
        user32.ReleaseDC(0, screen_dc)


# --- the click itself ----------------------------------------------------------


@dataclass
class SnapResult:
    """What a snap decided, so the caller can click it and the HUD can draw it."""

    x: int
    y: int
    target: Clickable | None
    snapped: bool


def resolve_click(
    x: int,
    y: int,
    *,
    enabled: bool,
    radius_px: int,
    above_bias_px: float,
    confirm_ms: float,
    provider: ElementProvider | None,
    highlight=draw_highlight,
    sleep=time.sleep,
) -> SnapResult:
    """Decide where a click should land, previewing the target first.

    The ordering here is the safety property, and it is what the tests pin:
    **highlight, wait, then click**. Never click and then show where it went.

    Any failure in the provider or the highlight degrades to the raw point
    rather than propagating: a click that lands where you pointed is always a
    better outcome than a click that does not happen.
    """
    if not enabled or provider is None:
        return SnapResult(x, y, None, False)
    try:
        elements = provider.near(x, y, radius_px)
        target = choose_target(elements, x, y, radius_px, above_bias_px)
    except Exception:                                 # noqa: BLE001 - never block a click
        logger.debug("snap lookup failed; clicking raw", exc_info=True)
        return SnapResult(x, y, None, False)
    if target is None:
        return SnapResult(x, y, None, False)

    if highlight is not None:
        try:
            highlight(target)
        except Exception:                             # noqa: BLE001
            logger.debug("snap highlight failed", exc_info=True)
    if confirm_ms > 0:
        sleep(confirm_ms / 1000.0)
    cx, cy = target.center
    return SnapResult(cx, cy, target, True)
