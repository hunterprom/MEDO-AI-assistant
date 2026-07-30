"""Vision backup for the UI scanner — see controls UIA can't (Electron/CapCut).

When a UIA scan comes back thin, this looks at a screenshot of the app window
with the local vision model (qwen2.5-vl) and turns what it sees into learnable
controls: a label per control, a coarse area, and the model's 0-1000 centre
point (re-resolved to live pixels at click time). Capture + model call are
injectable, so the parse is unit-tested without a screen or Ollama.

Honest limits: a 3B model's labels and points are approximate — great for
"where is X" (locate), best-effort for clicking (navigate re-resolves the point
against the live window, so a moved window still works if its layout hasn't).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from typing import List, Optional

from software.knowledge import UIElement

logger = logging.getLogger(__name__)

VISION_SCAN_PROMPT = (
    "This is a screenshot of ONE application window. List its interactive UI "
    "controls a user could click: buttons, tabs, tools, panels, menus, search "
    "boxes, sliders. Reply with ONLY a JSON array — no prose, no markdown. Each "
    "item: {\"label\": short control name, \"area\": one of "
    "top/left/right/center/bottom, \"x\": int 0-1000, \"y\": int 0-1000}. x and "
    "y are the control's CENTRE on THIS image, normalized 0-1000 (x left->right, "
    "y top->bottom). List up to 40 of the most useful, distinct controls."
)


def parse_vision_elements(raw: str) -> List[UIElement]:
    """Turn a model reply into vision ``UIElement``s. Tolerant of prose/junk:
    accepts a bare JSON array or the first ``[...]`` blob in the text."""
    if not raw:
        return []
    data = None
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, list):
        return []
    out: List[UIElement] = []
    seen = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label or label.lower() in seen:
            continue
        seen.add(label.lower())
        area = str(item.get("area") or "").strip().lower()
        xy: Optional[tuple] = None
        try:
            x, y = item.get("x"), item.get("y")
            if x is not None and y is not None:
                xy = (int(x), int(y))
        except Exception:
            xy = None
        out.append(UIElement(name=label, role="Vision", clickable=True,
                             source="vision", keywords=((area,) if area else ()),
                             vision_xy=xy))
    return out


class VisionProbe:
    """Async: screenshot the focused app window, ask the vision model, parse.

    ``capture(app_hint)`` returns a PIL image of the app window (or ``None``);
    ``describe(image_bytes)`` returns the model's text. Both are injectable; the
    defaults use MEDO's existing vision stack (``grab_desktop`` + ``_describe``)
    and the connectors' mechanisms to find the window rect."""

    def __init__(self, settings, mechanisms=None, *, capture=None,
                 describe=None) -> None:
        self._settings = settings
        self._mech = mechanisms
        self._capture = capture
        self._describe = describe

    async def __call__(self, app_hint: str) -> List[UIElement]:
        try:
            grabbed = await asyncio.to_thread(self._grab, app_hint)
        except Exception:
            logger.warning("vision capture failed for %r", app_hint, exc_info=True)
            return []
        # _grab returns (image, windowed); an injected capture returns just an
        # image (assumed window-framed).
        image, windowed = grabbed if isinstance(grabbed, tuple) else (grabbed, True)
        if image is None:
            return []
        try:
            raw = await self._ask(image)
        except Exception:
            logger.warning("vision model call failed", exc_info=True)
            return []
        elements = parse_vision_elements(raw)
        if not windowed:
            # The capture was the WHOLE desktop, so the model's 0-1000 points are
            # desktop-normalized — they can't be re-resolved to a window click.
            # Keep the labels (useful for locate) but drop the unusable coords.
            for el in elements:
                el.vision_xy = None
        return elements

    # -- default (real) backends ----------------------------------------------

    def _grab(self, app_hint: str):
        if self._capture is not None:
            return self._capture(app_hint)
        from skills.vision_skill import grab_desktop

        img, (ox, oy) = grab_desktop()
        rect = self._mech.foreground_rect() if self._mech is not None else None
        if rect:
            left, top, right, bottom = rect
            box = (max(0, left - ox), max(0, top - oy),
                   min(img.width, right - ox), min(img.height, bottom - oy))
            if box[2] > box[0] and box[3] > box[1]:
                return img.crop(box), True          # window-framed coords
        return img, False                           # whole desktop — coords unusable

    async def _ask(self, image) -> str:
        if self._describe is not None:
            return await self._describe(image)
        from skills.vision_skill import _describe, _shrink

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        small = _shrink(buf.getvalue())
        result = await _describe(self._settings, base64.b64encode(small).decode(),
                                 VISION_SCAN_PROMPT)
        return result.speech if result.success else ""
