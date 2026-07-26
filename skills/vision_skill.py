"""Seeing skills: describe the camera or the screen with a local vision model.

"what do you see" grabs the vision sidecar's latest camera frame; "read my
screen" captures the display. Either image goes to a small local Ollama vision
model (moondream by default). Both source projects pointed at this feature —
neither shipped it. Degrades gracefully when the sidecar, Ollama, or the model
are missing.

VRAM note: on a 12 GB GPU, loading the vision model evicts a partially
offloaded qwen3:30b, so the next chat pays a reload — ``llm.keep_alive``
softens the rest.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
from typing import Any

from core import mk
from core.config import Settings
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

#: Appended when the question was Macedonian. The instruction itself stays in
#: English: a 3B vision model comprehends the task far better that way, and it
#: still honours the output language.
ANSWER_MK = " Answer in Macedonian."

DESCRIBE_CAMERA_PROMPT = (
    "Describe what you see in one or two short spoken sentences. Plain text.\n"
    "Identify objects as specifically as the image allows: not 'a device' but "
    "'a black digital watch'; not 'a board' but 'an Arduino-style "
    "microcontroller board with a USB-B socket'. Small dark objects on a dark "
    "background are the usual failure — say what its shape, size relative to a "
    "hand, and any visible markings suggest. If you can read a part number, "
    "silkscreen or logo, read it out; that is usually the whole answer. When "
    "you genuinely cannot tell, name your best guess AND say it is a guess "
    "rather than inventing a confident label."
)
DESCRIBE_SCREEN_PROMPT = (
    "This is a computer screen. Describe what is on it in one or two short "
    "spoken sentences. Plain text."
)
READ_SCREEN_PROMPT = (
    "This is a computer screen. Read out the main text visible on it, briefly. "
    "If there is no readable text, say so in one short sentence."
)
POINT_PROMPT = (
    "This is a close-up of a computer screen, centered on the user's mouse "
    "cursor. Tell the user what the cursor is pointing at, in one or two "
    "short spoken sentences. Plain text."
)
#: For a photo or a picture rather than a scene — "what's in this picture".
#: A photograph wants composition and subject; an object wants a part number.
DESCRIBE_PICTURE_PROMPT = (
    "Describe this picture in two or three short spoken sentences: the "
    "subject, what is happening, and anything notable about the setting or "
    "style. Read out any text visible in the image. Plain text, no markdown."
)


def grab_desktop():
    """(screenshot of the WHOLE virtual desktop, its ``(origin_x, origin_y)``).

    ``pyautogui.screenshot()`` only captures the PRIMARY display, so on a
    multi-monitor setup anything on a second screen was invisible — and the
    cursor position (which is virtual-desktop wide, and can be negative for a
    monitor left of / above the primary) indexed the WRONG pixels, silently
    describing the wrong region instead of failing. Windows can grab every
    screen at once; elsewhere we fall back to the primary capture and return
    the origin so callers can translate/bounds-check honestly.
    """
    import sys

    if sys.platform == "win32":
        try:
            import ctypes

            from PIL import ImageGrab

            u = ctypes.windll.user32
            img = ImageGrab.grab(all_screens=True)
            # image (0,0) == virtual-desktop origin (SM_X/YVIRTUALSCREEN)
            return img, (int(u.GetSystemMetrics(76)), int(u.GetSystemMetrics(77)))
        except Exception:  # noqa: BLE001 - fall back to the primary grab
            pass
    import pyautogui

    return pyautogui.screenshot(), (0, 0)


def crop_box(cx: int, cy: int, width: int, height: int,
             size: int = 480) -> tuple[int, int, int, int]:
    """Square crop centered on the cursor, clamped inside the screen.

    Near an edge the box slides inward (stays ``size`` wide) rather than
    shrinking, so the model always gets the same amount of context. Pure —
    unit-tested without a screen.
    """
    size = min(size, width, height)
    left = max(0, min(cx - size // 2, width - size))
    top = max(0, min(cy - size // 2, height - size))
    return left, top, left + size, top + size


def _shrink(image_bytes: bytes, max_side: int = 1280) -> bytes:
    """Downscale big captures before the vision model sees them.

    qwen2.5-vl uses dynamic resolution: a full 2560x1440 screenshot explodes
    into thousands of image tokens and blows the 60 s timeout on a 12 GB GPU
    (surfaced as "can't reach my vision model"). ~1280 px keeps screen text
    legible and answers in seconds. Best-effort — undecodable bytes pass
    through untouched.
    """
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        scale = max_side / max(img.size)
        if scale >= 1.0:
            return image_bytes
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return image_bytes


async def _describe(settings: Settings, image_b64: str, prompt: str,
                    speak_mk: bool = False) -> SkillResult:
    """Send one base64 image to the local Ollama vision model."""
    import httpx

    if speak_mk:
        prompt += ANSWER_MK
    host = settings.llm.host.rstrip("/")
    model = settings.vision_llm.model
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    # One retry: the FIRST look after an idle spell hits a cold vision model
    # (~10 s to load a 3B), and a request landing mid-load gets refused — which
    # surfaced as "is Ollama running?" while Ollama was in fact running fine.
    last_exc: Exception | None = None
    for attempt in (0, 1):
        try:
            async with httpx.AsyncClient(timeout=settings.vision_llm.timeout_s) as client:
                resp = await client.post(f"{host}/api/generate", json=payload)
                if resp.status_code == 404:
                    return SkillResult(
                        f"Моделот за гледање не е инсталиран — пушти: ollama pull {model}."
                        if speak_mk else
                        f"The vision model isn't installed — run: ollama pull {model}.",
                        success=False,
                    )
                resp.raise_for_status()
                answer = (resp.json().get("response") or "").strip()
            break
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt == 0:
                logger.info("vision call failed (%s) — retrying once while the "
                            "model loads", exc)
                await asyncio.sleep(2.0)
    else:
        logger.warning("vision model unreachable: %s", last_exc)
        return SkillResult(
            "Не можам да го стигнам моделот за гледање — дали работи Ollama?"
            if speak_mk else
            "I can't reach my vision model right now — is Ollama running?",
            success=False,
        )
    return SkillResult(
        answer or ("Не можев да разберам што има таму." if speak_mk
                   else "I couldn't make anything out."),
        data={"model": model})


class SeeCameraSkill(Skill):
    name = "see_camera"
    description = "Describe what the webcam currently sees."
    routing_phrases = [
        "what am I holding up right now",
        "take a look through the webcam and tell me what's there",
        "what's in front of me",
        "describe what I'm showing you",
        "look through the camera and tell me what you notice",
        "what's sitting on my desk in view",
    ]

    patterns = [
        # "you" is tolerated as the common Whisper mishears (yuo/u) so a small
        # STT slip doesn't drop the query to the LLM (which then parrots stale
        # answers from memory instead of actually looking).
        # ...but a "see" query that names the SCREEN belongs to see_screen
        # (registered after this), so exclude it here.
        re.compile(r"\bwhat\s+(?:do|can|are)\s+(?:you|yuo|u)\s+see(?:ing)?\b"
                   r"(?!.*\b(?:screen|monitor|display)\b)", re.IGNORECASE),
        re.compile(r"\bdescribe\s+(?:what\s+you\s+see|the\s+(?:camera|room|view))\b", re.IGNORECASE),
        re.compile(r"\blook\s+(?:at\s+(?:me|this)|around)\b", re.IGNORECASE),
        # Deictic "look at what I'm showing you" phrasings that name no camera —
        # they reached the LLM, which fabricated or claimed it couldn't see.
        re.compile(r"\bwhat\s+am\s+i\s+holding(?:\s+up)?\b", re.IGNORECASE),
        re.compile(r"\bwhat(?:'?s| is)\s+in\s+front\s+of\s+me\b", re.IGNORECASE),
        # Modal expansion (mirrors see_screen): "could/would/will you see me",
        # "do you see anything", "tell me what you see" reached the LLM before.
        re.compile(r"\b(?:can|could|would|will|do)\s+(?:you|yuo|u)\s+see\s+"
                   r"(?:me|anything|this|us)\b", re.IGNORECASE),
        re.compile(r"\btell\s+me\s+what\s+(?:you|yuo|u)\s+see\b"
                   r"(?!.*\b(?:screen|monitor|display)\b)", re.IGNORECASE),
        # Camera-oriented phrasings (never 'screen' — that's the see_screen skill).
        re.compile(r"\b(?:see|look\s+at|check|use|through)\s+(?:the\s+|your\s+|my\s+)?"
                   r"(?:camera|webcam)\b", re.IGNORECASE),
        re.compile(r"\bwhat(?:'?s| is)?\s+(?:on|in\s+front\s+of)\s+(?:the\s+|your\s+|my\s+)?"
                   r"(?:camera|webcam)\b", re.IGNORECASE),
        # MK. The lookahead matters: this skill is registered BEFORE the screen
        # one, so a bare "што гледаш" must not swallow "што гледаш на екранот".
        re.compile(r"\bшто\s+гледаш\b(?!.*екран)", re.IGNORECASE),
        re.compile(r"\bопиши\s+(?:што\s+гледаш|ја\s+собата|ја\s+камерата)\b(?!.*екран)",
                   re.IGNORECASE),
        re.compile(r"\bдали\s+ме\s+гледаш\b|\bме\s+гледаш\s+ли\b", re.IGNORECASE),
        re.compile(r"\bпогледни\s+(?:ме|наоколу)\b", re.IGNORECASE),
    ]

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def execute(self, request: SkillRequest) -> SkillResult:
        import httpx

        speak_mk = mk.is_cyrillic(request.text)
        port = self._settings.vision.stream_port
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"http://127.0.0.1:{port}/frame.jpg")
                resp.raise_for_status()
                frame = resp.content
        except httpx.HTTPError:
            return SkillResult(
                "Камерата не работи — пушти го vision сидекарот и пробај пак."
                if speak_mk else
                "The camera isn't running — start the vision sidecar and try again.",
                success=False,
            )
        return await _describe(
            self._settings, base64.b64encode(frame).decode(),
            DESCRIBE_CAMERA_PROMPT, speak_mk
        )

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }


class SeeScreenSkill(Skill):
    name = "see_screen"
    description = "Describe what's on the screen, or read its text aloud."
    routing_phrases = [
        "can you tell what's on my monitor",
        "have a look at my display and tell me what it shows",
        "what does it say on the screen right now",
        "summarize whatever is open on my screen",
        "which window is this that's open",
        "tell me what's showing on my computer",
    ]

    patterns = [
        re.compile(r"\bwhat(?:'?s| is)\s+on\s+(?:my|the)\s+(?:screen|monitor|display)\b",
                   re.IGNORECASE),
        re.compile(r"\b(?:read|describe)\s+(?:my|the)\s+(?:screen|monitor|display)\b",
                   re.IGNORECASE),
        # "what do you see on the screen" — a see-query that names the screen.
        re.compile(r"\bwhat\s+(?:do|can|are)\s+(?:you|yuo|u)\s+see(?:ing)?\b"
                   r"(?=.*\b(?:screen|monitor|display)\b)", re.IGNORECASE),
        # "can/could/would you see/view/look at my/the/your screen" — these went
        # to the LLM, which claims it can't see screens instead of calling this
        # tool. Cover the modal (can/could/would) AND the possessive (my/the/
        # your): "could you see YOUR screen" tripped none of the old ones.
        re.compile(r"\b(?:can|could|would|will)\s+you\s+(?:see|view|check|look\s+at)"
                   r"\s+(?:my|the|your)\s+screen\b", re.IGNORECASE),
        re.compile(r"\b(?:look\s+at|check)\s+(?:my|the|your)\s+screen\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+am\s+i\s+looking\s+at\b", re.IGNORECASE),
        # Bare "see screen" — and the "C screen" Whisper produces for it, which
        # otherwise fell to the LLM and got a "no screen tool" hallucination.
        re.compile(r"\b(?:see|c)\s+(?:the\s+|my\s+|your\s+)?screen\b", re.IGNORECASE),
        # MK: "што гледаш на екранот", "што има на мојот екран". One optional
        # word before "екран" absorbs the possessive, which Whisper spells
        # several ways (мојот / твојот / твоот).
        re.compile(r"\bшто\s+(?:гледаш|има|е|гледате)\s+на\s+(?:\S+\s+)?екран",
                   re.IGNORECASE),
        re.compile(r"\b(?:прочитај|опиши|погледни|види)\s+(?:го\s+)?(?:\S+\s+)?екран",
                   re.IGNORECASE),
        re.compile(r"\bдали\s+(?:го\s+)?гледаш\s+(?:\S+\s+)?екран", re.IGNORECASE),
    ]

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def execute(self, request: SkillRequest) -> SkillResult:
        text = request.text.lower()
        wants_read = (
            request.args.get("mode") == "read"
            # whole word only: "already", "ready", "spread", "thread" all
            # contain "read" and silently flipped describe -> read (transcribe),
            # a different answer to "what's on my screen already".
            or bool(re.search(r"\bread\b", text))
            or "прочитај" in text          # "прочитај го екранот" = read it out
        )
        try:
            # Whole virtual desktop, not just the primary display — otherwise a
            # window on a second monitor is simply invisible to "read my screen".
            shot, _origin = await asyncio.to_thread(grab_desktop)
            buf = io.BytesIO()
            shot.save(buf, format="PNG")
            small = await asyncio.to_thread(_shrink, buf.getvalue())
        except Exception as exc:
            return SkillResult(
                f"Не успеав да го фатам екранот: {exc}" if mk.is_cyrillic(request.text)
                else f"I couldn't capture the screen: {exc}", success=False)
        prompt = READ_SCREEN_PROMPT if wants_read else DESCRIBE_SCREEN_PROMPT
        return await _describe(
            self._settings, base64.b64encode(small).decode(), prompt,
            mk.is_cyrillic(request.text)
        )

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["describe", "read"],
                            "description": "'read' transcribes visible text; 'describe' summarizes.",
                        }
                    },
                    "required": [],
                },
            },
        }


class PointAtSkill(Skill):
    """M11 deictic pointing: "what is this?" looks where the cursor is.

    Crops a square around the current mouse cursor (driven by hand in pointer
    mode or by the physical mouse — either way the position is the intent),
    and asks the vision model about JUST that region. The whole-utterance
    anchor on "what is this/that" keeps richer questions ("what is this song")
    on the LLM path; "што е ова" stays with see_bench (bench context).
    """

    name = "what_is_this"
    description = (
        "Identify what the mouse cursor is currently pointing at on screen "
        "(a close-up look at the region around the cursor)."
    )
    routing_phrases = [
        "identify what my cursor is on",
        "what's the thing I'm hovering over",
        "what's under the pointer right now",
        "what is the item my cursor is sitting on",
        "name whatever the mouse is pointing to",
    ]

    patterns = [
        re.compile(r"^\s*what(?:'?s|\s+is)\s+(?:this|that)\b[\s?.!]*$", re.IGNORECASE),
        re.compile(r"\bwhat\s+am\s+i\s+pointing\s+(?:at|to)\b", re.IGNORECASE),
        re.compile(r"\bwhat(?:'?s|\s+is)\s+(?:this|that|it)\s+(?:under|at|near)\s+"
                   r"(?:my|the)\s+(?:cursor|mouse|pointer)\b", re.IGNORECASE),
    ]

    def __init__(self, settings: Settings, capture=None, describe=None) -> None:
        self._settings = settings
        self._capture = capture or self._default_capture
        self._describe = describe or self._default_describe

    @staticmethod
    def _default_capture():
        """(desktop image, cursor xy IN IMAGE COORDS) — runs in a worker thread.

        The cursor reports virtual-desktop coordinates, so it's translated into
        the captured image's space; otherwise a cursor on a second monitor
        pointed at the wrong pixels of the primary one.
        """
        import pyautogui

        shot, (ox, oy) = grab_desktop()
        pos = pyautogui.position()
        return shot, (int(pos.x) - ox, int(pos.y) - oy)

    async def _default_describe(self, image_b64: str) -> SkillResult:
        return await _describe(self._settings, image_b64, POINT_PROMPT)

    async def execute(self, request: SkillRequest) -> SkillResult:
        try:
            shot, (cx, cy) = await asyncio.to_thread(self._capture)
        except Exception as exc:
            return SkillResult(f"I couldn't capture the screen: {exc}", success=False)
        # Honest failure beats a confidently wrong answer: if the cursor sits on
        # a display this capture doesn't cover, say so instead of clamping the
        # crop to the primary screen's edge and describing the wrong pixels.
        if not (0 <= cx < shot.width and 0 <= cy < shot.height):
            return SkillResult(
                "Стрелката е на екран што не можам да го фотографирам."
                if mk.is_cyrillic(request.text) else
                "The cursor is on a display I can't capture — move it to the "
                "main screen and ask again.",
                success=False,
            )
        crop = shot.crop(crop_box(cx, cy, shot.width, shot.height))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        result = await self._describe(base64.b64encode(buf.getvalue()).decode())
        result.data.setdefault("cursor", [cx, cy])
        return result

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
