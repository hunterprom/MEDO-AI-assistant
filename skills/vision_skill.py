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
import re
from typing import Any

from core.config import Settings
from skills.base import Skill, SkillRequest, SkillResult

DESCRIBE_CAMERA_PROMPT = (
    "Describe what you see in one or two short spoken sentences. Plain text."
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


async def _describe(settings: Settings, image_b64: str, prompt: str) -> SkillResult:
    """Send one base64 image to the local Ollama vision model."""
    import httpx

    host = settings.llm.host.rstrip("/")
    model = settings.vision_llm.model
    try:
        async with httpx.AsyncClient(timeout=settings.vision_llm.timeout_s) as client:
            resp = await client.post(
                f"{host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
            )
            if resp.status_code == 404:
                return SkillResult(
                    f"The vision model isn't installed — run: ollama pull {model}.",
                    success=False,
                )
            resp.raise_for_status()
            answer = (resp.json().get("response") or "").strip()
    except httpx.HTTPError:
        return SkillResult(
            "I can't reach my vision model right now — is Ollama running?",
            success=False,
        )
    return SkillResult(answer or "I couldn't make anything out.", data={"model": model})


class SeeCameraSkill(Skill):
    name = "see_camera"
    description = "Describe what the webcam currently sees."

    patterns = [
        re.compile(r"\bwhat\s+(?:do|can)\s+you\s+see\b", re.IGNORECASE),
        re.compile(r"\bdescribe\s+(?:what\s+you\s+see|the\s+(?:camera|room|view))\b", re.IGNORECASE),
        re.compile(r"\blook\s+(?:at\s+(?:me|this)|around)\b", re.IGNORECASE),
        re.compile(r"\bcan\s+you\s+see\s+me\b", re.IGNORECASE),
    ]

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def execute(self, request: SkillRequest) -> SkillResult:
        import httpx

        port = self._settings.vision.stream_port
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"http://127.0.0.1:{port}/frame.jpg")
                resp.raise_for_status()
                frame = resp.content
        except httpx.HTTPError:
            return SkillResult(
                "The camera isn't running — start the vision sidecar and try again.",
                success=False,
            )
        return await _describe(
            self._settings, base64.b64encode(frame).decode(), DESCRIBE_CAMERA_PROMPT
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

    patterns = [
        re.compile(r"\bwhat(?:'?s| is)\s+on\s+(?:my|the)\s+screen\b", re.IGNORECASE),
        re.compile(r"\b(?:read|describe)\s+(?:my|the)\s+screen\b", re.IGNORECASE),
        # "can you see my screen" / "look at my screen" / "what am I looking
        # at" went to the LLM, which would claim it can't see screens instead
        # of calling this tool — catch the common phrasings deterministically.
        re.compile(r"\bcan\s+you\s+see\s+(?:my|the)\s+screen\b", re.IGNORECASE),
        re.compile(r"\blook\s+at\s+(?:my|the)\s+screen\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+am\s+i\s+looking\s+at\b", re.IGNORECASE),
    ]

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def execute(self, request: SkillRequest) -> SkillResult:
        wants_read = (
            request.args.get("mode") == "read" or "read" in request.text.lower()
        )
        try:
            import pyautogui

            shot = await asyncio.to_thread(pyautogui.screenshot)
            buf = io.BytesIO()
            shot.save(buf, format="PNG")
            small = await asyncio.to_thread(_shrink, buf.getvalue())
        except Exception as exc:
            return SkillResult(f"I couldn't capture the screen: {exc}", success=False)
        prompt = READ_SCREEN_PROMPT if wants_read else DESCRIBE_SCREEN_PROMPT
        return await _describe(
            self._settings, base64.b64encode(small).decode(), prompt
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
        """(full-screen PIL image, cursor xy) — runs in a worker thread."""
        import pyautogui

        pos = pyautogui.position()
        return pyautogui.screenshot(), (int(pos.x), int(pos.y))

    async def _default_describe(self, image_b64: str) -> SkillResult:
        return await _describe(self._settings, image_b64, POINT_PROMPT)

    async def execute(self, request: SkillRequest) -> SkillResult:
        try:
            shot, (cx, cy) = await asyncio.to_thread(self._capture)
        except Exception as exc:
            return SkillResult(f"I couldn't capture the screen: {exc}", success=False)
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
