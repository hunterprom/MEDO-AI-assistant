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
        except Exception as exc:
            return SkillResult(f"I couldn't capture the screen: {exc}", success=False)
        prompt = READ_SCREEN_PROMPT if wants_read else DESCRIBE_SCREEN_PROMPT
        return await _describe(
            self._settings, base64.b64encode(buf.getvalue()).decode(), prompt
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
