"""Seeing skills degrade gracefully when the sidecar / Ollama are missing."""

from __future__ import annotations

from core.config import Settings
from skills.base import SkillRequest
from skills.vision_skill import SeeCameraSkill, _describe


def _settings_with_dead_ports() -> Settings:
    return Settings(
        vision={"stream_port": 59852},          # nothing listens here
        llm={"host": "http://127.0.0.1:59853"},  # nor here
    )


async def test_camera_skill_graceful_when_sidecar_down():
    skill = SeeCameraSkill(_settings_with_dead_ports())
    r = await skill.execute(SkillRequest(text="what do you see"))
    assert r.success is False
    assert "sidecar" in r.speech.lower() or "camera" in r.speech.lower()


async def test_describe_graceful_when_ollama_down():
    r = await _describe(_settings_with_dead_ports(), "aGVsbG8=", "describe")
    assert r.success is False
    assert "ollama" in r.speech.lower() or "vision model" in r.speech.lower()
