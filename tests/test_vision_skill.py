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


def test_vision_patterns_catch_can_you_see_phrasings():
    """Phrasings that used to reach the LLM, which then claimed blindness."""
    from skills.vision_skill import SeeScreenSkill

    cam = SeeCameraSkill(_settings_with_dead_ports())
    scr = SeeScreenSkill(_settings_with_dead_ports())
    assert cam.match("can you see me?") is not None
    for phrase in ("can you see my screen", "look at my screen",
                   "what am I looking at?"):
        assert scr.match(phrase) is not None, phrase
    # The generic pronoun question is NOT stolen (see_bench owns "што е ова").
    assert scr.match("what is this") is None


def test_system_prompt_forbids_claiming_blindness():
    from core.config import PersonalityConfig
    from llm.prompts import system_prompt

    prompt = system_prompt(PersonalityConfig())
    assert "see_camera" in prompt and "see_screen" in prompt
    assert "Never claim you lack cameras" in prompt
