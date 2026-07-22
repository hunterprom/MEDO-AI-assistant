"""Seeing skills degrade gracefully when the sidecar / Ollama are missing."""

from __future__ import annotations

import pytest

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


def test_camera_and_screen_queries_route_to_the_right_skill():
    """Camera vs screen must not collide — and a 'you->yuo' STT mishear or an
    'on your camera' suffix must still hit the camera skill (not fall to the
    LLM, which then parrots stale answers)."""
    cam = SeeCameraSkill(_settings_with_dead_ports())
    scr = SeeScreenSkill(_settings_with_dead_ports())

    def routes_to(q):
        c, s = cam.match(q) is not None, scr.match(q) is not None
        return "cam" if c and not s else "scr" if s and not c else "?"

    for q in ("what do you see", "what do yuo see on your camera",
              "what's on the webcam", "look at the camera", "use your camera",
              "can you see anything"):
        assert routes_to(q) == "cam", q
    for q in ("what do you see on the screen", "read my screen",
              "what's on my screen"):
        assert routes_to(q) == "scr", q


def test_system_prompt_forbids_claiming_blindness():
    from core.config import PersonalityConfig
    from llm.prompts import system_prompt

    prompt = system_prompt(PersonalityConfig())
    assert "see_camera" in prompt and "see_screen" in prompt
    assert "Never claim you lack cameras" in prompt


def test_shrink_downscales_big_images_and_passes_junk_through():
    import io as _io

    from PIL import Image

    from skills.vision_skill import _shrink

    big = _io.BytesIO()
    Image.new("RGB", (2560, 1440), "white").save(big, format="PNG")
    small = Image.open(_io.BytesIO(_shrink(big.getvalue())))
    assert max(small.size) == 1280 and small.size == (1280, 720)

    tiny = _io.BytesIO()
    Image.new("RGB", (640, 480)).save(tiny, format="PNG")
    assert _shrink(tiny.getvalue()) == tiny.getvalue()  # small stays untouched
    assert _shrink(b"not an image") == b"not an image"  # junk passes through


# --- M11 deictic pointing ----------------------------------------------------


def test_crop_box_centers_and_clamps():
    from skills.vision_skill import crop_box

    assert crop_box(1280, 720, 2560, 1440) == (1040, 480, 1520, 960)  # centered
    assert crop_box(0, 0, 2560, 1440) == (0, 0, 480, 480)             # corner
    assert crop_box(2560, 1440, 2560, 1440) == (2080, 960, 2560, 1440)
    assert crop_box(50, 50, 320, 240) == (0, 0, 240, 240)  # tiny screen shrinks


def test_point_at_patterns_are_anchored():
    from skills.vision_skill import PointAtSkill

    s = PointAtSkill(_settings_with_dead_ports())
    for phrase in ("what is this?", "What's this", "what is that",
                   "what am I pointing at", "what's it under my cursor"):
        assert s.match(phrase) is not None, phrase
    # Richer questions stay on the LLM path; bench keeps its Cyrillic trigger.
    for phrase in ("what is this song", "what is this file about",
                   "што е ова"):
        assert s.match(phrase) is None, phrase


@pytest.mark.asyncio
async def test_point_at_crops_around_cursor_and_describes():
    from PIL import Image

    from skills.base import SkillRequest
    from skills.vision_skill import PointAtSkill, SkillResult

    seen = {}

    def capture():
        return Image.new("RGB", (2560, 1440), "white"), (100, 100)

    async def describe(image_b64: str) -> SkillResult:
        import base64 as b64
        import io as _io

        img = Image.open(_io.BytesIO(b64.b64decode(image_b64)))
        seen["size"] = img.size
        return SkillResult("A settings icon.")

    skill = PointAtSkill(_settings_with_dead_ports(), capture=capture,
                         describe=describe)
    r = await skill.execute(SkillRequest(text="what is this?"))
    assert r.success and r.speech == "A settings icon."
    assert seen["size"] == (480, 480)          # exactly the cursor crop
    assert r.data["cursor"] == [100, 100]
