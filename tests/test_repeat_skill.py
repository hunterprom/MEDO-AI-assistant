"""'Say that again' replays the last reply verbatim (never a re-generation)."""

from __future__ import annotations

import pytest

from core.config import load_settings
from core.events import EventBus, RoutePath
from core.router import Router
from llm.client import OllamaClient
from skills.base import SkillRegistry, SkillRequest
from skills.datetime_skill import DateTimeSkill
from skills.repeat_skill import RepeatSkill


def test_match_phrasings():
    s = RepeatSkill()
    for x in ("say that again", "repeat that", "repeat it", "what did you say",
              "what did you just say", "come again", "one more time",
              "can you repeat that", "повтори", "што рече"):
        assert s.match(x) is not None, x
    # a follow-up asking ABOUT something is not a verbatim replay
    assert s.match("what did you say about the weather") is None
    assert s.match("repeat the last step of the recipe") is None


@pytest.mark.asyncio
async def test_replays_verbatim_from_context():
    s = RepeatSkill()
    res = await s.execute(SkillRequest(
        text="say that again", context={"last_reply": "It's 3:15 PM."}))
    assert res.success
    assert res.speech == "It's 3:15 PM."          # exact, not paraphrased


@pytest.mark.asyncio
async def test_honest_when_nothing_said_yet():
    s = RepeatSkill()
    res = await s.execute(SkillRequest(text="repeat that", context={}))
    assert res.success is False
    assert "haven't said" in res.speech.lower()


@pytest.mark.asyncio
async def test_macedonian_repeat():
    s = RepeatSkill()
    res = await s.execute(SkillRequest(
        text="повтори", context={"last_reply": "Часот е 15:15."}))
    assert res.success
    assert res.speech == "Часот е 15:15."


@pytest.mark.asyncio
async def test_router_wires_last_reply_end_to_end():
    """The router must expose the PRIOR reply so 'repeat that' echoes it — this
    proves the context enrichment + the add_turn ordering."""
    settings = load_settings()
    registry = SkillRegistry()
    registry.register(DateTimeSkill())
    registry.register(RepeatSkill())
    router = Router(settings, registry, OllamaClient(settings.llm), EventBus())
    router.model = None

    first = await router.route("what time is it")
    assert first.path is RoutePath.FAST and first.skill_name == "datetime"

    again = await router.route("repeat that")
    assert again.path is RoutePath.FAST and again.skill_name == "repeat"
    assert again.speech == first.speech           # verbatim replay of the clock


@pytest.mark.asyncio
async def test_router_repeat_with_no_history():
    settings = load_settings()
    registry = SkillRegistry()
    registry.register(RepeatSkill())
    router = Router(settings, registry, OllamaClient(settings.llm), EventBus())
    router.model = None
    res = await router.route("say that again")
    assert res.skill_name == "repeat"
    assert "haven't said" in res.speech.lower()   # nothing said yet


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
