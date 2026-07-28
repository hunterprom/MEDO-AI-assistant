"""Broadened natural-language triggers hit the right fast-path skill/action."""

from __future__ import annotations

import pytest

from core.config import load_settings
from skills.base import SkillRequest
from skills.media import MediaSkill
from skills.system import VolumeSkill
from skills.timers import TimerSkill, parse_duration
from skills.weather import WeatherSkill


def _hit(skill, q) -> bool:
    return bool(skill.match(q))


@pytest.mark.parametrize("q", [
    "do I need a jacket", "do I need an umbrella", "what's it like outside",
    "how's the weather", "how is it outside", "is it raining outside",
    "is it cold today",
])
def test_weather_natural_phrasings(q):
    assert _hit(WeatherSkill(load_settings().weather), q)


@pytest.mark.parametrize("q,secs", [
    ("wake me up in 20 minutes", 1200),
    ("wake me in an hour", 3600),
    ("ping me in 5 minutes", 300),
    ("set a countdown for 10 minutes", 600),
    ("nudge me in 2 minutes", 120),
])
def test_timer_natural_phrasings(q, secs):
    assert _hit(TimerSkill(lambda x: None), q)
    assert parse_duration(q) == secs


async def test_volume_natural_phrasings_route_to_the_right_action(monkeypatch):
    v = VolumeSkill()
    calls = []
    monkeypatch.setattr(v, "_step", lambda up: calls.append(("step", up)) or "ok")
    monkeypatch.setattr(v, "_set_absolute", lambda lvl: calls.append(("abs", lvl)) or "ok")

    async def route(q):
        calls.clear()
        await v.execute(SkillRequest(text=q, match=v.match(q)))
        return calls[0] if calls else None

    assert await route("it's too loud") == ("step", False)      # quieter
    assert await route("too quiet") == ("step", True)           # louder
    assert await route("I can't hear") == ("step", True)
    assert await route("speak up") == ("step", True)
    assert await route("max volume") == ("abs", 100)
    assert await route("full volume") == ("abs", 100)


async def test_media_put_on_music_is_playpause(monkeypatch):
    m = MediaSkill()
    monkeypatch.setattr(m, "_run", lambda action: action)  # skip OS media keys
    for q in ("put on some music", "turn on music"):
        r = await m.execute(SkillRequest(text=q, match=m.match(q)))
        assert r.data.get("action") == "playpause", q
