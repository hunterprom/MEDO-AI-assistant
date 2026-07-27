"""FAST-tier hallucination guards: everyday speech must not trigger skills.

A hallucination-audit pass found three skills whose bare keyword patterns fired
on unrelated utterances — the destructive PowerSkill on "I need some sleep", the
WeatherSkill on "sales forecast", the SystemInfoSkill on "improve my memory".
These lock in that the tightened patterns reject the false positives while still
matching the real commands.
"""

from __future__ import annotations

import pytest

from skills.system import PowerSkill, SystemInfoSkill
from skills.weather import WeatherSkill


def _matches(skill_cls, text: str) -> bool:
    return any(p.search(text) for p in skill_cls.patterns)


@pytest.mark.parametrize("text", [
    "I need to get some sleep tonight",
    "let's restart the discussion",
    "I had to restart the game",
    "shut off the music",
])
def test_power_ignores_everyday_speech(text):
    assert not _matches(PowerSkill, text)


@pytest.mark.parametrize("text", [
    "put the computer to sleep", "sleep the pc", "restart the computer",
    "reboot", "shut down", "lock the screen",
])
def test_power_still_matches_real_commands(text):
    assert _matches(PowerSkill, text)


@pytest.mark.parametrize("text", [
    "how do I improve my memory",
    "the battery in my car is dead",
])
def test_systeminfo_ignores_unrelated(text):
    assert not _matches(SystemInfoSkill, text)


@pytest.mark.parametrize("text", [
    "how much battery do I have", "what's my battery percentage",
    "how much memory am I using", "what's my cpu usage", "system memory",
])
def test_systeminfo_still_matches_real_questions(text):
    assert _matches(SystemInfoSkill, text)


@pytest.mark.parametrize("text", [
    "what temperature should I cook chicken at",
    "what's my GPU temperature",
    "sales forecast for Q3",
    "what's the sales forecast",
])
def test_weather_ignores_non_weather(text):
    assert not _matches(WeatherSkill, text)


@pytest.mark.parametrize("text", [
    "what's the temperature", "what's the temperature outside",
    "how hot is it", "how cold will it be", "what's the forecast",
    "weather forecast",
])
def test_weather_still_matches_real_questions(text):
    assert _matches(WeatherSkill, text)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
