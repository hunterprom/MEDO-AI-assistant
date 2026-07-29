"""FAST-tier hallucination guards: everyday speech must not trigger skills.

A hallucination-audit pass found three skills whose bare keyword patterns fired
on unrelated utterances — the destructive PowerSkill on "I need some sleep", the
WeatherSkill on "sales forecast", the SystemInfoSkill on "improve my memory".
These lock in that the tightened patterns reject the false positives while still
matching the real commands.
"""

from __future__ import annotations

import pytest

from skills.lion import LionModeSkill
from skills.system import PowerSkill, SystemInfoSkill
from skills.weather import WeatherSkill, _looks_like_city


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
    "reboot", "shut down", "shutdown", "shutdown the pc", "lock the screen",
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
    "free memory", "available memory", "how much free memory do I have",
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
    "what's the temperature tomorrow", "what will the temperature be tomorrow",
    "what's the temperature for tomorrow",
    "how hot is it", "how cold will it be", "what's the forecast",
    "weather forecast",
])
def test_weather_still_matches_real_questions(text):
    assert _matches(WeatherSkill, text)


# --- Round-3 audit regressions -------------------------------------------

@pytest.mark.parametrize("text", [
    "кое е најдоброто место",           # "which is the best place" — bare место
    "нема доволно место во собата",     # "not enough room in the room"
    "open my google drive",             # bare "drive" means Google Drive, not disk
    "take me for a drive",
])
def test_systeminfo_ignores_bare_place_and_drive(text):
    assert not _matches(SystemInfoSkill, text)


@pytest.mark.parametrize("text", [
    "колку простор имам",               # framed MK disk question
    "слободен простор на дискот",
    "how much disk space", "hard drive", "storage usage",
])
def test_systeminfo_still_matches_framed_disk(text):
    assert _matches(SystemInfoSkill, text)


@pytest.mark.parametrize("candidate,ok", [
    ("work", False), ("home", False), ("next quarter", False),
    ("the office", False), ("my house", False),
    ("London", True), ("New York", True), ("Скопје", True),
])
def test_weather_nonplace_capture_falls_back(candidate, ok):
    assert _looks_like_city(candidate) is ok


@pytest.mark.parametrize("candidate", ["The Hague", "the Netherlands", "London"])
def test_weather_group_path_keeps_determiner_led_real_cities(candidate):
    # The captured "in/for/at <city>" group frames a proper name, so a leading
    # article must NOT drop a real city (regression guard for the group path).
    assert _looks_like_city(candidate, allow_article=True) is True


@pytest.mark.parametrize("candidate", ["work", "my house", "next quarter",
                                       "the office", "the gym"])
def test_weather_group_path_still_drops_nonplaces(candidate):
    # "the office"/"the gym" mean the current location, not a city to geocode:
    # a leading article is stripped before the non-place check.
    assert _looks_like_city(candidate, allow_article=True) is False


def _lion_dir(text):
    m = next((p.search(text) for p in LionModeSkill.patterns if p.search(text)), None)
    assert m is not None, text
    gd = m.groupdict()
    return bool(gd.get("on") or gd.get("on2")), bool(gd.get("off") or gd.get("off2"))


@pytest.mark.parametrize("text,want_on", [
    ("enable lion mode", True), ("activate lion mode", True),
    ("disable lion mode", False), ("deactivate lion mode", False),
])
def test_lion_reversed_order_keeps_direction(text, want_on):
    on, off = _lion_dir(text)
    # Reversed word order ("enable lion mode") must set a direction, not fall
    # through to the bare-"lion mode" toggle branch.
    assert on is want_on and off is (not want_on)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
