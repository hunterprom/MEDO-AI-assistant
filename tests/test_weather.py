"""The greedy city capture, and the compound question that exposed it.

From a live session:

    YOU   what's the weather like in skopje and while telling me the weather
          can you tell me what should i do because i'm pretty bored
    MEDO  I couldn't find a place called skopje and while telling me the
          weather can you tell me what should i do because i'm pretty bored.

``_CITY`` is ``[\\w .'-]+`` and runs to end-of-line, so every word of the second
request became part of the place name. The trim has to be conservative in the
other direction too: plenty of real places have "and" in the middle of them.
"""

from __future__ import annotations

import pytest

from skills.weather import _cut_at_clause


@pytest.mark.parametrize("captured,expected", [
    ("skopje and while telling me what should i do because i'm bored", "skopje"),
    ("skopje and can you tell me what to wear", "skopje"),
    ("skopje because i want to know", "skopje"),
    ("london but tell me the news too", "london"),
    ("скопје и кажи ми што да правам", "скопје"),
])
def test_a_second_request_is_trimmed_off_the_city(captured, expected):
    assert _cut_at_clause(captured) == expected


@pytest.mark.parametrize("place", [
    "bosnia and herzegovina",
    "trinidad and tobago",
    "newcastle upon tyne",
    "stoke-on-trent",
    "frankfurt am main",
    "skopje",
])
def test_a_real_place_name_is_left_whole(place):
    """The trim cuts only before something no place name contains — a pronoun,
    a modal, or a question word."""
    assert _cut_at_clause(place) == place


def test_a_plain_list_of_cities_is_not_cut():
    """Answering the first city beats answering neither; it is not a new clause."""
    assert _cut_at_clause("skopje and london") == "skopje and london"


def test_the_reported_utterance_asks_about_the_right_place():
    from core.config import load_settings
    from skills.weather import WeatherSkill

    skill = WeatherSkill(load_settings().weather)
    text = ("what's the weather like in skopje and while telling me the weather "
            "can you tell me what should i do because i'm pretty bored")
    match = skill.match(text)
    assert match is not None
    captured = (match.groupdict().get("city") or "").strip()
    assert _cut_at_clause(captured) == "skopje", captured


def test_the_wake_name_survives_being_heard_as_middle():
    """Whisper rendered "Hey, MEDO" as "Hey, Middle" in the same session.

    QuitSkill matches the name plus a verb, so a mishearing it doesn't know
    silently misses the fast path and reaches the LLM — which has, before,
    narrated a shutdown that never happened.
    """
    from skills.system import QuitSkill

    skill = QuitSkill()
    for heard in ("close medo", "close middle", "close medal", "quit meadow"):
        assert skill.match(heard) is not None, heard
