"""Resistor colour-band decoder — pure arithmetic, tested with no network/brain.

This is the feature the feature-council workflow picked to build: a deterministic
answer to "what does brown black red gold mean" (which fell to the LLM before,
and a small model swaps the multiplier and tolerance bands).
"""

from __future__ import annotations

import pytest

from core.router import Router
from skills.base import SkillRequest
from skills.resistor import (
    ResistorSkill,
    decode_bands,
    human_ohms,
    parse_ohms,
    value_to_bands,
)


# --- pure functions ----------------------------------------------------------

def test_decode_four_band():
    assert decode_bands(["brown", "black", "red", "gold"]) == (1000.0, 5.0)
    assert decode_bands(["yellow", "violet", "orange"])[0] == 47000.0
    assert decode_bands(["brown", "black", "orange", "gold"]) == (10000.0, 5.0)


def test_decode_three_band_has_no_tolerance():
    ohms, tol = decode_bands(["red", "red", "brown"])
    assert ohms == 220.0
    assert tol is None                       # 3 bands -> 20% (None marker)


def test_decode_five_band():
    # 3 digits + multiplier + tolerance
    ohms, tol = decode_bands(["red", "red", "black", "red", "brown"])
    assert ohms == 22000.0
    assert tol == 1.0


def test_decode_rejects_bad_input():
    with pytest.raises(ValueError):
        decode_bands(["brown", "black"])              # too few
    with pytest.raises(ValueError):
        decode_bands(["mauve", "black", "red"])       # not a colour


def test_value_to_bands():
    assert value_to_bands(4700)[:3] == ["yellow", "violet", "red"]
    assert value_to_bands(220)[:3] == ["red", "red", "brown"]
    assert value_to_bands(1_000_000)[:3] == ["brown", "black", "green"]
    assert value_to_bands(4700)[-1] == "gold"         # 5% tolerance band


def test_human_ohms():
    assert human_ohms(4700) == "4.7k"
    assert human_ohms(1000) == "1k"
    assert human_ohms(470000) == "470k"
    assert human_ohms(1_000_000) == "1M"
    assert human_ohms(220) == "220"


def test_parse_ohms():
    assert parse_ohms("what colour bands for a 4.7k resistor") == 4700.0
    assert parse_ohms("colours for 220 ohms") == 220.0
    assert parse_ohms("10k") == 10000.0
    assert parse_ohms("1M resistor") == 1_000_000.0
    assert parse_ohms("brown black red") is None      # no number


def test_decode_then_encode_round_trips():
    ohms, _ = decode_bands(["yellow", "violet", "red", "gold"])   # 4.7k
    assert value_to_bands(ohms)[:3] == ["yellow", "violet", "red"]


# --- the skill ---------------------------------------------------------------

def test_matches_a_bare_colour_run():
    s = ResistorSkill()
    assert s.match("what does brown black red gold mean") is not None
    assert s.match("colour bands for a 4.7k resistor") is not None
    assert s.match("what's the weather like") is None     # not a resistor query


def test_is_semantically_reachable():
    assert ResistorSkill().routing_phrases
    assert Router._semantic_safe(ResistorSkill()) is True


async def _speak(skill, text):
    res = await skill.execute(SkillRequest(text=text, match=skill.match(text)))
    return res


@pytest.mark.asyncio
async def test_decode_direction_speaks_the_value():
    s = ResistorSkill()
    res = await _speak(s, "what does brown black red gold mean")
    assert res.success
    assert "1k ohms" in res.speech
    assert "5 percent" in res.speech


@pytest.mark.asyncio
async def test_value_direction_speaks_the_bands():
    s = ResistorSkill()
    res = await _speak(s, "what colour bands for a 4.7k resistor")
    assert res.success
    for band in ("yellow", "violet", "red"):
        assert band in res.speech
    res2 = await _speak(s, "resistor colours for 220 ohms")
    assert res2.success and "red" in res2.speech


@pytest.mark.asyncio
async def test_macedonian():
    s = ResistorSkill()
    res = await _speak(s, "кои бои се за отпорник од 10k")
    assert res.success
    assert "кафеава" in res.speech          # 10k -> brown black orange gold


@pytest.mark.asyncio
async def test_graceful_when_underspecified():
    s = ResistorSkill()
    res = await s.execute(SkillRequest(text="brown black"))   # only 2 bands, no value
    assert res.success is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
