"""SmallTalk fast-path: greetings answer instantly, real questions don't match.

The load-bearing half is the ADVERSARIAL non-matches — a fast skill that hijacks
"how are you going to fix this?" would be worse than the latency it saves.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from core import mk
from skills.base import SkillRequest
from skills.smalltalk import SmallTalkSkill


def _skill():
    return SmallTalkSkill(rng=random.Random(0))


def _matches(text: str) -> bool:
    return _skill().match(text) is not None


def _run(text: str):
    s = _skill()
    return asyncio.run(s.execute(SkillRequest(text=text, match=s.match(text))))


# -- positives ----------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "how are you", "How are you?", "how are you doing", "how are you doing today",
    "how are you feeling", "how's it going", "how are things",
    "hi", "hello", "hey", "hi there", "hello medo", "howdy",
    "thanks", "thank you", "thanks a lot", "cheers", "much appreciated",
    "good night", "goodnight", "good night medo",
    "како си", "здраво", "фала", "благодарам", "добра ноќ",
    # the exact transcript that motivated this
    "Please speak in English. How are you?",
])
def test_small_talk_matches(text):
    assert _matches(text), text


# -- adversarial non-matches (must fall through to a real skill / the LLM) -----

@pytest.mark.parametrize("text", [
    "how are you going to fix this?",
    "how are you planning to handle the migration",
    "how are you able to see my screen",
    "thanks to the refactor the tests pass now",
    "thank you note generator",
    "hi there is a bug in the parser",
    "hello world program in python",
    "good night's sleep improves focus",
    "how are the servers doing",          # not "how are YOU"
    "yo what's the weather",
])
def test_small_talk_ignores_real_requests(text):
    assert not _matches(text), text


# -- responses ----------------------------------------------------------------

def test_how_are_you_answers_in_english():
    r = _run("How are you?")
    assert r.data["smalltalk"] == "how_are_you"
    assert r.speech and not mk.is_cyrillic(r.speech)


def test_macedonian_gets_a_macedonian_reply():
    r = _run("како си")
    assert r.data["smalltalk"] == "how_are_you" and mk.is_cyrillic(r.speech)


def test_categories_are_distinguished():
    assert _run("thanks").data["smalltalk"] == "thanks"
    assert _run("good night").data["smalltalk"] == "night"
    assert _run("hello").data["smalltalk"] == "greet"


def test_is_a_fast_answer_skill_not_actuation():
    s = SmallTalkSkill()
    assert s.controls_pc is False and not getattr(s, "capabilities", frozenset())


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
