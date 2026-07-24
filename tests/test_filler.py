"""'Let me think' fillers for slow LLM answers (core/filler.py) + the router
hook that makes them LLM-path-only.

The feature: bridge dead air while a slow model composes. A big question gets a
quick acknowledgement; anything still silent after the delay gets one. It must
NEVER fire on the fast path — which is guaranteed by arming on the router's
on_llm_start hook, tested here.
"""

from __future__ import annotations

import random

import pytest

from core.config import FillerConfig, load_settings
from core.filler import Filler


def _filler(active=None, primary="en", **cfg):
    return Filler(FillerConfig(**cfg), active=active or ["en", "mk"],
                  primary=primary, rng=random.Random(0))


# --- big-question detection --------------------------------------------------

@pytest.mark.parametrize("text", [
    "explain recursion",
    "why is the sky blue",
    "tell me about the odyssey",
    "compare python and rust",
    "објасни ми го ова",
    "this question has quite a lot of words in it so it counts as big for sure",
])
def test_big_questions_are_recognised(text):
    assert _filler().is_big_question(text) is True


@pytest.mark.parametrize("text", [
    "what time is it", "turn on the lights", "open notepad", "set a timer", "",
])
def test_small_commands_are_not_big(text):
    assert _filler().is_big_question(text) is False


def test_big_question_gets_the_quick_delay():
    f = _filler(delay_s=8.0, big_delay_s=2.0)
    assert f.opening_delay("explain quantum tunnelling") == 2.0
    assert f.opening_delay("what time is it") == 8.0


def test_word_count_threshold_is_configurable():
    f = _filler(big_question_words=4)
    assert f.is_big_question("one two three four") is True
    assert f.is_big_question("one two three") is False


# --- phrase selection --------------------------------------------------------

def test_opening_and_waiting_come_from_the_language_bank():
    f = _filler(active=["en", "mk"], primary="en")
    assert f.opening("en") in _bank("en")[0]
    assert f.waiting("en") in _bank("en")[1]
    assert f.opening("mk") in _bank("mk")[0]


def test_non_active_language_degrades_to_primary():
    f = _filler(active=["en"], primary="en")
    # a Macedonian turn, but mk isn't active -> speak the primary (en) filler
    assert f.opening("mk") in _bank("en")[0]


def test_disabled_config_reports_not_enabled():
    assert _filler(enabled=False).enabled is False
    assert _filler(enabled=True).enabled is True


def _bank(code):
    from core.filler import _read_bank
    return _read_bank(code)


# --- the LLM-only guarantee (router hook) ------------------------------------

@pytest.mark.asyncio
async def test_on_llm_start_fires_on_llm_path_not_fast_path():
    from core.events import EventBus
    from core.router import Router
    from llm.client import OllamaClient
    from skills.base import SkillRegistry
    from skills.datetime_skill import DateTimeSkill

    settings = load_settings()
    registry = SkillRegistry()
    registry.register(DateTimeSkill())
    router = Router(settings, registry, OllamaClient(settings.llm), EventBus())
    router.model = None
    fired: list[str] = []

    # Fast path (datetime) must NOT arm the filler.
    from core.events import RoutePath
    r1 = await router.route("what time is it",
                            on_llm_start=lambda: fired.append("x"))
    assert r1.path is RoutePath.FAST
    assert fired == []

    # An unmatched utterance takes the LLM path -> the hook fires (before the
    # actual model call, so a filler armed on it is LLM-only by construction).
    await router.route("tell me a long story about compilers",
                       on_llm_start=lambda: fired.append("llm"))
    assert fired == ["llm"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
