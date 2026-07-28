"""M7 personality layer: quips are opt-in, bilingual, and never unsafe."""

from __future__ import annotations

import random
import re

import pytest

from core.config import PersonalityConfig, load_settings
from core.events import EventBus
from core.persona import CATEGORY_BY_SKILL, QUIPS, Persona
from core.router import Router
from llm.client import OllamaClient
from llm.prompts import system_prompt
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult


def _persona(rng_seed: int = 0, **overrides) -> Persona:
    cfg = PersonalityConfig(**overrides)
    return Persona(cfg, rng=random.Random(rng_seed))


# --- decoration rules --------------------------------------------------------


def test_wit_level_zero_never_decorates():
    p = _persona(wit_level=0.0)
    for _ in range(50):
        assert p.decorate("Timer set.", skill_name="timers",
                          user_text="set a timer") == "Timer set."


def test_minimal_style_never_decorates():
    p = _persona(style="minimal", wit_level=1.0)
    assert p.decorate("Timer set.", skill_name="timers",
                      user_text="set a timer") == "Timer set."


def test_wit_level_one_always_decorates_known_categories():
    p = _persona(wit_level=1.0)
    out = p.decorate("Timer set.", skill_name="timers", user_text="set a timer")
    assert out.startswith("Timer set.") and len(out) > len("Timer set.")
    quip = out[len("Timer set. "):]
    assert quip in QUIPS["timers"]["en"]


def test_unknown_skills_are_never_decorated():
    p = _persona(wit_level=1.0)
    for name in ("datetime", "weather", "recall_facts", "web_search"):
        assert name not in CATEGORY_BY_SKILL
        assert p.decorate("Reply.", skill_name=name, user_text="x") == "Reply."


def test_macedonian_input_gets_macedonian_quips():
    p = _persona(wit_level=1.0)
    out = p.decorate("Готово.", skill_name="volume", user_text="намали звук")
    quip = out[len("Готово. "):]
    assert quip in QUIPS["volume"]["mk"]
    assert re.search(r"[Ѐ-ӿ]", quip)


def test_pinned_quips_language_overrides_match():
    p = _persona(wit_level=1.0, quips_language="en")
    out = p.decorate("Готово.", skill_name="volume", user_text="намали звук")
    assert out[len("Готово. "):] in QUIPS["volume"]["en"]


# --- LLM prompt fragment -----------------------------------------------------


def test_prompt_fragment_per_style_and_size():
    seen = set()
    for style in ("dry_wit", "professional", "minimal"):
        frag = _persona(style=style).prompt_fragment()
        seen.add(frag)
        assert 0 < frag.count(".") <= 3  # 2-3 sentences max — num_ctx is tight
    assert len(seen) == 3  # styles actually differ


def test_system_prompt_carries_the_fragment():
    cfg = PersonalityConfig(style="professional")
    assert "strictly professional" in system_prompt(cfg)
    assert "strictly professional" not in system_prompt(PersonalityConfig())


# --- safety: confirmations and destructive actions stay literal --------------


class _DangerSkill(Skill):
    name = "power"  # a category with quips — must STILL stay literal
    description = "test-only destructive skill"
    patterns = [re.compile(r"\bshut everything down\b", re.IGNORECASE)]
    requires_confirmation = True

    async def execute(self, request: SkillRequest) -> SkillResult:
        if not request.context.get("confirmed"):
            return SkillResult("Are you sure?", needs_confirmation=True)
        return SkillResult("Done.")


class _TimerSkill(Skill):
    name = "timers"
    description = "test-only timer skill"
    patterns = [re.compile(r"\bset a timer\b", re.IGNORECASE)]

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("Timer set.")


class _FailingTimerSkill(_TimerSkill):
    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("I couldn't set that timer.", success=False)


def _router(skill: Skill) -> Router:
    settings = load_settings()
    settings.personality.wit_level = 1.0  # quip on EVERY eligible reply
    registry = SkillRegistry()
    registry.register(skill)
    router = Router(settings, registry, OllamaClient(settings.llm), EventBus())
    router._persona = Persona(settings.personality, rng=random.Random(7))
    return router


@pytest.mark.asyncio
async def test_confirmation_prompt_and_confirmed_action_stay_literal():
    router = _router(_DangerSkill())
    r1 = await router.route("shut everything down")
    assert r1.speech == "Are you sure?"          # safety prompt: no quip
    r2 = await router.route("yes")
    assert r2.speech == "Done."                  # destructive outcome: no quip


@pytest.mark.asyncio
async def test_errors_stay_literal():
    router = _router(_FailingTimerSkill())
    r = await router.route("set a timer")
    assert r.speech == "I couldn't set that timer."


@pytest.mark.asyncio
async def test_successful_fast_path_gets_the_quip():
    router = _router(_TimerSkill())
    r = await router.route("set a timer")
    assert r.speech.startswith("Timer set.")
    assert r.speech[len("Timer set. "):] in QUIPS["timers"]["en"]
