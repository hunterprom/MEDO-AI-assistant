"""M0 smoke tests: config loads, fast path routes without an LLM."""

from __future__ import annotations

import pytest

from core.config import load_settings
from core.events import EventBus, RoutePath
from core.router import OFFLINE_LLM_REPLY, Router
from llm.client import OllamaClient
from skills.base import SkillRegistry
from skills.datetime_skill import DateTimeSkill


def make_router(model: str | None = None) -> Router:
    settings = load_settings()
    registry = SkillRegistry()
    registry.register(DateTimeSkill())
    router = Router(settings, registry, OllamaClient(settings.llm), EventBus())
    router.model = model
    return router


def test_config_loads():
    settings = load_settings()
    assert settings.personality.name == "MEDO"
    assert settings.router.fast_path_enabled is True


@pytest.mark.asyncio
async def test_time_takes_fast_path():
    router = make_router()
    result = await router.route("what time is it")
    assert result.path is RoutePath.FAST
    assert result.skill_name == "datetime"
    assert "It's" in result.speech


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "utterance",
    ["what's the date today", "whats the date", "what is the date", "today's date"],
)
async def test_date_variants_take_fast_path(utterance):
    # Apostrophe-less "whats" (common from STT) must not fall through to the LLM,
    # which would hallucinate a date.
    router = make_router()
    result = await router.route(utterance)
    assert result.path is RoutePath.FAST
    assert result.data["kind"] == "date"


@pytest.mark.asyncio
async def test_unmatched_falls_to_llm_and_degrades_without_model():
    router = make_router(model=None)  # no model configured
    result = await router.route("tell me a joke about compilers")
    assert result.path is RoutePath.LLM
    assert result.speech == OFFLINE_LLM_REPLY
