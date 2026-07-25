"""Tier-2 semantic router (M2.5b/c/d): matcher logic + router wiring.

Two layers are exercised here, kept apart on purpose:

* :class:`SkillRouteIndex.match` — the PURE cosine matcher. No embedder, no
  router: fabricate vectors and assert the threshold/margin gate declines an
  ambiguous or weak match and accepts a confident one.
* :class:`Router` semantic tier — the WIRING. A deterministic fake embedder
  stands in for Ollama so the test proves that a by-meaning hit dispatches the
  right skill in live mode, is logged-but-not-taken in shadow mode, and falls
  through to the LLM when no skill wins. Embedding QUALITY is the model's job
  and is deliberately not asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import load_settings
from core.events import EventBus, RoutePath
from core.route_index import RouteEntry, SkillRouteIndex
from core.router import OFFLINE_LLM_REPLY, Router
from llm.client import OllamaClient
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult


# --- pure matcher ------------------------------------------------------------

def _index(vectors: dict[str, list[float]]) -> SkillRouteIndex:
    idx = SkillRouteIndex(":memory:", None, "test")
    idx._entries = [
        RouteEntry(name, name, np.array(v, dtype=np.float32), 1)
        for name, v in vectors.items()
    ]
    return idx


def test_match_confident_winner():
    idx = _index({"weather": [1, 0, 0], "news": [0, 1, 0], "datetime": [0, 0, 1]})
    hit = idx.match(np.array([0.9, 0.1, 0.0], dtype=np.float32))
    assert hit is not None
    assert hit[0] == "weather"
    assert hit[1] > 0.9


def test_match_declines_ambiguous_by_margin():
    # Halfway between weather and news: both ~equal → declined (falls to LLM).
    idx = _index({"weather": [1, 0, 0], "news": [0, 1, 0]})
    assert idx.match(np.array([0.7, 0.7, 0.0], dtype=np.float32)) is None


def test_match_declines_below_threshold():
    idx = _index({"weather": [1, 0, 0], "news": [0, 1, 0]})
    # Threshold 0.9 with a query only ~0.71 similar to the winner → declined.
    assert idx.match(np.array([0.8, 0.6, 0.0], dtype=np.float32),
                     threshold=0.9) is None


def test_match_empty_index_or_zero_query():
    assert _index({}).match(np.array([1, 0, 0], dtype=np.float32)) is None
    idx = _index({"weather": [1, 0, 0]})
    assert idx.match(np.array([0, 0, 0], dtype=np.float32)) is None   # zero norm
    assert idx.match(None) is None


# --- router wiring -----------------------------------------------------------

class _FakeSkill(Skill):
    """Minimal text-only skill: matches no regex, answers from request.text."""

    patterns: list = []

    def __init__(self, name: str, phrases: list[str]) -> None:
        self.name = name
        self.description = f"{name} skill"
        self.routing_phrases = phrases

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult(speech=f"{self.name}:{request.text}", success=True)


def _fake_embedder(texts):
    """Deterministic stand-in for Ollama: keyed on a marker word so the test is
    about WIRING, not embedding quality. 'warm' → weather axis, 'headline' →
    news axis, neither → the [1,1,1] diagonal (equidistant → declined)."""
    out = []
    for t in texts:
        tl = t.lower()
        if "warm" in tl:
            out.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        elif "headline" in tl:
            out.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        else:
            out.append(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    return out


def _router(tmp_path, *, shadow: bool):
    settings = load_settings()
    settings.memory.db_path = str(tmp_path / "route.db")
    settings.router.semantic_enabled = True
    settings.router.semantic_shadow = shadow
    registry = SkillRegistry()
    registry.register(_FakeSkill("weather", ["how warm is it outside today"]))
    registry.register(_FakeSkill("news", ["what are today's top headlines"]))
    router = Router(settings, registry, OllamaClient(settings.llm), EventBus())
    router.model = None                 # no LLM: fall-through hits OFFLINE reply
    router._embedder = _fake_embedder   # inject deterministic embedder
    router._route_index = None          # force a rebuild with the fake
    return router


@pytest.mark.asyncio
async def test_semantic_live_dispatches_by_meaning(tmp_path):
    router = _router(tmp_path, shadow=False)
    result = await router.route("is it warm right now")   # no regex; 'warm' axis
    assert result.path is RoutePath.SEMANTIC
    assert result.skill_name == "weather"
    assert result.speech.startswith("weather:")
    # the KeyError-guard: SEMANTIC must be a counted route
    assert router.stats[RoutePath.SEMANTIC] == 1


@pytest.mark.asyncio
async def test_semantic_shadow_logs_but_uses_llm(tmp_path):
    router = _router(tmp_path, shadow=True)
    result = await router.route("is it warm right now")
    # Shadow: the would-be route is only logged; the turn still goes to the LLM
    # (offline here), so no skill ran and the path is LLM.
    assert result.path is RoutePath.LLM
    assert result.speech == OFFLINE_LLM_REPLY
    assert router.stats[RoutePath.SEMANTIC] == 0


@pytest.mark.asyncio
async def test_semantic_declines_and_falls_to_llm(tmp_path):
    router = _router(tmp_path, shadow=False)
    # No marker word → equidistant → declined even in live mode → LLM.
    result = await router.route("please recite a poem for me")
    assert result.path is RoutePath.LLM
    assert result.speech == OFFLINE_LLM_REPLY


@pytest.mark.asyncio
async def test_fast_path_still_wins_over_semantic(tmp_path):
    # A regex hit must short-circuit BEFORE the semantic tier is consulted.
    from skills.datetime_skill import DateTimeSkill

    router = _router(tmp_path, shadow=False)
    router._registry.register(DateTimeSkill())
    result = await router.route("what time is it")
    assert result.path is RoutePath.FAST
    assert result.skill_name == "datetime"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
