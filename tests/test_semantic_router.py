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


def test_match_rejects_nonfinite_query_and_candidate_vectors():
    # A NaN/inf query used to slip past the `== 0.0` norm check and poison the
    # sort (every NaN comparison is False), so a real match could be missed or a
    # NaN handed back. It must decline instead.
    idx = _index({"weather": [1, 0, 0], "news": [0, 1, 0]})
    assert idx.match(np.array([np.nan, 0, 0], dtype=np.float32)) is None
    assert idx.match(np.array([np.inf, 0, 0], dtype=np.float32)) is None
    # A single corrupt CANDIDATE vector must be skipped, not sink the whole match:
    # the clean 'weather' entry still wins.
    idx2 = _index({"weather": [1, 0, 0], "broken": [np.nan, np.nan, np.nan]})
    hit = idx2.match(np.array([0.95, 0.05, 0.0], dtype=np.float32))
    assert hit is not None and hit[0] == "weather"


# --- router wiring -----------------------------------------------------------

class _FakeSkill(Skill):
    """Minimal text-only skill: matches no regex, answers from request.text."""

    patterns: list = []

    def __init__(self, name: str, phrases: list[str], *,
                 controls_pc: bool = False,
                 requires_confirmation: bool = False) -> None:
        self.name = name
        self.description = f"{name} skill"
        self.routing_phrases = phrases
        self.controls_pc = controls_pc
        self.requires_confirmation = requires_confirmation

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
async def test_cold_embedder_does_not_permanently_disable_the_tier(tmp_path):
    # A transient empty build (embed model cold on the first miss) must NOT latch
    # the index off for the whole session — the next miss, model now warm,
    # rebuilds and routes by meaning.
    calls = {"n": 0}

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] == 1:
            return None                      # cold on the first build -> empty index
        return _fake_embedder(texts)         # warm afterwards

    router = _router(tmp_path, shadow=False)
    router._embedder = flaky
    router._route_index = None
    r1 = await router.route("is it warm right now")
    assert r1.path is RoutePath.LLM          # tier declined (empty build), fell through
    assert router._route_index is None       # NOT latched to the permanent-off sentinel
    router._route_index_retry_at = None      # simulate the retry cooldown elapsing
    r2 = await router.route("is it warm right now")
    assert r2.path is RoutePath.SEMANTIC and r2.skill_name == "weather"


@pytest.mark.asyncio
async def test_semantic_never_dispatches_an_unsafe_skill(tmp_path):
    # The safety invariant, tested at the dispatch decision: even if a stale
    # cached vector for a PC-controlling skill were somehow present in the index
    # AND matched a query, the semantic tier must refuse to reach it by meaning
    # (a route with no regex and no confirmation prompt).
    from core.route_index import RouteEntry

    def emb(texts):
        out = []
        for t in texts:
            tl = t.lower()
            if "erase" in tl or "wipe" in tl:
                out.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            elif "warm" in tl:
                out.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
            else:
                out.append(np.array([1.0, 1.0, 1.0], dtype=np.float32))
        return out

    router = _router(tmp_path, shadow=False)
    router._embedder = emb
    router._route_index = None
    router._registry.register(
        _FakeSkill("wipe_disk", ["erase the whole disk now"], controls_pc=True))
    # _ensure_route_index excludes the controls_pc skill (it is not semantic-safe),
    # so smuggle a stale entry in as if a cached vector had outlived a flag flip.
    idx = await router._ensure_route_index()
    idx._entries.append(RouteEntry("wipe_disk", "erase", np.array([0, 0, 1], np.float32), 1))

    # The matcher WOULD pick wipe_disk (its own axis), but the guard rejects it.
    assert await router._semantic_route("please wipe it") is None
    result = await router.route("please wipe it")
    assert result.path is RoutePath.LLM          # fell through; nothing destructive ran


@pytest.mark.asyncio
async def test_fast_path_still_wins_over_semantic(tmp_path):
    # A regex hit must short-circuit BEFORE the semantic tier is consulted.
    from skills.datetime_skill import DateTimeSkill

    router = _router(tmp_path, shadow=False)
    router._registry.register(DateTimeSkill())
    result = await router.route("what time is it")
    assert result.path is RoutePath.FAST
    assert result.skill_name == "datetime"


# --- adaptive route memory (M2.5e) wiring ------------------------------------
# The store itself is unit-tested in test_route_memory.py. Here we prove the
# ROUTER: a learned exemplar shortcuts a phrase the curated tier declines, the
# shadow gate still applies to learned hits, learning is gated + safe, and the
# whole thing is inert when the flag is off. _fake_embedder maps a no-marker
# phrase to the [1,1,1] diagonal, which the curated tier declines (equidistant).

def _mem_router(tmp_path, *, shadow: bool):
    router = _router(tmp_path, shadow=shadow)
    router._settings.router.route_memory_enabled = True
    router._route_memory = None            # rebuild with the flag on
    return router


@pytest.mark.asyncio
async def test_learned_exemplar_shortcuts_a_declined_phrase(tmp_path):
    router = _mem_router(tmp_path, shadow=False)
    weather = router._registry.get("weather")
    # Teach it a phrase the CURATED tier can't route (fake embedder -> [1,1,1]).
    await router._learn_route("please do the weather thing", weather)
    result = await router.route("please do the weather thing")
    assert result.path is RoutePath.SEMANTIC
    assert result.skill_name == "weather"
    assert router.stats[RoutePath.SEMANTIC] == 1


@pytest.mark.asyncio
async def test_route_memory_bootstraps_from_an_empty_store(tmp_path):
    # Regression: an empty RouteMemory is falsy (it defines __len__), so the old
    # `return self._route_memory or None` collapsed a valid-but-empty store to
    # None and the FIRST exemplar could never be written. In production the first
    # _ensure_route_memory call comes from _semantic_route (empty store), then the
    # learn that follows must still write row 1 through the cached branch — the
    # ordering the earlier test front-loaded around.
    from core.router import RouteResult

    router = _mem_router(tmp_path, shadow=False)
    # 1) Production ordering: the semantic tier runs first and builds the (empty)
    #    store; it must not deadlock the feature by returning None thereafter.
    assert await router._semantic_route("please do the weather thing") is None
    # 2) The LLM then resolves it to one learnable skill -> learn on the empty store.
    res = RouteResult(path=RoutePath.LLM, speech="warm", skill_name="weather",
                      data={"tool_call_count": 1})
    await router._maybe_learn_route("please do the weather thing", res,
                                    answering_confirmation=False)
    assert len(router._route_memory) == 1          # first exemplar actually persisted
    # 3) ...so the same phrasing now shortcuts on the SEMANTIC path.
    result = await router.route("please do the weather thing")
    assert result.path is RoutePath.SEMANTIC and result.skill_name == "weather"


@pytest.mark.asyncio
async def test_shadow_gate_applies_to_learned_hits(tmp_path):
    router = _mem_router(tmp_path, shadow=True)
    await router._learn_route("please do the weather thing",
                              router._registry.get("weather"))
    result = await router.route("please do the weather thing")
    # Shadow: the learned route is logged only; the turn still goes to the LLM.
    assert result.path is RoutePath.LLM
    assert result.speech == OFFLINE_LLM_REPLY
    assert router.stats[RoutePath.SEMANTIC] == 0


@pytest.mark.asyncio
async def test_destructive_resolution_is_never_learned(tmp_path):
    from core.route_memory import RouteMemory
    from core.router import RouteResult

    router = _mem_router(tmp_path, shadow=False)
    router._registry.register(
        _FakeSkill("wipe_disk", ["erase everything"], controls_pc=True))
    res = RouteResult(path=RoutePath.LLM, speech="done", skill_name="wipe_disk",
                      data={"tool_call_count": 1})
    await router._maybe_learn_route("wipe the disk", res, answering_confirmation=False)
    # A PC-controlling skill must never become a learnable shortcut.
    store = RouteMemory(router._settings.memory.db_path,
                        router._settings.memory.embed_model or "none")
    assert len(store) == 0


@pytest.mark.asyncio
async def test_multi_tool_turn_is_not_learned(tmp_path):
    from core.route_memory import RouteMemory
    from core.router import RouteResult

    router = _mem_router(tmp_path, shadow=False)
    res = RouteResult(path=RoutePath.LLM, speech="a and b", skill_name="weather",
                      data={"tool_call_count": 2})           # ambiguous multi-tool
    await router._maybe_learn_route("do two things", res, answering_confirmation=False)
    store = RouteMemory(router._settings.memory.db_path,
                        router._settings.memory.embed_model or "none")
    assert len(store) == 0


@pytest.mark.asyncio
async def test_route_memory_inert_when_disabled(tmp_path):
    from core.route_memory import RouteMemory
    from core.router import RouteResult

    router = _router(tmp_path, shadow=False)                 # flag stays default False
    res = RouteResult(path=RoutePath.LLM, speech="warm", skill_name="weather",
                      data={"tool_call_count": 1})
    await router._maybe_learn_route("some new phrasing", res,
                                    answering_confirmation=False)
    store = RouteMemory(router._settings.memory.db_path,
                        router._settings.memory.embed_model or "none")
    assert len(store) == 0                                    # learning off -> nothing
    # And a never-seen phrase still falls straight through to the LLM.
    result = await router.route("some new phrasing")
    assert result.path is RoutePath.LLM
    assert result.speech == OFFLINE_LLM_REPLY


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
