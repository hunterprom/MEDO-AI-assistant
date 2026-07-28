"""M2.5a — the skill embedding index and its cache.

No Ollama here: a fake embedder returns deterministic vectors, so the tests pin
the CACHING and INVALIDATION behaviour (the whole point of the index) rather
than the embedding model.
"""

from __future__ import annotations

import numpy as np

from core.route_index import SkillRouteIndex, cache_key
from skills.base import Skill, SkillRequest, SkillResult


class _Sk(Skill):
    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("ok")


def make(name, phrases=None, description="does a thing"):
    cls = type(f"S_{name}", (_Sk,), {
        "name": name, "description": description,
        "routing_phrases": list(phrases or [])})
    return cls()


class CountingEmbedder:
    """Deterministic 8-dim vectors; counts how many strings it was asked to embed."""

    def __init__(self):
        self.calls = 0
        self.embedded = 0

    def __call__(self, texts):
        self.calls += 1
        self.embedded += len(texts)
        out = []
        for t in texts:
            v = np.zeros(8, dtype="float32")
            for ch in t.lower():
                v[ord(ch) % 8] += 1.0
            out.append(v)
        return out


# -- routing surface -----------------------------------------------------------

def test_routing_surface_prefers_phrases_then_description():
    s = make("weather", ["will it rain", "do I need a coat"], "weather report")
    assert s.routing_surface() == ["will it rain", "do I need a coat", "weather report"]
    assert "will it rain" in s.routing_text() and "weather report" in s.routing_text()


def test_routing_surface_falls_back_to_description_then_name():
    assert make("news", [], "the headlines").routing_surface() == ["the headlines"]
    bare = make("odd_skill", [], "")
    assert bare.routing_surface() == ["odd skill"]      # name, underscores spaced


# -- cache key -----------------------------------------------------------------

def test_cache_key_changes_with_each_input():
    a = cache_key("weather", "will it rain", "nomic")
    assert a == cache_key("weather", "will it rain", "nomic")      # stable
    assert a != cache_key("weather", "will it POUR", "nomic")      # text changed
    assert a != cache_key("news", "will it rain", "nomic")         # skill changed
    assert a != cache_key("weather", "will it rain", "other-model")  # embedder changed


def test_cache_key_has_no_boundary_collisions():
    assert cache_key("ab", "c", "x") != cache_key("a", "bc", "x")


# -- build + cache -------------------------------------------------------------

def test_build_embeds_every_skill_first_time(tmp_path):
    emb = CountingEmbedder()
    idx = SkillRouteIndex(tmp_path / "r.db", emb, "fake")
    skills = [make("weather", ["will it rain"]), make("news", ["headlines"])]
    report = idx.build(skills)
    assert report.enabled and report.fresh == 2 and report.cached == 0
    assert emb.embedded == 2
    assert len(idx) == 2
    assert {e.skill for e in idx.entries()} == {"weather", "news"}


def test_rebuild_is_a_cache_hit_and_embeds_nothing(tmp_path):
    emb = CountingEmbedder()
    skills = [make("weather", ["will it rain"]), make("news", ["headlines"])]
    SkillRouteIndex(tmp_path / "r.db", emb, "fake").build(skills)
    embedded_after_first = emb.embedded

    emb2 = CountingEmbedder()
    idx2 = SkillRouteIndex(tmp_path / "r.db", emb2, "fake")   # same db file
    report = idx2.build(skills)
    assert report.cached == 2 and report.fresh == 0
    assert emb2.embedded == 0                    # nothing re-embedded
    assert embedded_after_first == 2
    assert len(idx2) == 2                         # still fully populated from cache


def test_changing_a_skills_text_reembeds_only_it(tmp_path):
    emb = CountingEmbedder()
    idx = SkillRouteIndex(tmp_path / "r.db", emb, "fake")
    idx.build([make("weather", ["will it rain"]), make("news", ["headlines"])])

    emb2 = CountingEmbedder()
    idx2 = SkillRouteIndex(tmp_path / "r.db", emb2, "fake")
    report = idx2.build([make("weather", ["will it POUR"]),   # changed
                         make("news", ["headlines"])])         # unchanged
    assert report.fresh == 1 and report.cached == 1
    assert emb2.embedded == 1
    fresh = next(r for r in report.rows if r.status == "fresh")
    assert fresh.skill == "weather"


def test_changing_the_embedder_invalidates_everything(tmp_path):
    emb = CountingEmbedder()
    skills = [make("weather", ["will it rain"]), make("news", ["headlines"])]
    SkillRouteIndex(tmp_path / "r.db", emb, "nomic").build(skills)

    emb2 = CountingEmbedder()
    report = SkillRouteIndex(tmp_path / "r.db", emb2, "other-model").build(skills)
    assert report.fresh == 2 and report.cached == 0     # different vector space


# -- embedder unavailable ------------------------------------------------------

def test_no_embedder_skips_without_crashing(tmp_path):
    idx = SkillRouteIndex(tmp_path / "r.db", None, "none")
    report = idx.build([make("weather", ["will it rain"])])
    assert report.enabled is False and report.skipped == 1 and len(idx) == 0


def test_embedder_returning_none_is_survived(tmp_path):
    idx = SkillRouteIndex(tmp_path / "r.db", lambda texts: None, "fake")
    report = idx.build([make("weather", ["will it rain"])])
    assert report.skipped == 1 and len(idx) == 0


# -- pruning + open-ness -------------------------------------------------------

def test_prune_drops_removed_skills(tmp_path):
    emb = CountingEmbedder()
    idx = SkillRouteIndex(tmp_path / "r.db", emb, "fake")
    idx.build([make("weather", ["rain"]), make("news", ["headlines"])])
    removed = idx.prune(["weather"])              # news deleted from config
    assert removed == 1
    # a later build with only weather sees just its own cached row
    idx2 = SkillRouteIndex(tmp_path / "r.db", CountingEmbedder(), "fake")
    report = idx2.build([make("weather", ["rain"])])
    assert report.cached == 1 and len(report.rows) == 1


def test_index_accepts_any_routing_source(tmp_path):
    # Not a Skill — just something with .name + .routing_text() (+ optional
    # routing_surface). Proves plugins / MEDO Link caps can register later.
    class Cap:
        name = "device_lamp"
        def routing_text(self): return "turn on the lamp · living room light"
        def routing_surface(self): return ["turn on the lamp", "living room light"]
    idx = SkillRouteIndex(tmp_path / "r.db", CountingEmbedder(), "fake")
    report = idx.build([Cap()])
    assert report.fresh == 1 and idx.entries()[0].skill == "device_lamp"
    assert report.rows[0].phrases == 2


# -- the real registry ---------------------------------------------------------

def test_the_live_registry_produces_routing_text_for_every_skill():
    from core.config import load_settings
    from core.docindex import DocumentIndex
    from main import Announcer, build_registry

    reg = build_registry(load_settings(), Announcer(),
                         doc_index=DocumentIndex(":memory:", None, []))
    for s in reg.all():
        assert s.routing_text().strip(), s.name        # every skill has a surface
