"""P1-1: the council picks experts by meaning, with the word lists as backstop.

The trigger-word lists have a failure mode that no amount of tuning fixes: a
question about a topic nobody thought to write down reaches nobody, and a
specialist added through ``council.extra`` has no list at all — they could only
ever be summoned by name. Embedding each role description fixes both, and these
tests pin the three properties that make the change safe to ship:

* meaning wins when it is confident,
* the word list still decides when embeddings are unavailable or unsure,
* and a user-added specialist works with no keywords whatsoever.

The embedder is faked throughout. What is under test is the selection rule, not
whether ``nomic-embed-text`` has opinions about resistors.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import load_settings
from core.council import (
    COUNCIL,
    Specialist,
    load_council,
    rank_specialists,
    rank_specialists_semantic,
    trigger_hits,
)
from core.council_index import CouncilIndex
from skills.base import SkillRequest


def _spec(key: str, triggers: tuple[str, ...] = ()) -> Specialist:
    return Specialist(key, f"the {key}", f"You are a {key}.", triggers)


# --- the specialist's routing surface ----------------------------------------


def test_a_specialist_exposes_the_routing_source_contract():
    """``.name`` + ``.routing_text()`` is what core/route_index.py consumes."""
    member = COUNCIL[0]
    assert member.name == member.key
    assert isinstance(member.routing_text(), str)
    assert member.routing_text()


def test_the_role_description_is_what_gets_embedded():
    member = _spec("welder", ("weld", "arc"))
    surface = member.routing_surface()
    assert "You are a welder." in surface
    assert "welder" in surface[0]
    assert any("weld" in part for part in surface)


def test_a_specialist_with_no_triggers_still_has_a_surface():
    """The whole point: an extra specialist arrives with no keyword list."""
    member = _spec("astrologer")
    assert member.routing_text().strip()


# --- the selection rule (pure) -----------------------------------------------


def test_meaning_picks_the_expert_no_keyword_would_have():
    """"Will this melt?" reaches materials without "melt" being in any list."""
    council = (_spec("materials", ("alloy",)), _spec("law", ("contract",)))
    picked, method = rank_specialists_semantic(
        "will this bracket survive 200 degrees", council,
        {"materials": 0.71, "law": 0.20})
    assert method == "semantic"
    assert [m.key for m in picked] == ["materials"]


def test_nobody_over_the_threshold_falls_back_to_the_word_list():
    council = (_spec("materials", ("alloy",)), _spec("law", ("contract",)))
    picked, method = rank_specialists_semantic(
        "is this contract fair", council,
        {"materials": 0.11, "law": 0.19}, threshold=0.45)
    assert method == "wordlist"
    assert [m.key for m in picked] == ["law"]


def test_no_scores_at_all_falls_back_to_the_word_list():
    council = (_spec("materials", ("alloy",)), _spec("law", ("contract",)))
    picked, method = rank_specialists_semantic("about the alloy", council, {})
    assert method == "wordlist"
    assert [m.key for m in picked] == ["materials"]


def test_the_word_list_breaks_a_near_tie():
    """0.612 does not really beat 0.607 — but a trigger hit is real evidence."""
    council = (_spec("physics", ()), _spec("electrical", ("resistor",)))
    picked, _ = rank_specialists_semantic(
        "what resistor do I need", council,
        {"physics": 0.612, "electrical": 0.607}, tie_band=0.02)
    assert picked[0].key == "electrical"


def test_the_word_list_cannot_overturn_a_clear_semantic_winner():
    """A keyword hit is a tiebreaker, not a veto."""
    council = (_spec("physics", ()), _spec("electrical", ("resistor",)))
    picked, _ = rank_specialists_semantic(
        "explain the resistor's role in this thermodynamic model", council,
        {"physics": 0.80, "electrical": 0.50}, tie_band=0.02)
    assert picked[0].key == "physics"


def test_the_limit_is_respected():
    council = tuple(_spec(k) for k in ("a", "b", "c", "d"))
    picked, _ = rank_specialists_semantic(
        "anything", council, {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6}, limit=2)
    assert [m.key for m in picked] == ["a", "b"]


def test_ordering_is_stable_for_identical_scores():
    """Same question, same answer — roster order settles a true tie."""
    council = tuple(_spec(k) for k in ("a", "b", "c"))
    scores = {"a": 0.6, "b": 0.6, "c": 0.6}
    first, _ = rank_specialists_semantic("q", council, scores)
    second, _ = rank_specialists_semantic("q", council, scores)
    assert [m.key for m in first] == [m.key for m in second] == ["a", "b", "c"]


def test_trigger_hits_counts_what_the_word_list_would_have_seen():
    member = _spec("electrical", ("resistor", "voltage"))
    assert trigger_hits("what resistor and what voltage", member) == 2
    assert trigger_hits("what colour is the sky", member) == 0


def test_a_user_added_specialist_is_selectable_with_no_keywords():
    """The headline promise: config gives a prompt, MEDO does the rest."""
    council = load_council({"astrologer": {
        "title": "the astrologer", "prompt": "You read star charts."}})
    added = next(m for m in council if m.key == "astrologer")
    assert added.triggers == ()
    assert rank_specialists("what does my star chart say", council) == []
    picked, method = rank_specialists_semantic(
        "what does my star chart say", council, {"astrologer": 0.66})
    assert method == "semantic"
    assert picked[0].key == "astrologer"


# --- the index ---------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic vectors: one dimension per keyword, so scores are readable."""

    def __init__(self, vocabulary: list[str]) -> None:
        self._vocab = vocabulary
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        out = []
        for text in texts:
            low = text.lower()
            vec = np.array([1.0 if word in low else 0.0 for word in self._vocab],
                           dtype=np.float32)
            if not vec.any():
                vec[0] = 0.001            # never a zero vector
            out.append(vec)
        return out


def _index(tmp_path, embedder) -> CouncilIndex:
    return CouncilIndex(tmp_path / "council.db", embedder, "fake-embedder")


def test_the_index_embeds_every_specialist(tmp_path):
    council = (_spec("electrical", ("resistor",)), _spec("law", ("contract",)))
    index = _index(tmp_path, _FakeEmbedder(["resistor", "contract"]))
    assert index.build(council) == 2


def test_the_index_is_cached_between_runs(tmp_path):
    council = (_spec("electrical", ("resistor",)),)
    embedder = _FakeEmbedder(["resistor"])
    _index(tmp_path, embedder).build(council)
    assert embedder.calls == 1
    _index(tmp_path, embedder).build(council)     # a second process, same cache
    assert embedder.calls == 1                    # nothing re-embedded


def test_changing_a_specialist_re_embeds_only_that_one(tmp_path):
    embedder = _FakeEmbedder(["resistor", "contract"])
    index = _index(tmp_path, embedder)
    index.build((_spec("electrical", ("resistor",)), _spec("law", ("contract",))))
    assert embedder.calls == 1
    index.build((_spec("electrical", ("resistor", "capacitor")),
                 _spec("law", ("contract",))))
    assert embedder.calls == 2


def test_a_removed_specialist_is_pruned_from_the_cache(tmp_path):
    embedder = _FakeEmbedder(["resistor", "contract"])
    index = _index(tmp_path, embedder)
    index.build((_spec("electrical", ("resistor",)), _spec("law", ("contract",))))
    assert index.build((_spec("electrical", ("resistor",)),)) == 1


def test_the_index_scores_a_question_against_every_specialist(tmp_path):
    council = (_spec("electrical", ("resistor",)), _spec("law", ("contract",)))
    index = _index(tmp_path, _FakeEmbedder(["resistor", "contract"]))
    index.build(council)
    scores = index.scores("what resistor should I use")
    assert scores["electrical"] > scores["law"]


def test_no_embedder_means_no_scores_and_no_crash(tmp_path):
    index = CouncilIndex(tmp_path / "c.db", None, "none")
    assert index.build((_spec("electrical"),)) == 0
    assert index.scores("anything") == {}


def test_an_embedder_that_raises_degrades_to_no_scores(tmp_path):
    def broken(texts):
        raise RuntimeError("ollama is down")

    council = (_spec("electrical", ("resistor",)),)
    index = _index(tmp_path, _FakeEmbedder(["resistor"]))
    index.build(council)
    index._embedder = broken
    assert index.scores("what resistor") == {}


def test_the_council_index_does_not_collide_with_the_skill_index(tmp_path):
    """A specialist and a skill can share a name; their vectors must not."""
    from core.route_index import SkillRouteIndex

    db = tmp_path / "shared.db"
    embedder = _FakeEmbedder(["weather", "physics"])

    class _Skill:
        name = "weather"

        def routing_text(self):
            return "the weather forecast"

        def routing_surface(self):
            return ["the weather forecast"]

    SkillRouteIndex(db, embedder, "fake").build([_Skill()])
    council_index = CouncilIndex(db, embedder, "fake")
    assert council_index.build((_spec("weather", ("physics",)),)) == 1
    # Both survive: the skill index still holds its own row.
    skills = SkillRouteIndex(db, embedder, "fake")
    assert len(skills.build([_Skill()]).rows) == 1


# --- end to end through the skill --------------------------------------------


@pytest.mark.asyncio
async def test_convene_selects_semantically(tmp_path, monkeypatch):
    from skills.council import ConveneCouncilSkill

    settings = load_settings()
    settings.council.semantic_selection = True
    settings.logging.route_telemetry = False
    monkeypatch.setattr(settings.memory, "db_path", str(tmp_path / "m.db"))

    asked: list[str] = []

    async def ask(system, user):
        asked.append(system)
        return "An answer."

    async def synthesize(question, notes):
        return "Synthesized."

    skill = ConveneCouncilSkill(settings, ask=ask, synthesize=synthesize)
    # A stand-in index that is confident about exactly one specialist.
    class _Index:
        def scores(self, question):
            return {"electrical": 0.9}

        def __len__(self):
            return 1

    skill._index = _Index()
    result = await skill.execute(SkillRequest(
        text="convene the council on this",
        args={"question": "how do I stop this thing catching fire"}))
    assert result.success
    assert len(asked) == 1
    assert "electrical engineer" in asked[0]


@pytest.mark.asyncio
async def test_convene_falls_back_when_the_index_is_unavailable(tmp_path,
                                                                monkeypatch):
    """Ollama down must cost accuracy, never an answer."""
    from skills.council import ConveneCouncilSkill

    settings = load_settings()
    settings.council.semantic_selection = True
    settings.logging.route_telemetry = False
    monkeypatch.setattr(settings.memory, "db_path", str(tmp_path / "m.db"))

    async def ask(system, user):
        return "An answer."

    skill = ConveneCouncilSkill(settings, ask=ask)
    skill._index = False                     # permanently unavailable
    result = await skill.execute(SkillRequest(
        text="convene the council",
        args={"question": "what resistor do I need for this circuit"}))
    assert result.success


@pytest.mark.asyncio
async def test_switching_semantic_selection_off_uses_the_word_list(tmp_path,
                                                                   monkeypatch):
    from skills.council import ConveneCouncilSkill

    settings = load_settings()
    settings.council.semantic_selection = False
    monkeypatch.setattr(settings.memory, "db_path", str(tmp_path / "m.db"))

    async def ask(system, user):
        return "An answer."

    skill = ConveneCouncilSkill(settings, ask=ask)
    assert await skill._ensure_index() is None
    picked = await skill._select("what resistor do I need", 3)
    assert picked and picked[0].key == "electrical"


@pytest.mark.asyncio
async def test_a_semantic_pick_is_logged_against_the_word_list(tmp_path,
                                                               monkeypatch):
    from core.metrics import MetricsStore
    from skills.council import ConveneCouncilSkill

    settings = load_settings()
    settings.council.semantic_selection = True
    settings.logging.route_telemetry = True
    monkeypatch.setattr(settings.memory, "db_path", str(tmp_path / "m.db"))

    skill = ConveneCouncilSkill(settings, ask=None)

    class _Index:
        def scores(self, question):
            return {"law": 0.9}

        def __len__(self):
            return 1

    skill._index = _Index()
    await skill._select("what resistor do I need for this circuit", 3)

    store = MetricsStore(tmp_path / "m.db")
    with store._connect() as conn:
        rows = conn.execute("SELECT method, chosen, alternative, agreed "
                            "FROM council_selections").fetchall()
    assert rows == [("embedding_vs_wordlist", "law", "electrical", 0)]


# --- the index must cover the FULL roster, not the enabled subset ------------
#
# Regression: `council.disabled` is a live HUD toggle, but the index was built
# from the enabled subset. A specialist switched back on afterwards had no
# vector, scored -1, and was silently skipped — so a legal question went to
# whoever else cleared the threshold, and that expert answered it confidently.

@pytest.mark.asyncio
async def test_a_specialist_re_enabled_at_runtime_is_still_reachable(tmp_path,
                                                                     monkeypatch):
    from core.council_index import CouncilIndex
    from skills.council import ConveneCouncilSkill

    settings = load_settings()
    settings.council.semantic_selection = True
    settings.logging.route_telemetry = False
    settings.council.disabled = ["law"]          # off when the index is built
    monkeypatch.setattr(settings.memory, "db_path", str(tmp_path / "m.db"))

    skill = ConveneCouncilSkill(settings, ask=None)
    index = CouncilIndex(tmp_path / "c.db",
                         _FakeEmbedder(["law", "contract", "legal", "circuit"]),
                         "fake")
    # What _ensure_index now passes: the full roster.
    assert index.build(skill._roster) == len(skill._roster)
    assert "law" in index.scores("a legal contract question")
    skill._index = index

    settings.council.disabled = []                # re-enabled from the HUD
    picked = await skill._select("is this contract legal", 3)
    assert picked and picked[0].key == "law"


def test_the_index_covers_a_disabled_specialist(tmp_path):
    from core.council_index import CouncilIndex
    from skills.council import ConveneCouncilSkill

    settings = load_settings()
    settings.council.disabled = ["law", "medicine"]
    skill = ConveneCouncilSkill(settings, ask=None)
    index = CouncilIndex(tmp_path / "c.db", _FakeEmbedder(["law"]), "fake")
    index.build(skill._roster)
    scored = index.scores("anything")
    assert "law" in scored and "medicine" in scored


@pytest.mark.asyncio
async def test_a_disabled_specialist_is_still_never_selected(tmp_path,
                                                             monkeypatch):
    """Indexing everyone must not make a switched-off expert answerable."""
    from skills.council import ConveneCouncilSkill

    settings = load_settings()
    settings.council.semantic_selection = True
    settings.logging.route_telemetry = False
    settings.council.disabled = ["law"]
    monkeypatch.setattr(settings.memory, "db_path", str(tmp_path / "m.db"))

    skill = ConveneCouncilSkill(settings, ask=None)

    class _Index:
        def scores(self, question):
            return {"law": 0.99}                 # the index still knows them

        def __len__(self):
            return 1

    skill._index = _Index()
    picked = await skill._select("is this contract legal", 3)
    assert "law" not in [m.key for m in picked]
