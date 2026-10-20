"""P2-2: MEDO remembers that yesterday happened.

The conversation window is ten turns in a deque that dies with the process, so
every session began as the first one MEDO had ever had. Episodic memory keeps a
few model-written sentences per session — not the transcript — embedded for
recall by meaning.

Two things are load-bearing and get most of the attention here: the memory must
FAIL SILENTLY (it is written at shutdown, where an exception replaces a clean
exit with a traceback), and recall must work in Macedonian, because a fabricated
memory is worse than none and that is exactly what an English-only regex causes
when it drops the question through to the LLM.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from core.config import load_settings
from core.episodes import (
    Episode,
    EpisodeStore,
    summarize_session,
    transcript,
)
from skills.base import SkillRequest
from skills.episode_recall import EpisodeRecallSkill, parse_window


class _Embedder:
    """A vector per keyword, so similarity is readable in the assertions."""

    def __init__(self, vocab: list[str]) -> None:
        self._vocab = vocab

    def __call__(self, texts):
        out = []
        for text in texts:
            low = text.lower()
            vec = np.array([1.0 if w in low else 0.0 for w in self._vocab],
                           dtype=np.float32)
            if not vec.any():
                vec[0] = 0.001
            out.append(vec)
        return out


def _store(tmp_path, vocab=("roof", "guttering", "python", "bug")) -> EpisodeStore:
    return EpisodeStore(tmp_path / "medo.db", _Embedder(list(vocab)), "fake")


# --- storing ------------------------------------------------------------------


def test_an_episode_is_stored_and_read_back(tmp_path):
    store = _store(tmp_path)
    assert store.add("We talked about the roof and the guttering.", turns=6)
    episodes = store.recent()
    assert len(episodes) == 1
    assert "guttering" in episodes[0].summary
    assert episodes[0].turns == 6


def test_an_empty_summary_is_not_stored(tmp_path):
    store = _store(tmp_path)
    assert store.add("   ") is None
    assert store.count() == 0


def test_storing_without_an_embedder_still_works(tmp_path):
    """No Ollama is a degraded memory, not a broken one."""
    store = EpisodeStore(tmp_path / "medo.db", None, "none")
    assert store.add("We talked about the roof.")
    assert store.recent()[0].summary.endswith("roof.")


def test_an_unwritable_database_does_not_raise(tmp_path):
    store = EpisodeStore(tmp_path / "nope" / "deeper" / "medo.db", None, "none")
    assert store.add("anything") is None      # logged, not raised


# --- recall by meaning --------------------------------------------------------


def test_recall_finds_the_relevant_episode(tmp_path):
    store = _store(tmp_path)
    store.add("We debugged a python bug in the router.")
    store.add("We talked about the roof and the guttering.")
    found = store.search("roof", limit=1)
    assert "guttering" in found[0].summary


def test_recall_falls_back_to_substring_without_an_embedder(tmp_path):
    store = EpisodeStore(tmp_path / "medo.db", None, "none")
    store.add("We debugged a python bug in the router.")
    store.add("We talked about the roof and the guttering.")
    found = store.search("roof", limit=1)
    assert "guttering" in found[0].summary


def test_recall_falls_back_to_recency_when_nothing_matches(tmp_path):
    """Answering "nothing" because the model is cold reads as amnesia."""
    store = EpisodeStore(tmp_path / "medo.db", None, "none")
    store.add("We talked about the roof.")
    assert store.search("submarines", limit=1)


def test_vectors_from_another_embed_model_are_ignored(tmp_path):
    """Cosine across two embedding spaces is confident nonsense."""
    path = tmp_path / "medo.db"
    EpisodeStore(path, _Embedder(["roof"]), "model-a").add("The roof again.")
    other = EpisodeStore(path, _Embedder(["roof"]), "model-b")
    assert other._by_similarity(other._rows(), np.array([1.0], dtype=np.float32)) == []
    assert other.search("roof")          # still answers, via the fallbacks


def test_recall_can_be_limited_to_a_time_window(tmp_path):
    store = EpisodeStore(tmp_path / "medo.db", None, "none")
    now = datetime.now().astimezone()
    store.add("Old conversation about the roof.",
              ended_at=(now - timedelta(days=10)).isoformat(timespec="seconds"))
    store.add("Yesterday's conversation about the roof.",
              ended_at=(now - timedelta(days=1)).isoformat(timespec="seconds"))
    window = (now - timedelta(days=2), now)
    found = store.search("roof", limit=5, window=window)
    assert len(found) == 1
    assert "Yesterday" in found[0].summary


# --- retention ----------------------------------------------------------------


def test_old_episodes_are_pruned(tmp_path):
    store = EpisodeStore(tmp_path / "medo.db", None, "none")
    old = (datetime.now().astimezone() - timedelta(days=200)).isoformat()
    store.add("Ancient history.", ended_at=old)
    store.add("Recent history.")
    assert store.prune(keep_days=90) == 1
    assert [e.summary for e in store.recent()] == ["Recent history."]


def test_a_pinned_episode_survives_the_sweep(tmp_path):
    store = EpisodeStore(tmp_path / "medo.db", None, "none")
    old = (datetime.now().astimezone() - timedelta(days=200)).isoformat()
    episode_id = store.add("Ancient but important.", ended_at=old)
    assert store.pin(episode_id)
    assert store.prune(keep_days=90) == 0
    assert store.count() == 1


def test_pinning_can_be_released(tmp_path):
    store = EpisodeStore(tmp_path / "medo.db", None, "none")
    old = (datetime.now().astimezone() - timedelta(days=200)).isoformat()
    episode_id = store.add("Ancient.", ended_at=old)
    store.pin(episode_id)
    store.pin(episode_id, False)
    assert store.prune(keep_days=90) == 1


def test_zero_retention_days_keeps_everything(tmp_path):
    store = EpisodeStore(tmp_path / "medo.db", None, "none")
    old = (datetime.now().astimezone() - timedelta(days=2000)).isoformat()
    store.add("Ancient.", ended_at=old)
    assert store.prune(keep_days=0) == 0
    assert store.count() == 1


def test_an_episode_can_be_deleted_outright(tmp_path):
    store = EpisodeStore(tmp_path / "medo.db", None, "none")
    episode_id = store.add("Forget this.")
    assert store.delete(episode_id) is True
    assert store.count() == 0


# --- summarising --------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_session_is_summarised_by_the_local_model():
    seen = {}

    async def ask(system, user):
        seen["system"], seen["user"] = system, user
        return "  We planned the roof repair.  "

    turns = [("what about the roof", "it needs guttering"),
             ("when", "before winter")]
    assert await summarize_session(turns, ask) == "We planned the roof repair."
    assert "roof" in seen["user"]
    assert "only state what the transcript actually shows" in seen["system"]


@pytest.mark.asyncio
async def test_a_one_turn_session_is_not_summarised():
    """"What time is it" is a command; summarising it costs a model call."""
    called = []

    async def ask(system, user):
        called.append(user)
        return "x"

    assert await summarize_session([("what time is it", "noon")], ask) == ""
    assert called == []


@pytest.mark.asyncio
async def test_no_model_means_no_summary_and_no_crash():
    assert await summarize_session([("a", "b"), ("c", "d")], None) == ""


@pytest.mark.asyncio
async def test_a_failing_model_does_not_raise_at_shutdown():
    async def ask(system, user):
        raise RuntimeError("ollama is gone")

    assert await summarize_session([("a", "b"), ("c", "d")], ask) == ""


def test_the_transcript_keeps_the_most_recent_turns():
    turns = [(f"q{i}", f"a{i}") for i in range(100)]
    text = transcript(turns, limit=3)
    assert "q99" in text and "q0" not in text


# --- how long ago it was ------------------------------------------------------


@pytest.mark.parametrize("days,expected", [
    (0, "earlier today"), (1, "yesterday"), (10, "last week"), (400, "months ago"),
])
def test_when_reads_as_speech_not_a_timestamp(days, expected):
    now = datetime.now().astimezone()
    episode = Episode(1, "", (now - timedelta(days=days)).isoformat(), "s")
    assert expected in episode.when(now)


def test_an_unparseable_timestamp_does_not_crash():
    assert Episode(1, "", "not-a-date", "s").when() == "at some point"


# --- the time-window parser, in both languages --------------------------------


def test_yesterday_is_a_calendar_day_not_the_last_24_hours():
    now = datetime(2026, 8, 25, 14, 30).astimezone()
    start, end = parse_window("what did we talk about yesterday", now)
    assert (start.day, start.hour) == (24, 0)
    assert (end.day, end.hour) == (25, 0)


@pytest.mark.parametrize("phrase", ["вчера", "синоќа"])
def test_yesterday_in_macedonian(phrase):
    now = datetime(2026, 8, 25, 14, 30).astimezone()
    start, _end = parse_window(f"за што зборувавме {phrase}", now)
    assert start.day == 24


def test_last_week_in_both_languages():
    now = datetime(2026, 8, 25, 14, 30).astimezone()     # a Tuesday
    en = parse_window("what did we discuss last week", now)
    mk = parse_window("за што зборувавме минатата недела", now)
    assert en == mk
    assert (now - en[0]).days >= 7


def test_a_number_of_days_ago_in_both_languages():
    now = datetime(2026, 8, 25, 12, 0).astimezone()
    assert parse_window("what did we say 3 days ago", now)[0].day == 22
    assert parse_window("за што зборувавме пред 3 дена", now)[0].day == 22


def test_no_time_word_means_any_time():
    assert parse_window("what did we talk about") is None


# --- the skill ----------------------------------------------------------------


def _skill(tmp_path, store=None):
    settings = load_settings()
    return EpisodeRecallSkill(settings, store or _store(tmp_path))


@pytest.mark.parametrize("utterance", [
    "what did we talk about yesterday",
    "what were we discussing last week",
    "what have we been working on",
    "remind me what we decided",
    "our last conversation",
])
def test_the_fast_path_claims_english_recall_questions(tmp_path, utterance):
    assert _skill(tmp_path).match(utterance) is not None


@pytest.mark.parametrize("utterance", [
    "за што зборувавме вчера",
    "за што разговаравме минатата недела",
    "што решивме вчера",
    "на што работевме",
])
def test_the_fast_path_claims_macedonian_recall_questions(tmp_path, utterance):
    """English-only patterns drop these to the LLM, which invents a memory."""
    assert _skill(tmp_path).match(utterance) is not None


@pytest.mark.parametrize("utterance", [
    "what time is it",
    "what do you remember about my dentist",
    "remind me to call mum",
])
def test_the_fast_path_leaves_other_questions_alone(tmp_path, utterance):
    assert _skill(tmp_path).match(utterance) is None


@pytest.mark.asyncio
async def test_recall_reads_back_what_it_finds(tmp_path):
    store = _store(tmp_path)
    now = datetime.now().astimezone()
    store.add("We planned the roof repair.",
              ended_at=(now - timedelta(days=1)).isoformat(timespec="seconds"))
    skill = _skill(tmp_path, store)
    result = await skill.execute(SkillRequest(
        text="what did we talk about yesterday",
        match=skill.match("what did we talk about yesterday")))
    assert "roof repair" in result.speech
    assert "esterday" in result.speech
    assert result.data["episodes"] == 1


@pytest.mark.asyncio
async def test_recall_answers_in_macedonian_when_asked_in_macedonian(tmp_path):
    store = _store(tmp_path)
    now = datetime.now().astimezone()
    store.add("Разговаравме за покривот.",
              ended_at=(now - timedelta(days=1)).isoformat(timespec="seconds"))
    skill = _skill(tmp_path, store)
    result = await skill.execute(SkillRequest(text="за што зборувавме вчера"))
    assert "покривот" in result.speech


@pytest.mark.asyncio
async def test_asking_about_yesterday_does_not_answer_with_today(tmp_path):
    """A memory offered for the wrong day is a fabricated one."""
    store = _store(tmp_path)
    store.add("Разговаравме за покривот.")          # stored NOW, i.e. today
    skill = _skill(tmp_path, store)
    result = await skill.execute(SkillRequest(text="за што зборувавме вчера"))
    assert result.data["episodes"] == 0


@pytest.mark.asyncio
async def test_recall_says_so_when_there_is_nothing(tmp_path):
    skill = _skill(tmp_path)
    result = await skill.execute(SkillRequest(text="what did we talk about yesterday"))
    assert "don't have anything" in result.speech
    assert result.data["episodes"] == 0


@pytest.mark.asyncio
async def test_recall_says_so_in_macedonian_when_there_is_nothing(tmp_path):
    skill = _skill(tmp_path)
    result = await skill.execute(SkillRequest(text="за што зборувавме вчера"))
    assert "Немам" in result.speech


@pytest.mark.asyncio
async def test_recall_declines_when_the_feature_is_off(tmp_path):
    settings = load_settings()
    settings.memory.episodes_enabled = False
    skill = EpisodeRecallSkill(settings, _store(tmp_path))
    result = await skill.execute(SkillRequest(text="what did we talk about"))
    assert result.success is False
    assert "not keeping memories" in result.speech


@pytest.mark.asyncio
async def test_recall_uses_the_tool_arguments_on_the_llm_path(tmp_path):
    store = _store(tmp_path)
    now = datetime.now().astimezone()
    store.add("We debugged a python bug.",
              ended_at=(now - timedelta(days=1)).isoformat(timespec="seconds"))
    store.add("We talked about the roof.",
              ended_at=(now - timedelta(days=30)).isoformat(timespec="seconds"))
    skill = _skill(tmp_path, store)
    result = await skill.execute(SkillRequest(
        text="remind me", args={"about": "python bug", "when": "yesterday"}))
    assert "python bug" in result.speech


def test_the_skill_is_registered_in_the_live_registry():
    from main import Announcer, build_registry

    registry = build_registry(load_settings(), Announcer())
    assert registry.get("recall_conversation") is not None


# --- one language per reply ---------------------------------------------------
#
# Regression: recall spliced an English time phrase onto a Macedonian summary —
# "On saturday: Разговаравме за покривот." The old code translated a lookup
# table of three phrases ("yesterday", "earlier today", "last week") and let
# every other one through in English. Also `str.capitalize()` lowercases the
# rest of the string, so English got "on saturday" too.

@pytest.mark.parametrize("days", [0, 1, 2, 3, 5, 10, 25, 200])
def test_when_is_macedonian_end_to_end(days):
    now = datetime.now().astimezone()
    episode = Episode(1, "", (now - timedelta(days=days)).isoformat(), "s")
    phrase = episode.when(now, language="mk")
    assert phrase.strip()
    # No Latin letters at all: a single stray English word is the whole bug.
    assert not any("a" <= c.lower() <= "z" for c in phrase), phrase


@pytest.mark.parametrize("days", [0, 1, 2, 3, 5, 10, 25, 200])
def test_when_is_english_end_to_end(days):
    now = datetime.now().astimezone()
    episode = Episode(1, "", (now - timedelta(days=days)).isoformat(), "s")
    phrase = episode.when(now, language="en")
    assert phrase.strip()
    assert not any("Ѐ" <= c <= "ӿ" for c in phrase), phrase


def test_the_weekday_is_not_locale_dependent():
    """strftime('%A') returns whatever the OS locale is; both are spelled out."""
    now = datetime(2026, 8, 25, 12, 0).astimezone()          # a Tuesday
    saturday = Episode(1, "", (now - timedelta(days=3)).isoformat(), "s")
    assert saturday.when(now, "en") == "on Saturday"
    assert saturday.when(now, "mk") == "во сабота"


@pytest.mark.asyncio
async def test_a_macedonian_recall_has_no_english_in_it(tmp_path):
    store = EpisodeStore(tmp_path / "m.db", None, "none")
    now = datetime.now().astimezone()
    for days in (3, 25):
        store.add("Разговаравме за покривот.",
                  ended_at=(now - timedelta(days=days)).isoformat(timespec="seconds"))
    skill = EpisodeRecallSkill(load_settings(), store)
    result = await skill.execute(SkillRequest(text="за што зборувавме"))
    assert not any("a" <= c.lower() <= "z" for c in result.speech), result.speech


@pytest.mark.asyncio
async def test_an_english_recall_capitalises_the_weekday(tmp_path):
    store = EpisodeStore(tmp_path / "m.db", None, "none")
    now = datetime.now().astimezone()
    store.add("We talked about the roof.",
              ended_at=(now - timedelta(days=3)).isoformat(timespec="seconds"))
    skill = EpisodeRecallSkill(load_settings(), store)
    result = await skill.execute(SkillRequest(text="what did we talk about"))
    assert " on " not in result.speech.lower()[:3]
    assert result.speech[0].isupper()
    # capitalize() would have lowercased the weekday it follows.
    assert "saturday" not in result.speech and "sunday" not in result.speech


# --- a named topic must be MATCHED, not approximated -------------------------
#
# Regression: "what did we say about quantum physics" returned the most recent
# episode — the roof — which a listener hears as MEDO confirming a conversation
# that never happened. Recency is the right answer to "what did we talk about";
# for a named subject it is a fabricated memory.

@pytest.mark.parametrize("text,expected", [
    ("what did we talk about", ""),
    ("what did we talk about yesterday", ""),
    ("what were we discussing last week", ""),
    ("what did we say about quantum physics", "quantum physics"),
    ("what did we decide about the roof", "the roof"),
    ("what did we say regarding the invoice", "the invoice"),
    # Macedonian anchors on the VERB: "за" opens the stock framing itself, so
    # keying on the preposition read every broad question as a named topic.
    ("за што зборувавме", ""),
    ("за што зборувавме вчера", ""),
    ("за што зборувавме за покривот", "покривот"),
    ("на што работевме за проектот", "проектот"),
])
def test_a_named_topic_is_told_apart_from_the_stock_framing(text, expected):
    assert EpisodeRecallSkill._named_topic(text) == expected


def test_the_last_preposition_wins():
    """The leftmost match would swallow the rest of the sentence as the topic."""
    assert EpisodeRecallSkill._named_topic(
        "what did we say about it, and about the roof") == "the roof"


def test_an_unmatched_named_topic_returns_nothing(tmp_path):
    store = EpisodeStore(tmp_path / "m.db", _Embedder(["roof", "quantum"]), "fake")
    store.add("We talked about the roof.")
    assert store.search("quantum", require_relevance=True) == []


def test_a_matched_named_topic_still_returns_it(tmp_path):
    store = EpisodeStore(tmp_path / "m.db", _Embedder(["roof", "quantum"]), "fake")
    store.add("We talked about the roof.")
    assert store.search("roof", require_relevance=True)


def test_a_broad_question_still_falls_back_to_recency(tmp_path):
    """Without a named topic, the most recent conversation IS the answer."""
    store = EpisodeStore(tmp_path / "m.db", _Embedder(["roof", "quantum"]), "fake")
    store.add("We talked about the roof.")
    assert store.search("submarines", require_relevance=False)


@pytest.mark.asyncio
async def test_recall_does_not_invent_a_conversation(tmp_path):
    store = EpisodeStore(tmp_path / "m.db", _Embedder(["roof", "quantum"]), "fake")
    now = datetime.now().astimezone()
    store.add("We talked about the roof and the guttering.",
              ended_at=(now - timedelta(days=2)).isoformat(timespec="seconds"))
    skill = EpisodeRecallSkill(load_settings(), store)
    result = await skill.execute(
        SkillRequest(text="what did we say about quantum physics"))
    assert "roof" not in result.speech
    assert result.data["episodes"] == 0


@pytest.mark.asyncio
async def test_recall_does_not_invent_one_on_the_llm_path_either(tmp_path):
    store = EpisodeStore(tmp_path / "m.db", _Embedder(["roof", "quantum"]), "fake")
    store.add("We talked about the roof.")
    skill = EpisodeRecallSkill(load_settings(), store)
    result = await skill.execute(SkillRequest(
        text="remind me", args={"about": "quantum physics"}))
    assert result.data["episodes"] == 0
