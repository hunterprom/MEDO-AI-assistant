"""Round 6 — the failures a live English session exposed (2026-08-08).

The transcript, and what each line proved:

* "make me an app" -> "What should the app do?" -> "I want the app to track the
  weather in Macedonia" -> *"It's 22 degrees and clear in Jonesboro."* Two bugs
  in one line: the reply capture was abandoned because the answer happened to
  match the weather skill, and the geocoder resolved a country to a village in
  Louisiana.
* "Give me a morning brief." -> **115 seconds**, of which 72 were the 30B
  thinking model reformatting sentences that were already final.
* "...Worth remembering: I don't drink coffee." — the one fact in the store,
  announced as the day's schedule.
* "What feature would you like implemented?" -> a fenced JSON function schema,
  read out loud, braces and all.
* "I can locate_in_app weather app, but..." — an internal tool name spoken.
* "What features would you like added to your system?" -> *"Lion mode is off."*
  The semantic tier answered a question about MEDO with a security skill.
"""

from __future__ import annotations

import asyncio
import re

import numpy as np
import pytest

from core.config import load_settings
from core.events import EventBus
from core.facts import FactsStore
from core.router import Router, _is_about_medo
from core.speech_text import (
    FenceFilter,
    leaked_tool_names,
    looks_like_markup,
    strip_markup,
)
from llm.client import OllamaClient
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult
from skills.weather import WeatherSkill, _fold


@pytest.fixture
def router_factory():
    """A router with no model, so an unmatched turn can't reach the network."""
    def build():
        settings = load_settings()
        settings.router.semantic_enabled = False
        registry = SkillRegistry()
        registry.register(WeatherSkill(settings.weather))
        router = Router(settings, registry, OllamaClient(settings.llm), EventBus())
        router.model = None
        return router, registry
    return build


# --- the reply capture that got stolen -------------------------------------

class _Asker(Skill):
    """Asks an OPEN question, like "what should the app do?"."""

    name = "asker"
    description = "asks"
    patterns = [re.compile(r"\bask me\b", re.IGNORECASE)]

    def __init__(self, open_question: bool = True) -> None:
        self.captured: list[str] = []
        self._open = open_question

    async def execute(self, request: SkillRequest) -> SkillResult:
        if request.context.get("captured_reply"):
            self.captured.append(request.text)
            return SkillResult("got it")
        return SkillResult("What should the app do?", await_reply=True,
                           reply_is_open=self._open)


def test_open_capture_keeps_an_answer_that_matches_another_skill(router_factory):
    """The bug: a spec mentioning weather was eaten by the weather skill."""
    router, registry = router_factory()
    asker = registry.register(_Asker())

    asyncio.run(router.route("ask me"))
    asyncio.run(router.route(
        "I want the app to track the weather in Macedonia, every village"))

    assert asker.captured == [
        "I want the app to track the weather in Macedonia, every village"]


def test_narrow_capture_still_yields_to_a_real_command(router_factory):
    """The existing behaviour must survive: a NON-open question still lets a
    genuine command through rather than swallowing it as an answer."""
    router, registry = router_factory()
    asker = registry.register(_Asker(open_question=False))

    asyncio.run(router.route("ask me"))
    result = asyncio.run(router.route("what's the weather"))

    assert asker.captured == []
    assert result.skill_name == "weather"


def test_open_capture_is_armed_only_for_one_turn(router_factory):
    router, registry = router_factory()
    asker = registry.register(_Asker())

    asyncio.run(router.route("ask me"))
    asyncio.run(router.route("something that tracks water"))
    result = asyncio.run(router.route("what's the weather"))

    assert len(asker.captured) == 1
    assert result.skill_name == "weather"


def test_make_app_asks_an_open_question():
    from core.app_builder import AppBuilder
    from core.config import AppBuilderConfig
    from skills.app_builder_skill import MakeAppSkill

    skill = MakeAppSkill(AppBuilder(AppBuilderConfig(enabled=True)))
    out = asyncio.run(skill.execute(SkillRequest(text="make me an app")))
    assert out.await_reply and out.reply_is_open


# --- the geocoder that answered with Louisiana ------------------------------

def _rank(results, query):
    return max(results, key=lambda x: WeatherSkill._geo_rank(x, query))["name"]


def test_exact_name_beats_a_fuzzy_alternate_name_hit():
    """Open-Meteo led "Macedonia" with Jonesboro, LA — an alternate-name match."""
    results = [
        {"name": "Jonesboro", "population": 4587, "feature_code": "PPLA2"},
        {"name": "Macedonia", "population": 11686, "feature_code": "PPL"},
        {"name": "Macedonia", "population": 245, "feature_code": "PPL"},
    ]
    assert _rank(results, "Macedonia") == "Macedonia"


def test_a_capital_beats_a_bigger_namesake():
    results = [
        {"name": "Skopje", "population": 474889, "feature_code": "PPLC"},
        {"name": "Skopje", "population": None, "feature_code": "AIRP"},
    ]
    assert _rank(results, "Skopje") == "Skopje"


def test_population_breaks_a_tie_between_exact_matches():
    results = [
        {"name": "Paris", "population": 24782, "feature_code": "PPLA2"},
        {"name": "Paris", "population": 2138551, "feature_code": "PPLC"},
    ]
    best = max(results, key=lambda x: WeatherSkill._geo_rank(x, "Paris"))
    assert best["population"] == 2138551


def test_missing_population_does_not_crash_the_ranking():
    results = [{"name": "Nowhere", "population": None, "feature_code": ""},
               {"name": "Nowhere", "population": "n/a", "feature_code": ""}]
    assert _rank(results, "Nowhere") == "Nowhere"


def test_accents_fold_when_comparing_names():
    assert _fold("Macedônia") == _fold("Macedonia")
    assert _fold("  ZÜRICH ") == _fold("zurich")


@pytest.mark.parametrize("spoken", ["Macedonia", "macedonia", "Македонија"])
def test_the_country_is_not_a_village_in_ohio(spoken):
    """Locals say "Macedonia"; the gazetteer files the country as North Macedonia."""
    from skills.weather import _PLACE_ALIASES

    assert _PLACE_ALIASES[_fold(spoken)] == "North Macedonia"


# --- the briefing that took two minutes -------------------------------------

def test_briefing_fetches_its_network_sections_concurrently():
    """Weather waited for the news feed for no reason."""
    from skills.briefing import BriefingSkill

    order: list[str] = []

    class _Slow(Skill):
        name = "slow"
        description = "slow"

        def __init__(self, label: str, delay: float) -> None:
            self._label, self._delay = label, delay

        async def execute(self, request: SkillRequest) -> SkillResult:
            order.append(f"{self._label}-start")
            await asyncio.sleep(self._delay)
            order.append(f"{self._label}-end")
            return SkillResult(f"{self._label} says hello.")

    skill = BriefingSkill(
        sections=["weather", "news"],
        weather=_Slow("weather", 0.05), news=_Slow("news", 0.01))
    out = asyncio.run(skill.execute(SkillRequest(text="morning briefing")))

    # Both start before either finishes — that is what concurrency means here.
    assert order[:2] == ["weather-start", "news-start"]
    # …and the configured section ORDER is still what gets spoken.
    assert out.speech.index("weather says") < out.speech.index("news says")


def test_one_failing_section_never_kills_the_briefing():
    from skills.briefing import BriefingSkill

    class _Broken(Skill):
        name = "broken"
        description = "broken"

        async def execute(self, request: SkillRequest) -> SkillResult:
            raise RuntimeError("feed is down")

    class _Fine(Skill):
        name = "fine"
        description = "fine"

        async def execute(self, request: SkillRequest) -> SkillResult:
            return SkillResult("It's 26 degrees.")

    skill = BriefingSkill(sections=["weather", "news"],
                          weather=_Fine(), news=_Broken())
    out = asyncio.run(skill.execute(SkillRequest(text="morning briefing")))
    assert "26 degrees" in out.speech


# --- the coffee fact announced as a schedule --------------------------------

def test_search_returns_nothing_when_no_fact_is_relevant(tmp_path):
    """`relevant()` handed back its whole store when the store was small — which
    is how "I don't drink coffee" became the morning's upcoming plans."""
    store = FactsStore(str(tmp_path / "f.db"),
                       embedder=lambda texts: [_vec(t) for t in texts])
    store.add("I don't drink coffee")

    assert store.relevant("appointment or plan for today", 3) == [
        "I don't drink coffee"]                       # unchanged: pads for the prompt
    assert store.search("appointment or plan for today", 3) == []


def test_search_finds_a_fact_that_does_match(tmp_path):
    store = FactsStore(str(tmp_path / "f.db"),
                       embedder=lambda texts: [_vec(t) for t in texts])
    store.add("dentist appointment tomorrow at three")
    assert store.search("dentist appointment tomorrow at three", 3) == [
        "dentist appointment tomorrow at three"]


def test_search_is_quiet_without_an_embedder(tmp_path):
    store = FactsStore(str(tmp_path / "f.db"))
    store.add("I don't drink coffee")
    assert store.search("anything at all") == []


def _vec(text: str):
    """A toy embedding: one dimension per keyword, so similarity is decidable.

    float32 numpy, because the store persists vectors with ``.tobytes()``.
    """
    words = ("appointment", "plan", "schedule", "today", "tomorrow",
             "dentist", "coffee", "drink", "three")
    lowered = text.lower()
    return np.array([1.0 if w in lowered else 0.0 for w in words],
                    dtype=np.float32)


def test_briefing_skips_upcoming_when_nothing_matches():
    from skills.briefing import BriefingSkill

    class _Empty:
        def relevant(self, query, limit):
            return ["I don't drink coffee"]

        def search(self, query, limit=5, min_score=0.55):
            return []

    skill = BriefingSkill(sections=["upcoming"], weather=None, news=None,
                          facts=_Empty())
    out = asyncio.run(skill.execute(SkillRequest(text="morning briefing")))
    assert "coffee" not in out.speech
    assert "Worth remembering" not in out.speech


# --- the JSON schema that got read out loud ---------------------------------

_SCHEMA_REPLY = (
    'I\'d like to propose a new feature: "Install App". Here is how the '
    'function could be defined: ```json\n'
    '{"name": "install_app", "parameters": {"app": {"type": "string"}}}\n'
    '```\nIs this the kind of feature you were thinking of?'
)


def test_a_fenced_schema_is_recognised_as_written_not_spoken():
    assert looks_like_markup(_SCHEMA_REPLY)


def test_stripping_leaves_only_the_speakable_prose():
    out = strip_markup(_SCHEMA_REPLY)
    assert "Install App" in out and "kind of feature" in out
    assert "{" not in out and '"type"' not in out and "```" not in out


@pytest.mark.parametrize("written", [
    '{"name": "open_website", "arguments": {"url": "x"}}',
    "Here you go:\n```python\nprint(1)\n```",
    'The config is {"key": 1}.',
])
def test_written_answers_are_flagged(written):
    assert looks_like_markup(written)


@pytest.mark.parametrize("spoken", [
    "It's 34 degrees and partly cloudy in Skopje.",
    "I'll build it in the background and let you know when it's ready.",
    "Во Скопје е 34 степени и ведро.",
    "Nothing much — I'm ready when you are.",
])
def test_ordinary_speech_is_left_alone(spoken):
    assert not looks_like_markup(spoken)
    assert strip_markup(spoken) == spoken


def test_markdown_furniture_is_removed_but_the_words_survive():
    md = ("Would you like: * A **guided** meditation? "
          "* A [calming story](http://x) to listen to?")
    out = strip_markup(md)
    assert out == ("Would you like: A guided meditation? "
                   "A calming story to listen to?")


def test_arithmetic_is_not_mistaken_for_a_bullet():
    assert strip_markup("the answer is 2 * 3 which is 6") == (
        "the answer is 2 * 3 which is 6")


def test_a_reply_that_is_only_a_code_block_strips_to_nothing():
    assert strip_markup("```\nprint('hi')\n```") == ""


def test_an_unterminated_fence_is_still_dropped():
    """A cut-off stream leaves the fence open; speaking the rest is wrong."""
    assert strip_markup("Here it is: ```json\n{\"a\": 1,") == "Here it is:"


# --- the tool name spoken out loud ------------------------------------------

def test_a_tool_name_in_prose_is_caught():
    reply = ("I can locate_in_app weather app, but I don't know if it tracks "
             "every village.")
    assert leaked_tool_names(reply, ["locate_in_app", "weather"]) == [
        "locate_in_app"]


def test_snake_case_that_is_not_a_tool_is_ignored():
    reply = "Save it as my_notes_file and you're done."
    assert leaked_tool_names(reply, ["locate_in_app", "open_website"]) == []


def test_leak_detection_survives_a_junk_registry():
    assert leaked_tool_names("anything", None) == []
    assert leaked_tool_names("", ["open_website"]) == []


# --- streaming: the fence arrives in pieces ---------------------------------

def test_a_fence_split_across_chunks_is_never_spoken():
    f = FenceFilter()
    chunks = ["Here is the schema: ", "``", "`js", "on\n{\"a\": 1,", " \"b\": 2}\n",
              "``", "`", " Is that what you wanted?"]
    out = "".join(f.feed(c) for c in chunks) + f.flush()
    assert "{" not in out and "```" not in out
    assert "Here is the schema:" in out and "Is that what you wanted?" in out


def test_backticks_that_never_become_a_fence_are_not_swallowed():
    f = FenceFilter()
    out = "".join(f.feed(c) for c in ["it costs ``", " about ten"]) + f.flush()
    assert "about ten" in out


def test_a_stream_cut_off_inside_a_fence_emits_nothing_from_it():
    f = FenceFilter()
    out = "".join(f.feed(c) for c in ["talk ", "```json\n", "{\"a\": 1"]) + f.flush()
    assert out.strip() == "talk"


def test_plain_streaming_text_passes_through_untouched():
    f = FenceFilter()
    chunks = ["It's 34 ", "degrees and ", "partly cloudy."]
    assert "".join(f.feed(c) for c in chunks) + f.flush() == (
        "It's 34 degrees and partly cloudy.")


# --- the question about MEDO answered with a security skill -----------------

@pytest.mark.parametrize("question", [
    "What features would you like to be added in your system other than "
    "installing applications?",
    "What feature would you like to be implemented in your system?",
    "what can you do",
    "what could you do for me",
    "who are you",
    "tell me about yourself",
    "would you like a new capability",
    "what do you think about your architecture",
    "што можеш да правиш",
])
def test_questions_about_medo_skip_the_semantic_tier(question):
    assert _is_about_medo(question)


@pytest.mark.parametrize("command", [
    "do any of my apps need updating",
    "which programs have updates waiting",
    "what's the weather in Skopje",
    "open steam",
    "check for updates",
    "give me a morning brief",
    "make me an app that tracks my water intake",
    "отвори стим",
    "направи апликација за времето",
])
def test_real_commands_still_reach_the_semantic_tier(command):
    assert not _is_about_medo(command)
