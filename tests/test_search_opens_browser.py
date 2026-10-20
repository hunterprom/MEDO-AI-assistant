"""A search COMMAND opens the browser; a QUESTION still gets answered.

The transcript this came from:

    YOU   Search up for Mazda RX-7
    MEDO  FAST · 19879 MS
          "The Mazda RX-7 is a Japanese sports car built around Mazda's
           rotary Wankel engine, and the third-generation FD from 1992 to
           2002 is the one people tend to obsess over, ..."

Three things wrong with that, in order of how much they cost:

1. It answered a command. "Search up X" is an instruction to go and look —
   the right answer is the results page, clickable, in the browser MEDO is
   already driving. A paragraph is not that.
2. It took 19.9 s to produce the paragraph (five DDG snippets, then the 30B
   model narrating them).
3. The query it actually searched for was "up for Mazda RX-7" — the pattern
   captures everything after the verb, and "up" belongs to the phrasal verb.

A question is a different utterance and keeps the old behaviour: "what's the
latest on X", anything reaching this skill by MEANING, and the model's own
web_search tool calls all still come back as text. ``browser_scope: always``
moves those into the browser too.
"""

from __future__ import annotations

import pytest

from core.config import WebSearchConfig, load_settings
from skills.base import SkillRequest
from skills.sites import SITES
from skills.websearch import ENGINES, WebSearchSkill, resolve_engine


class _Opener:
    """A stand-in for the browser: records URLs, never opens anything."""

    def __init__(self, result: bool = True) -> None:
        self.urls: list[str] = []
        self._result = result

    def __call__(self, url: str) -> bool:
        self.urls.append(url)
        return self._result


def _skill(opener=None, summarize=None, safety=None, **cfg) -> WebSearchSkill:
    skill = WebSearchSkill(summarize, config=WebSearchConfig(**cfg),
                           opener=opener or _Opener(), safety=safety)
    # Any DDG call on the browser path is a bug: the whole point is that we
    # never touch the network for a command.
    skill._search = lambda q: pytest.fail(f"searched DDG for {q!r} instead of opening")
    return skill


async def _summarize(query: str, block: str) -> str:
    return f"summary of {query}"


# --- the command path --------------------------------------------------------


@pytest.mark.asyncio
async def test_search_up_opens_the_browser_instead_of_answering():
    opener = _Opener()
    skill = _skill(opener)
    text = "Search up for Mazda RX-7"
    result = await skill.execute(SkillRequest(text=text, match=skill.match(text)))
    assert result.success
    assert opener.urls == ["https://www.google.com/search?q=Mazda+RX-7"]
    assert result.speech == "Searching Google for Mazda RX-7."
    assert result.data["opened"] is True


@pytest.mark.asyncio
async def test_the_phrasal_up_is_not_part_of_the_query():
    # "search up for X" captures "up for X"; both particles belong to the verb.
    opener = _Opener()
    skill = _skill(opener)
    text = "search up for the best beginner soldering iron please"
    await skill.execute(SkillRequest(text=text, match=skill.match(text)))
    assert opener.urls == [
        "https://www.google.com/search?q=best+beginner+soldering+iron"]


@pytest.mark.parametrize("text", [
    "google the nearest hardware store",
    "look up the Mazda RX-7",
    "search for pasta recipes",
])
@pytest.mark.asyncio
async def test_every_english_search_verb_is_a_command(text):
    opener = _Opener()
    skill = _skill(opener)
    result = await skill.execute(SkillRequest(text=text, match=skill.match(text)))
    assert len(opener.urls) == 1, text
    assert result.speech.startswith("Searching Google for "), text


@pytest.mark.parametrize("text", [
    # Every one of these sent junk into the query, invisible while the answer
    # was a spoken paragraph and plainly wrong once it's in the address bar:
    # "internet for the Mazda RX-7", "online for the…", "info on the…".
    "search up for Mazda RX-7",
    "search for the Mazda RX-7",
    "search the web for the Mazda RX-7",
    "search the internet for the Mazda RX-7",
    "search online for the Mazda RX-7",
    "search up some info on the Mazda RX-7",
    "google the Mazda RX-7",
    "look up the Mazda RX-7",
    "look into the Mazda RX-7",
    "find me info on the Mazda RX-7",
    "find information about the Mazda RX-7",
    "check the web for the Mazda RX-7",
])
@pytest.mark.asyncio
async def test_every_phrasing_sends_the_same_clean_query(text):
    opener = _Opener()
    skill = _skill(opener)
    match = skill.match(text)
    assert match is not None, text
    await skill.execute(SkillRequest(text=text, match=match))
    assert opener.urls == ["https://www.google.com/search?q=Mazda+RX-7"], text


@pytest.mark.parametrize("text", [
    # A search COMMAND that the verb list didn't own, so it fell to the LLM and
    # came back as a spoken paragraph — the same ask answered two ways.
    "look into the Mazda RX-7",
    "find me info on the Mazda RX-7",
    "find information about the Mazda RX-7",
    "check the web for the Mazda RX-7",
    "check online for the Mazda RX-7",
    "најди ми информации за мазда рх-7",
    "провери на интернет за мазда рх-7",
])
def test_the_other_ways_of_saying_go_and_look_are_commands(text):
    skill = WebSearchSkill(config=WebSearchConfig())
    match = skill.match(text)
    assert match is not None, text
    assert any(match.groupdict().get(g)
               for g in WebSearchSkill._COMMAND_GROUPS), text


@pytest.mark.parametrize("text", [
    "check the internet connection",      # not a search — "for" is mandatory
    "look into it",
    "look into this for me",
    "look into my inbox",
    "look up to your heroes",
    "look up when you walk",
    "google is a great company",
    "search your feelings",
])
def test_the_widened_verbs_do_not_steal_ordinary_speech(text):
    assert WebSearchSkill(config=WebSearchConfig()).match(text) is None, text


def test_every_capture_group_is_classified():
    # A pattern whose group is in neither list would search for "" — the
    # extraction loop reads these two tuples and nothing else.
    known = set(WebSearchSkill._COMMAND_GROUPS) | set(WebSearchSkill._QUESTION_GROUPS)
    declared = {name for p in WebSearchSkill.patterns for name in p.groupindex}
    assert declared == known, f"unclassified: {declared ^ known}"


@pytest.mark.asyncio
async def test_macedonian_command_opens_and_answers_in_macedonian():
    opener = _Opener()
    skill = _skill(opener)
    text = "барај мазда рх-7"
    result = await skill.execute(SkillRequest(text=text, match=skill.match(text)))
    assert len(opener.urls) == 1
    assert result.speech == "Пребарувам Google за мазда рх-7."


# --- the question path -------------------------------------------------------


@pytest.mark.asyncio
async def test_what_is_the_latest_on_is_a_question_not_a_command():
    opener = _Opener()
    skill = WebSearchSkill(_summarize, config=WebSearchConfig(), opener=opener)
    skill._search = lambda q: [{"title": "T", "body": "B"}]
    text = "what is the latest on the Mazda rotary engine"
    result = await skill.execute(SkillRequest(text=text, match=skill.match(text)))
    assert opener.urls == []
    assert result.speech.startswith("summary of ")


@pytest.mark.asyncio
async def test_a_question_reaching_the_skill_by_meaning_is_answered():
    # Semantic tier: no regex match, no args — the utterance IS the query.
    opener = _Opener()
    skill = WebSearchSkill(_summarize, config=WebSearchConfig(), opener=opener)
    skill._search = lambda q: [{"title": "GT", "body": "Masahiro Andoh"}]
    result = await skill.execute(
        SkillRequest(text="who composed the music for Gran Turismo"))
    assert opener.urls == []
    assert result.speech == "summary of who composed the music for Gran Turismo"


@pytest.mark.asyncio
async def test_the_models_tool_call_gets_text_not_a_window():
    # A window hands the model nothing to reason over, and a model with
    # nothing invents. Under the default scope the tool path never opens.
    opener = _Opener()
    skill = WebSearchSkill(None, config=WebSearchConfig(), opener=opener)
    skill._search = lambda q: [{"title": "T", "body": "B"}]
    result = await skill.execute(SkillRequest(
        text="search x", args={"query": "x"}, context={"via": "tool"}))
    assert opener.urls == []
    assert result.success and "T" in result.speech and "B" in result.speech


# --- browser_scope: always ---------------------------------------------------


@pytest.mark.asyncio
async def test_scope_always_opens_for_a_question_too():
    opener = _Opener()
    skill = _skill(opener, browser_scope="always")
    result = await skill.execute(
        SkillRequest(text="who composed the music for Gran Turismo"))
    assert len(opener.urls) == 1
    assert result.data["opened"] is True


@pytest.mark.asyncio
async def test_scope_always_still_hands_the_model_its_snippets():
    # The window goes up AND the text comes back: opening a browser must never
    # be the reason the model has nothing to answer from.
    opener = _Opener()
    skill = WebSearchSkill(None, config=WebSearchConfig(browser_scope="always"),
                           opener=opener)
    skill._search = lambda q: [{"title": "T", "body": "B"}]
    result = await skill.execute(SkillRequest(
        text="search x", args={"query": "x"}, context={"via": "tool"}))
    assert len(opener.urls) == 1
    assert "T" in result.speech and "B" in result.speech


# --- the ways it must fall back ----------------------------------------------


@pytest.mark.asyncio
async def test_open_in_browser_false_restores_the_old_behaviour():
    opener = _Opener()
    skill = WebSearchSkill(_summarize, config=WebSearchConfig(open_in_browser=False),
                           opener=opener)
    skill._search = lambda q: [{"title": "T", "body": "B"}]
    text = "search up for Mazda RX-7"
    result = await skill.execute(SkillRequest(text=text, match=skill.match(text)))
    assert opener.urls == []
    assert result.speech == "summary of Mazda RX-7"


@pytest.mark.asyncio
async def test_no_config_means_no_browser_at_all():
    # A bare WebSearchSkill() is what every other test in the suite builds.
    # It must never open a window as a side effect of this feature.
    skill = WebSearchSkill(_summarize)
    skill._search = lambda q: [{"title": "T", "body": "B"}]
    skill._opener = lambda url: pytest.fail(f"opened {url} with no config")
    text = "search up for Mazda RX-7"
    result = await skill.execute(SkillRequest(text=text, match=skill.match(text)))
    assert result.speech == "summary of Mazda RX-7"


@pytest.mark.asyncio
async def test_pc_control_off_answers_out_loud_instead_of_opening():
    # web_search is deliberately NOT controls_pc (that would bar it from the
    # semantic tier), so it honours the master switch itself.
    class _Safety:
        pc_control_enabled = False

    opener = _Opener()
    skill = WebSearchSkill(_summarize, config=WebSearchConfig(), opener=opener,
                           safety=_Safety())
    skill._search = lambda q: [{"title": "T", "body": "B"}]
    text = "search up for Mazda RX-7"
    result = await skill.execute(SkillRequest(text=text, match=skill.match(text)))
    assert opener.urls == []
    assert result.speech == "summary of Mazda RX-7"


@pytest.mark.asyncio
async def test_a_browser_that_refuses_never_gets_claimed_as_opened():
    # webbrowser.open returns False with no browser; browser_opener returns
    # False for a blocked host. Either way, saying "Searching Google" would be
    # the same fabricated 'done' the whole fixlist is about.
    opener = _Opener(result=False)
    skill = WebSearchSkill(_summarize, config=WebSearchConfig(), opener=opener)
    skill._search = lambda q: [{"title": "T", "body": "B"}]
    text = "search up for Mazda RX-7"
    result = await skill.execute(SkillRequest(text=text, match=skill.match(text)))
    assert opener.urls == ["https://www.google.com/search?q=Mazda+RX-7"]
    assert "Searching" not in result.speech
    assert result.speech == "summary of Mazda RX-7"


@pytest.mark.asyncio
async def test_a_query_that_cleans_away_to_nothing_asks_instead_of_opening():
    opener = _Opener()
    skill = _skill(opener)
    text = "search up"
    match = skill.match(text)
    if match is not None:                     # "search up" alone captures "up"
        result = await skill.execute(SkillRequest(text=text, match=match))
        assert opener.urls == []
        assert result.success is False


# --- engine resolution -------------------------------------------------------


def test_engine_defaults_to_google():
    assert resolve_engine("google").label == "Google"
    assert resolve_engine("").label == "Google"


def test_an_unknown_engine_falls_back_rather_than_breaking_search():
    # A typo in config.yaml costs you your preferred engine, not the feature.
    assert resolve_engine("gogle").label == "Google"


@pytest.mark.parametrize("name", sorted(ENGINES))
def test_every_shorthand_engine_resolves_to_a_usable_template(name):
    site = resolve_engine(name)
    assert "{q}" in site.search and site.search.startswith("https://")
    assert site.label


def test_any_site_table_entry_works_as_an_engine():
    # "engine: wikipedia" and your own skills.sites entries, aliases included.
    assert resolve_engine("wikipedia").key == "wikipedia"
    assert resolve_engine("википедија").key == "wikipedia"
    assert resolve_engine("github").key == "github"


def test_a_custom_url_template_is_accepted_verbatim():
    site = resolve_engine("https://searx.example.org/?q={q}")
    assert site.search == "https://searx.example.org/?q={q}"
    assert site.label == "searx.example.org"


@pytest.mark.asyncio
async def test_a_cyrillic_query_on_wikipedia_lands_on_the_mk_mirror():
    # build_url honours Site.search_mk, so the engine setting inherits it.
    opener = _Opener()
    skill = _skill(opener, engine="wikipedia")
    text = "барај охридско езеро"
    await skill.execute(SkillRequest(text=text, match=skill.match(text)))
    assert opener.urls and opener.urls[0].startswith("https://mk.wikipedia.org/")


@pytest.mark.asyncio
async def test_duckduckgo_engine_opens_duckduckgo():
    opener = _Opener()
    skill = _skill(opener, engine="duckduckgo")
    text = "search up for Mazda RX-7"
    result = await skill.execute(SkillRequest(text=text, match=skill.match(text)))
    assert opener.urls == ["https://duckduckgo.com/?q=Mazda+RX-7"]
    assert result.speech == "Searching DuckDuckGo for Mazda RX-7."


# --- wiring ------------------------------------------------------------------


def test_config_yaml_carries_the_section():
    cfg = load_settings().web_search
    assert cfg.open_in_browser is True
    assert cfg.browser_scope == "commands"
    assert resolve_engine(cfg.engine, SITES).label


def test_the_skill_is_not_controls_pc():
    # controls_pc would exclude it from the semantic tier (Router._semantic_safe),
    # and the semantic tier is exactly the path that must keep ANSWERING.
    from core.router import Router

    assert Router._semantic_safe(WebSearchSkill(config=WebSearchConfig()))
