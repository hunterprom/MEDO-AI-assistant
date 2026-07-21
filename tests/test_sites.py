"""Site-scoped search: "search X on YouTube" opens YouTube's own results."""

from __future__ import annotations

import pytest

from skills.base import SkillRequest
from skills.sites import (
    SiteSearchSkill,
    build_url,
    clean_query,
    load_sites,
    resolve_site,
)


@pytest.fixture
def skill():
    """Skill with a recording opener instead of a real browser."""
    s = SiteSearchSkill(opener=lambda url: s.opened.append(url) or True)
    s.opened = []
    return s


async def _run(skill, text):
    m = skill.match(text)
    assert m is not None, f"no pattern matched: {text!r}"
    return await skill.execute(SkillRequest(text=text, match=m))


# --- the site table ----------------------------------------------------------


def test_resolve_site_by_key_label_and_alias():
    assert resolve_site("youtube").key == "youtube"
    assert resolve_site("YouTube").key == "youtube"
    assert resolve_site("  You Tube. ").key == "youtube"   # spoken, punctuated
    assert resolve_site("yt").key == "youtube"
    assert resolve_site("email").key == "gmail"            # what people say
    assert resolve_site("nowhere") is None


def test_resolve_site_understands_macedonian():
    # Whisper writes site names in Cyrillic when the user speaks Macedonian.
    for spoken, key in (("јутјуб", "youtube"), ("редит", "reddit"),
                        ("стим", "steam"), ("гитхаб", "github"),
                        ("пошта", "gmail"), ("мејл", "gmail"),
                        ("википедија", "wikipedia")):
        assert resolve_site(spoken).key == key, spoken


def test_build_url_escapes_for_the_template_style():
    # Query-string templates want "+", path templates want "%20".
    assert build_url(resolve_site("youtube"), "drone motors") == \
        "https://www.youtube.com/results?search_query=drone+motors"
    assert build_url(resolve_site("spotify"), "pink floyd") == \
        "https://open.spotify.com/search/pink%20floyd"


def test_cyrillic_query_uses_the_macedonian_wikipedia():
    assert "mk.wikipedia.org" in build_url(resolve_site("wikipedia"), "Скопје")
    assert "en.wikipedia.org" in build_url(resolve_site("wikipedia"), "einstein")


def test_clean_query_strips_spoken_politeness():
    assert clean_query("drone motors please") == "drone motors"
    assert clean_query("  cheap servos, for me. ") == "cheap servos"
    assert clean_query("резервни делови те молам") == "резервни делови"


def test_config_can_add_and_override_sites():
    sites = load_sites({
        "pazar3": {"label": "Pazar3", "search": "https://www.pazar3.mk/oglasi?q={q}",
                   "home": "https://www.pazar3.mk", "aliases": ["пазар3"]},
        "youtube": {"label": "YouTube", "search": "https://my.mirror/?q={q}",
                    "home": "https://my.mirror"},
    })
    assert resolve_site("пазар3", sites).key == "pazar3"
    assert resolve_site("youtube", sites).search == "https://my.mirror/?q={q}"
    # A malformed entry is skipped, never fatal at startup.
    assert resolve_site("junk", load_sites({"junk": {"label": "no search url"}})) is None


# --- patterns ----------------------------------------------------------------


@pytest.mark.parametrize("phrase", [
    "search drone motors on youtube",
    "search for drone motors on youtube",
    "search youtube for drone motors",
    "youtube search drone motors",
    "find sniper elite on steam",
    "look up quadruped robots on reddit",
    "check my email for the invoice",
    "search bearings on amazon",
    "look up einstein on wikipedia",
    "барај мачки на јутјуб",
    "најди ми игра на стим",
    "барај на јутјуб како да печатам",
    "пребарај на редит за роботи",
    "провери во пошта за сметката",
])
def test_matches_site_searches_in_both_languages(skill, phrase):
    assert skill.match(phrase) is not None, phrase


@pytest.mark.parametrize("phrase", [
    "search for pasta recipes",          # plain web search
    "open youtube",                      # bare open -> open_website
    "отвори јутјуб",
    "open chrome",                       # app launch
    "search my documents for arduino",   # documents RAG
    "find the file report",              # local files
    "set a timer for 5 minutes",
    "what is the weather",
    "барај рецепт за пица",              # no site named -> web search
])
def test_does_not_steal_other_skills(skill, phrase):
    assert skill.match(phrase) is None, phrase


# --- execution ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_opens_the_sites_own_search(skill):
    r = await _run(skill, "search drone motors on youtube")
    assert r.success
    assert skill.opened == ["https://www.youtube.com/results?search_query=drone+motors"]
    assert "YouTube" in r.speech and "drone motors" in r.speech


@pytest.mark.asyncio
async def test_email_search_reaches_gmail(skill):
    await _run(skill, "check my email for the invoice")
    assert skill.opened[0].startswith("https://mail.google.com/mail/u/0/#search/")


@pytest.mark.asyncio
async def test_macedonian_request_gets_a_macedonian_reply(skill):
    r = await _run(skill, "барај мачки на јутјуб")
    assert skill.opened[0].startswith("https://www.youtube.com/results")
    assert r.speech.startswith("Пребарувам")


@pytest.mark.asyncio
async def test_site_without_a_query_just_opens_it(skill):
    r = await skill.execute(SkillRequest(text="", args={"site": "reddit"}))
    assert skill.opened == ["https://www.reddit.com"]
    assert "Opening" in r.speech


@pytest.mark.asyncio
async def test_unknown_site_fails_cleanly(skill):
    r = await skill.execute(SkillRequest(text="", args={"site": "myspace"}))
    assert r.success is False and skill.opened == []


@pytest.mark.asyncio
async def test_missing_browser_is_reported(skill):
    s = SiteSearchSkill(opener=lambda url: False)  # webbrowser found nothing
    r = await s.execute(SkillRequest(text="", args={"site": "youtube", "query": "x"}))
    assert r.success is False and "browser" in r.speech.lower()


def test_is_gated_by_the_pc_control_switch(skill):
    # Opening a browser acts on the machine, so the HUD switch must cover it.
    assert skill.controls_pc is True


def test_tool_schema_offers_the_known_sites(skill):
    schema = skill.tool_schema()["function"]
    assert schema["name"] == "site_search"
    enum = schema["parameters"]["properties"]["site"]["enum"]
    assert {"youtube", "gmail", "reddit", "steam", "github"} <= set(enum)
