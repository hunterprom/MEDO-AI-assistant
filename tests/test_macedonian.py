"""Macedonian commands on the fast path — apps, websites, files, web search.

The fast path is regex, so bilingualism isn't something the LLM can paper over:
a skill that only knows "open" is deaf to "отвори". These tests pin the spoken
Macedonian forms MEDO is actually given by faster-whisper (Cyrillic), including
the imperative clitics ("отвори ми го хром") that pile up between the verb and
its object, and check that the reply comes back in the language it was asked in.
"""

from __future__ import annotations

import pytest

from core import mk
from core.config import load_settings
from core.safety import PathWhitelist
from main import Announcer, build_registry
from skills.apps import AppsSkill
from skills.base import SkillRequest
from skills.files import FilesSkill
from skills.web_open import OpenWebsiteSkill
from skills.websearch import WebSearchSkill


@pytest.fixture
def registry():
    return build_registry(load_settings(), Announcer())


# --- the shared vocabulary ---------------------------------------------------


def test_is_cyrillic_separates_the_languages():
    assert mk.is_cyrillic("отвори хром")
    assert mk.is_cyrillic("барај мачки на јутјуб")
    assert not mk.is_cyrillic("open chrome")
    assert not mk.is_cyrillic("")


# --- apps --------------------------------------------------------------------


@pytest.fixture
def apps():
    return AppsSkill({
        "chrome": {"windows": "start chrome", "darwin": "open -a 'Google Chrome'",
                   "linux": "google-chrome"},
        "spotify": {"windows": "start spotify", "darwin": "open -a Spotify",
                    "linux": "spotify"},
        "terminal": {"windows": "start cmd", "darwin": "open -a Terminal",
                     "linux": "x-terminal-emulator"},
    })


@pytest.mark.parametrize("phrase,app,action", [
    ("отвори хром", "chrome", "open"),
    ("отвори ми го хром", "chrome", "open"),          # clitics between the two
    ("стартувај спотифај", "spotify", "open"),
    ("вклучи го терминалот", "terminal", "open"),
    ("пушти спотифај", "spotify", "open"),
    ("затвори хром", "chrome", "close"),
    ("исклучи го спотифај", "spotify", "close"),
])
@pytest.mark.asyncio
async def test_macedonian_app_commands(apps, phrase, app, action, monkeypatch):
    launched, killed = [], []
    monkeypatch.setattr("skills.apps.run_detached", lambda cmd: launched.append(cmd))
    monkeypatch.setattr(AppsSkill, "_close", lambda self, key: killed.append(key))

    m = apps.match(phrase)
    assert m is not None, phrase
    r = await apps.execute(SkillRequest(text=phrase, match=m))
    assert r.success, phrase
    assert r.data["app"] == app and r.data["action"] == action
    # Answered in Macedonian, not English.
    assert r.speech.startswith("Отворам" if action == "open" else "Затворам")


def test_macedonian_app_names_do_not_break_english(apps):
    assert apps.match("open chrome") is not None
    assert apps.match("close spotify") is not None
    assert apps.match("what time is it") is None


# --- websites ----------------------------------------------------------------


@pytest.fixture
def web():
    s = OpenWebsiteSkill(opener=lambda url: s.opened.append(url) or True)
    s.opened = []
    return s


@pytest.mark.asyncio
@pytest.mark.parametrize("phrase,url", [
    ("отвори јутјуб", "https://www.youtube.com"),
    ("оди на редит", "https://www.reddit.com"),
    ("отвори гмаил", "https://mail.google.com"),
    ("отвори тинкеркад", "https://tinkercad.com"),
    ("отвори ја страната гитхаб", "https://github.com"),
    # Macedonian speech spells a URL out: "тинкеркад точка ком".
    ("отвори тинкеркад точка ком", "https://tinkercad.com"),
])
async def test_macedonian_opens_websites(web, phrase, url):
    m = web.match(phrase)
    assert m is not None, phrase
    r = await web.execute(SkillRequest(text=phrase, match=m))
    assert web.opened == [url], phrase
    assert r.speech.startswith("Отворам")


@pytest.mark.asyncio
async def test_macedonian_browser_search(web):
    phrase = "барај ардуино сензори во прелистувач"
    m = web.match(phrase)
    assert m is not None
    await web.execute(SkillRequest(text=phrase, match=m))
    assert web.opened[0].startswith("https://duckduckgo.com/?q=")


def test_macedonian_does_not_steal_app_launches(web):
    # "хром" is an app, not a site — AppsSkill must keep it.
    for phrase in ("отвори хром", "пушти музика", "затвори спотифај"):
        assert web.match(phrase) is None, phrase


# --- files -------------------------------------------------------------------


@pytest.fixture
def files(tmp_path):
    (tmp_path / "izvestaj-2026.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    return FilesSkill(PathWhitelist([str(tmp_path)]))


@pytest.mark.asyncio
@pytest.mark.parametrize("phrase", [
    "најди датотека izvestaj",
    "најди ја датотеката izvestaj",
    "барај фајл izvestaj",
    "најди izvestaj на компјутерот",
])
async def test_macedonian_file_search(files, phrase):
    m = files.match(phrase)
    assert m is not None, phrase
    r = await files.execute(SkillRequest(text=phrase, match=m))
    assert r.success and r.data["count"] == 1
    assert r.speech.startswith("Најдов")


@pytest.mark.asyncio
async def test_macedonian_file_open(files, monkeypatch):
    opened = []
    monkeypatch.setattr("skills.files.open_path", lambda p: opened.append(p))
    phrase = "отвори ја датотеката izvestaj"
    m = files.match(phrase)
    assert m is not None
    r = await files.execute(SkillRequest(text=phrase, match=m))
    assert r.success and len(opened) == 1
    assert r.speech.startswith("Отворам")


@pytest.mark.asyncio
async def test_english_file_search_still_speaks_english(files):
    phrase = "find the file izvestaj"
    r = await files.execute(SkillRequest(text=phrase, match=files.match(phrase)))
    assert r.speech.startswith("Found")


# --- web search --------------------------------------------------------------


@pytest.mark.parametrize("phrase", [
    "барај рецепт за пица",
    "пребарај цена на филамент",
    "гугни колку чини 3д принтер",
    "што е ново за вештачка интелигенција",
])
def test_macedonian_web_search_patterns(phrase):
    assert WebSearchSkill().match(phrase) is not None, phrase


@pytest.mark.asyncio
async def test_macedonian_offline_message_is_macedonian(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr("ddgs.DDGS", _boom)
    phrase = "барај рецепт за пица"
    skill = WebSearchSkill(summarize=None)
    r = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert r.success is False and "офлајн" in r.speech


# --- routing (the part that actually breaks) ---------------------------------


@pytest.mark.parametrize("phrase,skill_name", [
    # Macedonian lands on the same skills as its English twin.
    ("отвори хром", "apps"),
    ("затвори спотифај", "apps"),
    ("отвори јутјуб", "open_website"),
    ("оди на редит", "open_website"),
    ("барај мачки на јутјуб", "site_search"),
    ("најди ми игра на стим", "site_search"),
    ("провери во пошта за сметката", "site_search"),
    ("најди ја датотеката извештај", "files"),
    ("барај рецепт за пица", "web_search"),
    # …and the English side is untouched.
    ("open chrome", "apps"),
    ("open youtube", "open_website"),
    ("search drone motors on youtube", "site_search"),
    ("check my email for the invoice", "site_search"),
    ("search my computer for invoice", "files"),
    ("search for pasta recipes", "web_search"),
    ("what time is it", "datetime"),
])
def test_registry_routes_both_languages(registry, phrase, skill_name):
    hit = registry.find_match(phrase)
    assert hit is not None, f"nothing matched: {phrase!r}"
    assert hit[0].name == skill_name, f"{phrase!r} went to {hit[0].name}"
