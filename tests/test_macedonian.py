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
from core.config import NewsConfig, WeatherConfig, load_settings
from core.safety import PathWhitelist
from main import Announcer, build_registry
from skills.apps import AppsSkill
from skills.base import SkillRequest
from skills.datetime_skill import DateTimeSkill
from skills.files import FilesSkill
from skills.news import NewsSkill
from skills.vision_skill import SeeCameraSkill, SeeScreenSkill
from skills.weather import WeatherSkill
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


def test_to_latin_romanises_place_names():
    # Open-Meteo's geocoder indexes Latin names only.
    assert mk.to_latin("Скопје") == "Skopje"
    assert mk.to_latin("Битола") == "Bitola"
    assert mk.to_latin("Охрид") == "Ohrid"
    assert mk.to_latin("Ѓорче") == "Gjorche"      # digraph, capitalised
    assert mk.to_latin("Skopje") == "Skopje"      # already Latin, untouched


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


# --- news / weather / time ----------------------------------------------------
#
# These three are why the bilingual gap actually bit: a Macedonian news or
# weather question matched no pattern, fell through to the LLM path, and — on a
# CLI-agent provider, which by design gets none of MEDO's tool schemas — came
# back as "I don't have permission to search the web in this session". The fast
# path has to answer these itself, whatever brain is loaded.


@pytest.mark.parametrize("phrase", [
    "вести", "најсвежи вести за Скопје", "дај ми ги вестите",
    "новости", "наслови", "што има ново",
])
def test_macedonian_news_patterns(phrase):
    assert NewsSkill(NewsConfig(feeds=["x"])).match(phrase) is not None, phrase


def test_news_does_not_steal_a_topic_web_search():
    # "што е ново за X" names a topic — that's a web search, not the headlines.
    assert NewsSkill(NewsConfig(feeds=["x"])).match(
        "што е ново за вештачка интелигенција") is None


def test_macedonian_news_uses_macedonian_feeds():
    config = NewsConfig(feeds=["https://en.example/rss"],
                        feeds_mk=["https://mk.example/rss"])
    skill = NewsSkill(config)
    assert skill._pick_feeds(speak_mk=True) == ["https://mk.example/rss"]
    assert skill._pick_feeds(speak_mk=False) == ["https://en.example/rss"]
    # No Macedonian sources configured => fall back rather than answer nothing.
    assert NewsSkill(NewsConfig(feeds=["https://en.example/rss"]))._pick_feeds(
        speak_mk=True) == ["https://en.example/rss"]


@pytest.mark.parametrize("phrase,city", [
    ("какво е времето", None),
    ("какво е времето во Скопје", "Скопје"),
    ("времето во Битола", "Битола"),
    ("прогноза за Охрид", "Охрид"),
    ("колку степени е надвор", None),
    ("дали ќе врне утре", None),
])
def test_macedonian_weather_patterns_capture_cyrillic_cities(phrase, city):
    m = WeatherSkill(WeatherConfig()).match(phrase)
    assert m is not None, phrase
    gd = m.groupdict()
    got = next((v for k, v in gd.items() if k.startswith("city") and v), None)
    assert (got.strip() if got else None) == city


def test_weather_leaves_bare_vreme_alone():
    # "време" is both "weather" and "time" — claiming it would break the clock.
    assert WeatherSkill(WeatherConfig()).match("колку е часот") is None


@pytest.mark.asyncio
async def test_weather_geocode_retries_romanised():
    """Cyrillic finds nothing, so the romanised name is tried before giving up."""
    tried: list[str] = []

    class _Client:
        async def get(self, url, params):
            tried.append(params["name"])
            hit = params["name"] == "Skopje"

            class _R:
                @staticmethod
                def raise_for_status(): ...
                @staticmethod
                def json():
                    return {"results": [{"latitude": 42.0, "longitude": 21.4,
                                         "name": "Skopje"}]} if hit else {"results": []}
            return _R()

    geo = await WeatherSkill(WeatherConfig())._geocode(_Client(), "Скопје")
    assert tried == ["Скопје", "Skopje"]
    assert geo is not None and geo[2] == "Skopje"


@pytest.mark.asyncio
async def test_macedonian_time_and_date_answer_in_macedonian():
    skill = DateTimeSkill()
    time_r = await skill.execute(SkillRequest(text="колку е часот",
                                              match=skill.match("колку е часот")))
    date_r = await skill.execute(SkillRequest(text="кој датум е денес",
                                              match=skill.match("кој датум е денес")))
    assert time_r.speech.startswith("Часот е") and time_r.data["kind"] == "time"
    assert date_r.speech.startswith("Денес е") and date_r.data["kind"] == "date"
    # English is untouched.
    en = await skill.execute(SkillRequest(text="what time is it",
                                          match=skill.match("what time is it")))
    assert en.speech.startswith("It's")


# --- seeing --------------------------------------------------------------------


@pytest.mark.parametrize("phrase", [
    "што гледаш на екранот",
    "што има на мојот екран",
    "прочитај го екранот",
    "опиши го екранот",
    "дали го гледаш екранот",
    # Exactly what Whisper handed the router when this broke — misspelling and
    # all ("твоот" for "твојот"), which is why one loose word is allowed before
    # "екран" instead of an exact possessive.
    "Какото се свиѓа што гледаш на екранот, што гледаш на твоот екран.",
])
def test_macedonian_screen_requests(phrase):
    assert SeeScreenSkill(load_settings()).match(phrase) is not None, phrase


@pytest.mark.parametrize("phrase", [
    "што гледаш", "што гледаш сега", "дали ме гледаш", "погледни ме",
])
def test_macedonian_camera_requests(phrase):
    assert SeeCameraSkill(load_settings()).match(phrase) is not None, phrase


def test_camera_does_not_swallow_screen_requests():
    """SeeCamera is registered first, so its bare "што гледаш" must yield."""
    camera = SeeCameraSkill(load_settings())
    assert camera.match("што гледаш на екранот") is None
    assert camera.match("прочитај го екранот") is None


def test_vision_prompt_asks_for_macedonian_only_when_asked_in_it():
    from skills.vision_skill import ANSWER_MK

    assert ANSWER_MK.strip().startswith("Answer in")  # instruction stays English


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
    # The exact request that used to reach the LLM and get refused.
    ("најсвежи вести за Скопје", "news"),
    ("какво е времето во Скопје", "weather"),
    ("колку е часот", "datetime"),
    ("што гледаш на екранот", "see_screen"),
    ("дали ме гледаш", "see_camera"),
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
