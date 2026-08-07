"""Regression tests for the 2026-08 Macedonian transcript.

The session that prompted these was almost entirely in Macedonian and almost
entirely wrong::

    YOU   отвори стима                     -> "да, можам да го instaliram."
    YOU   Направи апликација за … прогноза  -> "Во Skopje е 34 степени и ведро."
    YOU   Направи апликација.               -> "Морам да извежdam ' winget' …"
    YOU   … може да ми направиш 3D модел …  -> "Нема possibilities."
    YOU   Збори на македонски …             -> "I can help with that! …"

Three separate defects hid behind one symptom:

1. **The fast path was English-only.** Every making verb, part noun and app
   name pattern existed in English and nowhere else, so Macedonian requests
   fell through to the LLM, which narrated or denied instead of acting.
2. **A too-greedy weather pattern** claimed a bare "прогноза", so a request to
   BUILD a forecast app was answered with the current temperature.
3. **Nothing checked the reply.** "Reply in Macedonian" is a suggestion to a
   small local model; it answered in English, or welded Latin stems into
   Cyrillic words ("извежdam").

Each test below pins one of those at the layer that failed.
"""

from __future__ import annotations

import pytest

from core import mk, reply_language
from core.config import WeatherConfig, load_settings
from core.events import EventBus
from core.router import Router
from llm.client import OllamaClient
from skills.app_builder_skill import _MAKE_APP, _MAKE_APP_MK
from skills.apps import AppsSkill
from skills.base import SkillRegistry
from skills.maker_studio import CodeBuildSkill, DesignCircuitSkill, Model3DSkill
from skills.weather import WeatherSkill

# --- 1. declension: "отвори стимА" is the same app as "стим" -------------------

_APPS = {
    "steam": {"windows": "start steam://open/main"},
    "spotify": {"windows": "start spotify"},
    "notepad": {"windows": "start notepad"},
    "chrome": {"windows": "start chrome"},
}


def _launch(text: str) -> str | None:
    """What AppsSkill would actually open for a spoken line."""
    skill = AppsSkill(_APPS)
    m = skill.match(text)
    if m is None:
        return None
    return skill._resolve_key(m.groupdict().get("app") or "")


@pytest.mark.parametrize(
    "spoken",
    ["отвори стим", "отвори стима", "отвори го стимот", "стартувај стим",
     "отвори ми го стимот"],
)
def test_declined_app_names_still_launch(spoken):
    # The transcript's "отвори стима" reached no skill at all: \b after the
    # name refused the definite article, so the launch fell through to the LLM.
    assert _launch(spoken) == "steam", spoken


def test_close_still_closes_a_declined_name():
    skill = AppsSkill(_APPS)
    m = skill.match("затвори го стимот")
    assert m is not None and m.groupdict().get("mk_close")


def test_english_launch_is_unchanged():
    assert _launch("open steam") == "steam"
    assert _launch("open up spotify") == "spotify"


def test_undeclined_keeps_the_spoken_form_first():
    # A stem is only ever a fallback — a real name that happens to end in "а"
    # must still resolve as itself.
    assert mk.undeclined("стимот")[0] == "стимот"
    assert "стим" in mk.undeclined("стима")
    assert mk.undeclined("на") == ["на"]          # too short to strip


# --- 2. "направи апликација…" builds; it does not report the weather ----------

def _weather():
    return WeatherSkill(WeatherConfig())


@pytest.mark.parametrize(
    "spoken",
    ["Направи апликација за следене на време прогноза.",
     "Направи апликација за време прогноза.",
     "направи ми апликација за прогноза",
     "make me an app that tracks the weather"],
)
def test_build_request_is_not_a_weather_question(spoken):
    assert _weather().match(spoken) is None, spoken


@pytest.mark.parametrize(
    "spoken",
    ["каква е прогнозата", "прогнозата за Скопје", "какво е времето во Скопје",
     "колку степени е", "what's the weather", "will it rain tomorrow"],
)
def test_real_weather_questions_still_reach_the_weather_skill(spoken):
    assert _weather().match(spoken) is not None, spoken


@pytest.mark.parametrize(
    "spoken,spec",
    [("Направи апликација за следене на време прогноза.", "следене"),
     ("направи ми една програма што ги следи трошоците", "трошоц"),
     ("изгради скрипта за преименување фајлови", "преименување")],
)
def test_macedonian_make_app_captures_the_spec(spoken, spec):
    m = _MAKE_APP_MK.search(spoken)
    assert m is not None, spoken
    assert spec in (m.groupdict().get("spec") or "")


def test_bare_macedonian_make_app_matches_with_no_spec():
    # "Направи апликација." must reach the builder and get asked what to build,
    # not reach the LLM and get a half-translated excuse about winget.
    m = _MAKE_APP_MK.search("Направи апликација.")
    assert m is not None
    assert not (m.groupdict().get("spec") or "").strip()


def test_english_make_app_is_unchanged():
    assert _MAKE_APP.search("build me an app that logs expenses") is not None


# --- 3. Maker Studio answers in Macedonian too --------------------------------

def _desc(skill_cls, text: str) -> str | None:
    for pattern in skill_cls.patterns:
        m = pattern.search(text)
        if m:
            return m.groupdict().get("desc")
    return None


@pytest.mark.parametrize(
    "spoken,expected",
    [("Добро, може да ми направиш 3D модел од телефон.", "телефон"),
     ("направи ми 3д модел на држач за телефон", "држач"),
     ("испечати ми држач за камера", "држач"),
     ("тродимензионален модел на куќиште", "куќиште")],
)
def test_macedonian_3d_requests_reach_the_studio(spoken, expected):
    # MEDO denied being able to make 3D models while the Studio sat behind it —
    # purely because every pattern was English. The conjugated verb ("направиШ")
    # is why "3д модел" itself has to be the cue.
    assert expected in (_desc(Model3DSkill, spoken) or ""), spoken


def test_macedonian_circuit_and_code_requests_route():
    assert "LED" in (_desc(DesignCircuitSkill, "нацртај ми шема за LED со отпорник") or "")
    assert "сензор" in (_desc(DesignCircuitSkill, "струјно коло за сензор за температура") or "")
    assert "преименување" in (_desc(CodeBuildSkill, "напиши ми скрипта за преименување фајлови") or "")


@pytest.mark.parametrize(
    "spoken",
    ["design a business case", "make a video clip", "a stand-up meeting"],
)
def test_english_non_parts_still_do_not_reach_the_studio(spoken):
    skill = Model3DSkill.__new__(Model3DSkill)   # patterns only; no engine needed
    assert Model3DSkill.match(skill, spoken) is None, spoken


def test_english_part_requests_are_unchanged():
    assert _desc(Model3DSkill, "3d print a phone stand") == "phone stand"
    assert _desc(Model3DSkill, "print a wall mount") == "wall mount"


# --- 4. the reply is checked, not just requested ------------------------------

@pytest.mark.parametrize(
    "reply",
    ['да, можам да го instaliram.',
     'Морам да извежdam " winget" и тогаш можеш да ме направи апликација',
     "Нема possibilities.",
     "I can help with that! Can you please provide more context or specify what "
     "kind of assistance you need?"],
)
def test_off_language_replies_are_caught(reply):
    assert reply_language.off_language(reply, "mk"), reply


@pytest.mark.parametrize(
    "reply",
    ["Во Скопје е 34 степени и ведро.",
     "Отворам го Steam.",
     "Готово — ја отворив папката.",
     # A brand or command name keeps its own spelling; one Latin token in an
     # otherwise Macedonian sentence is normal speech, not a broken reply.
     "Инсталирај го со winget, потоа пробај повторно и ќе работи како што треба.",
     "Здраво! Како можам да помогнам?"],
)
def test_correct_macedonian_replies_are_left_alone(reply):
    assert not reply_language.off_language(reply, "mk"), reply


def test_english_replies_to_english_are_left_alone():
    assert not reply_language.off_language("It's 34 degrees and clear in Skopje.", "en")


def test_no_detected_language_means_no_judgement():
    # Typed input carries no language code; we never asked for one, so we must
    # not second-guess the reply.
    assert not reply_language.off_language("Нема possibilities.", None)
    assert reply_language.violation_score("Нема possibilities.", None) == 0


def test_welded_words_score_worse_than_merely_foreign_ones():
    welded = reply_language.violation_score("да го извежdam", "mk")
    foreign = reply_language.violation_score("да го инсталирам instaliram", "mk")
    assert welded > foreign > 0
    assert reply_language.mixed_script_words("извежdam") == ["извежdam"]


# --- 5. the router re-asks once, and never returns something worse ------------

class _FakeLLM:
    """Returns each queued reply in turn; records how many times it was asked."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.calls = 0

    async def chat(self, model, messages, **kw):
        self.calls += 1
        content = self._replies.pop(0) if self._replies else ""
        return {"content": content}


def _router() -> Router:
    settings = load_settings()
    return Router(settings, SkillRegistry(), OllamaClient(settings.llm), EventBus())


@pytest.mark.asyncio
async def test_off_language_reply_is_re_asked_and_replaced():
    llm = _FakeLLM("Го отворам Стим.")
    fixed = await _router()._repair_language(
        llm, "m", [], "да, можам да го instaliram.", "mk")
    assert fixed == "Го отворам Стим."
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_a_clean_reply_is_never_re_asked():
    llm = _FakeLLM("should not be used")
    fixed = await _router()._repair_language(
        llm, "m", [], "Го отворам Стим.", "mk")
    assert fixed is None
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_a_retry_that_is_no_better_keeps_the_original():
    # The whole point of scoring both attempts: a second bad answer must not
    # replace the first one.
    llm = _FakeLLM("Still totally in English, I'm afraid.")
    assert await _router()._repair_language(
        llm, "m", [], "Нема possibilities.", "mk") is None


@pytest.mark.asyncio
async def test_a_failing_retry_keeps_the_original_reply():
    class _Broken:
        async def chat(self, *a, **kw):
            raise RuntimeError("ollama went away")

    # We already have an answer; a failed repair must not lose it or raise.
    assert await _router()._repair_language(
        _Broken(), "m", [], "Нема possibilities.", "mk") is None


@pytest.mark.asyncio
async def test_repair_is_skipped_without_a_detected_language():
    llm = _FakeLLM("unused")
    assert await _router()._repair_language(llm, "m", [], "anything", None) is None
    assert llm.calls == 0


# --- 6. a Macedonian sentence should not contain a Latin city -----------------

@pytest.mark.parametrize(
    "latin,cyrillic",
    [("Skopje", "Скопје"), ("Bitola", "Битола"), ("Ohrid", "Охрид"),
     ("London", "Лондон")],
)
def test_city_names_come_back_in_cyrillic(latin, cyrillic):
    assert mk.to_cyrillic(latin) == cyrillic


@pytest.mark.parametrize("latin", ["New York", "Washington", "Tokyo"])
def test_untransliterable_cities_keep_their_latin_spelling(latin):
    # Romanisation is lossy, so the reverse refuses rather than guessing: a "w"
    # or "y" has no Macedonian letter, and a half-converted name is worse than
    # an honest Latin one.
    assert mk.to_cyrillic(latin) == ""


def test_already_cyrillic_text_is_left_alone():
    assert mk.to_cyrillic("Скопје") == ""       # nothing to do; caller keeps it
    assert mk.to_latin("Скопје") == "Skopje"


# --- 7. garbled speech is a reason to ask, not to invent ----------------------

def test_prompt_forbids_inventing_things_to_fit_misheard_words():
    # "О-пен-с-теам" (spoken "open Steam") produced a confident description of
    # "Open Pen Team, a popular accessibility tool" — a product that does not
    # exist. The rule that was missing is an explicit one about garbled input.
    from llm.prompts import system_prompt

    text = system_prompt(load_settings().personality).lower()
    assert "garbled" in text
    assert "never invent a product" in text
    assert "ask which one instead of opening" in text
