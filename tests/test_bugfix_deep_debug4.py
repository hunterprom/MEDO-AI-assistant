"""Deep-debug loop, iteration 4 — content-skill args/parse bugs.

Each test pins a bug found by the iter-4 audit of the content skills:

  APPS1    AppsSkill ignored request.args, so every LLM tool-call failed with
           "I didn't catch which app." (match is None on the tool path).
  NEWS1    The advertised `count` parameter was ignored; news(count=10) always
           returned the fixed default.
  WEATHER1 A non-JSON 200 (captive portal) raised JSONDecodeError past the
           httpx-only handler; a shape-shifted JSON crashed on data["current"].
"""

from __future__ import annotations

import asyncio

import pytest

from skills.base import SkillRequest


# --- APPS1 -------------------------------------------------------------------

def test_apps_actuates_on_llm_tool_path(monkeypatch):
    """AppsSkill must open the app from request.args, not only request.match."""
    import skills.apps as apps_mod

    launched: list = []
    monkeypatch.setattr(apps_mod, "run_detached", lambda cmd: launched.append(cmd))

    skill = apps_mod.AppsSkill({"chrome": {"windows": "start chrome",
                                           "darwin": "open -a Google Chrome",
                                           "linux": "google-chrome"}})
    # LLM tool path: structured args, match is None.
    req = SkillRequest(text="open chrome", match=None,
                       args={"action": "open", "app": "chrome"})
    result = asyncio.run(skill.execute(req))

    assert result.success, result.speech
    assert result.data.get("app") == "chrome"
    assert launched, "run_detached was never called — the app did not open"


def test_apps_unknown_app_on_tool_path_is_graceful(monkeypatch):
    import skills.apps as apps_mod
    monkeypatch.setattr(apps_mod, "run_detached", lambda cmd: None)
    skill = apps_mod.AppsSkill({"chrome": {"windows": "start chrome"}})
    req = SkillRequest(text="open steam", match=None,
                       args={"action": "open", "app": "steam"})
    result = asyncio.run(skill.execute(req))
    assert not result.success
    assert "steam" in result.speech.lower()


def test_apps_fast_path_still_works(monkeypatch):
    """The regex fast path must be unaffected by the args-first change."""
    import skills.apps as apps_mod
    launched: list = []
    monkeypatch.setattr(apps_mod, "run_detached", lambda cmd: launched.append(cmd))
    skill = apps_mod.AppsSkill({"chrome": {"windows": "start chrome"}})
    m = skill.match("open chrome")
    assert m is not None
    result = asyncio.run(skill.execute(SkillRequest(text="open chrome", match=m)))
    assert result.success and launched


# --- NEWS1 -------------------------------------------------------------------

def _make_news_skill(entries: int):
    import skills.news as news_mod

    class _Resp:
        content = b"<rss></rss>"

        def raise_for_status(self):
            return None

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            return _Resp()

    class _Parsed:
        pass

    parsed = _Parsed()
    parsed.entries = [{"title": f"Headline {i}"} for i in range(entries)]

    return news_mod, _Client, parsed


def test_news_honours_count_arg(monkeypatch):
    news_mod, client_cls, parsed = _make_news_skill(entries=30)
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", client_cls)
    import feedparser
    monkeypatch.setattr(feedparser, "parse", lambda content: parsed)

    from core.config import NewsConfig
    skill = news_mod.NewsSkill(NewsConfig(feeds=["http://feed"]), max_items=5)
    req = SkillRequest(text="give me the top 10 headlines", args={"count": 10})
    result = asyncio.run(skill.execute(req))
    assert result.success, result.speech
    assert result.data.get("count") == 10, "count arg was ignored"


def test_news_count_arg_clamped_and_string_tolerant(monkeypatch):
    news_mod, client_cls, parsed = _make_news_skill(entries=30)
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", client_cls)
    import feedparser
    monkeypatch.setattr(feedparser, "parse", lambda content: parsed)

    from core.config import NewsConfig
    skill = news_mod.NewsSkill(NewsConfig(feeds=["http://feed"]), max_items=5)
    # A string count (model habit) must not crash, and 999 must clamp to <=20.
    req = SkillRequest(text="all the news", args={"count": "999"})
    result = asyncio.run(skill.execute(req))
    assert result.success
    assert result.data.get("count") == 20


def test_news_default_without_count(monkeypatch):
    news_mod, client_cls, parsed = _make_news_skill(entries=30)
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", client_cls)
    import feedparser
    monkeypatch.setattr(feedparser, "parse", lambda content: parsed)

    from core.config import NewsConfig
    skill = news_mod.NewsSkill(NewsConfig(feeds=["http://feed"]), max_items=5)
    result = asyncio.run(skill.execute(SkillRequest(text="news")))
    assert result.data.get("count") == 5


# --- WEATHER1 ----------------------------------------------------------------

def _weather_skill_returning(monkeypatch, *, body=None, raise_json=False):
    import skills.weather as weather_mod
    import httpx

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            if raise_json:
                import json
                raise json.JSONDecodeError("no json", "<html>", 0)
            return body

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return weather_mod


def _weather_config():
    from core.config import Settings
    return Settings().weather


def test_weather_captive_portal_non_json_is_graceful(monkeypatch):
    """A 200 with an HTML login page must degrade, not raise JSONDecodeError."""
    weather_mod = _weather_skill_returning(monkeypatch, raise_json=True)
    skill = weather_mod.WeatherSkill(_weather_config())
    result = asyncio.run(skill.execute(SkillRequest(text="what's the weather")))
    assert not result.success  # spoken offline message, no exception


def test_weather_shape_shift_json_is_graceful(monkeypatch):
    """A 200 whose JSON lacks `current` must degrade, not KeyError."""
    weather_mod = _weather_skill_returning(monkeypatch, body={"error": True})
    skill = weather_mod.WeatherSkill(_weather_config())
    result = asyncio.run(skill.execute(SkillRequest(text="what's the weather")))
    assert not result.success


def test_weather_happy_path_still_works(monkeypatch):
    body = {"current": {"temperature_2m": 21.4, "weather_code": 0},
            "daily": {"temperature_2m_max": [22, 24], "temperature_2m_min": [12, 14],
                      "weather_code": [1, 2]}}
    weather_mod = _weather_skill_returning(monkeypatch, body=body)
    skill = weather_mod.WeatherSkill(_weather_config())
    result = asyncio.run(skill.execute(SkillRequest(text="what's the weather now")))
    assert result.success
    assert result.data.get("temp_c") == 21


# --- STREAM1 — streamed preamble must not suppress a tool answer -------------
#
# When a model narrates a filler sentence before calling a tool, that sentence
# streams via on_delta; the tool's answer does NOT. The old loop keyed off "did
# anything stream?" and dropped the answer. The fix: the router flags whether
# `speech` was the streamed reply, and only THAT is skipped by the loop.

import re as _re

from core.config import load_settings
from core.events import EventBus
from core.router import Router
from llm.client import OllamaClient
from skills.base import Skill, SkillRegistry, SkillResult


class _AnswerSkill(Skill):
    name = "weather_probe"
    description = "returns a direct answer"
    patterns = [_re.compile(r"\bnever matches\b")]

    async def execute(self, request):
        return SkillResult("It's 21 degrees and clear.")


def _stream_router(monkeypatch, fake_chat):
    settings = load_settings()
    registry = SkillRegistry()
    registry.register(_AnswerSkill())
    router = Router(settings, registry, OllamaClient(settings.llm), EventBus())
    router.model = "test-model"
    monkeypatch.setattr(router._llm, "chat", fake_chat)
    return router


@pytest.mark.asyncio
async def test_tool_answer_not_flagged_streamed(monkeypatch):
    """A tool answer preceded by a streamed preamble keeps streamed_reply=False."""
    got = []

    async def fake_chat(model, messages, tools=None, temperature=None, on_delta=None):
        if on_delta is not None:
            on_delta("I'll check the weather now. ")   # the filler preamble
        return {"content": "", "tool_calls": [
            {"function": {"name": "weather_probe", "arguments": {}}}]}

    router = _stream_router(monkeypatch, fake_chat)
    result = await router.route("what's the weather", on_delta=got.append)
    assert result.speech == "It's 21 degrees and clear."
    assert result.streamed_reply is False, "tool answer wrongly marked as streamed"
    assert got, "preamble should have streamed (this is the trap the fix guards)"


@pytest.mark.asyncio
async def test_pure_llm_reply_is_flagged_streamed(monkeypatch):
    """A plain model reply (no tools) that streamed IS flagged, so it isn't re-spoken."""
    async def fake_chat(model, messages, tools=None, temperature=None, on_delta=None):
        if on_delta is not None:
            on_delta("Hello there.")
        return {"content": "Hello there."}

    router = _stream_router(monkeypatch, fake_chat)
    result = await router.route("hi", on_delta=lambda s: None)
    assert result.speech == "Hello there."
    assert result.streamed_reply is True


@pytest.mark.asyncio
async def test_retried_reply_not_flagged_streamed(monkeypatch):
    """A reply that had to be re-fetched off-stream must be spoken, not skipped."""
    calls = {"n": 0}

    async def fake_chat(model, messages, tools=None, temperature=None, on_delta=None):
        calls["n"] += 1
        if calls["n"] == 1:
            # A botched tool call leaked as text -> triggers the off-stream retry.
            return {"content": '{"name": "weather_probe", "arguments": {}}'}
        return {"content": "The clean answer."}

    router = _stream_router(monkeypatch, fake_chat)
    result = await router.route("weather", on_delta=lambda s: None)
    assert result.speech == "The clean answer."
    assert result.streamed_reply is False


def test_loop_already_voiced_logic():
    """The loop speaks result.speech unless (drained AND streamed_reply)."""
    def already_voiced(count, streamed_reply):
        return count > 0 and streamed_reply

    # streamed LLM reply, sentences drained -> already voiced, don't re-speak.
    assert already_voiced(2, True) is True
    # tool answer after a preamble drained -> NOT voiced, must speak.
    assert already_voiced(2, False) is False
    # short reply, nothing drained -> must speak whole.
    assert already_voiced(0, True) is False
    # fast path -> must speak.
    assert already_voiced(0, False) is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
