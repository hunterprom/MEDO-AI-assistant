"""M3: tool registry/dispatch and graceful offline degradation of web skills."""

from __future__ import annotations

import httpx
import pytest

from core.config import load_settings
from llm.tools import build_tools, dispatch_tool
from main import Announcer, build_registry
from skills.base import SkillRequest
from skills.news import NewsSkill
from skills.weather import WeatherSkill
from skills.websearch import WebSearchSkill


@pytest.fixture
def registry():
    return build_registry(load_settings(), Announcer())


# --- tool registry / dispatch ------------------------------------------------
def test_build_tools_exposes_web_skills(registry):
    names = {t["function"]["name"] for t in build_tools(registry)}
    assert {"weather", "news", "web_search", "datetime", "power"} <= names
    # every tool has a valid function schema
    for tool in build_tools(registry):
        assert tool["type"] == "function"
        assert "parameters" in tool["function"]


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_is_graceful(registry):
    result = await dispatch_tool(registry, "no_such_tool", {}, {})
    assert result.success is False


# --- offline degradation (network forced to fail) ----------------------------
class _FailingClient:
    """Stand-in httpx.AsyncClient whose requests always fail (offline)."""

    def __init__(self, *a, **k) -> None: ...
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def get(self, *a, **k): raise httpx.ConnectError("offline")


@pytest.mark.asyncio
async def test_weather_offline(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _FailingClient)
    skill = WeatherSkill(load_settings().weather)
    result = await skill.execute(SkillRequest(text="what's the weather"))
    assert result.success is False
    assert "offline" in result.speech.lower()


@pytest.mark.asyncio
async def test_news_offline(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _FailingClient)
    skill = NewsSkill(load_settings().news)
    result = await skill.execute(SkillRequest(text="the news"))
    assert result.success is False
    assert "offline" in result.speech.lower()


@pytest.mark.asyncio
async def test_websearch_offline(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr("ddgs.DDGS", _boom)
    skill = WebSearchSkill(summarize=None)
    result = await skill.execute(SkillRequest(text="search for python", args={"query": "python"}))
    assert result.success is False
    assert "offline" in result.speech.lower()


@pytest.mark.asyncio
async def test_websearch_tool_path_returns_raw_results(monkeypatch):
    # via=tool => return raw results for the model to summarize (no summarizer call).
    class _DDGS:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, q, max_results=5): return [{"title": "T", "body": "B"}]

    monkeypatch.setattr("ddgs.DDGS", _DDGS)
    skill = WebSearchSkill(summarize=None)
    result = await skill.execute(
        SkillRequest(text="search x", args={"query": "x"}, context={"via": "tool"})
    )
    assert result.success and "T" in result.speech and "B" in result.speech
