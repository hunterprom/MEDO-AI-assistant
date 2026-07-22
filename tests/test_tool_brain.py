"""Auto tool-brain: CLI agents borrow a local tool-capable model for live info."""

from __future__ import annotations

from core.config import load_settings
from core.events import EventBus
from core.router import _LIVE_INFO_RE, Router
from llm.client import OllamaClient
from skills.base import SkillRegistry


def _router(provider: str, tool_brain: str = "qwen3:30b") -> Router:
    s = load_settings()
    s.llm.provider = provider
    s.llm.tool_brain_model = tool_brain
    r = Router(s, SkillRegistry(), OllamaClient(s.llm), EventBus())
    r.model = "sonnet"
    # pretend Ollama is up so we exercise the borrow decision, not the probe
    tb = r._tool_brain_client()
    if tb is not None:
        tb.is_available = lambda: True  # type: ignore[method-assign]
    return r


def test_live_info_regex_matches_web_queries():
    for q in ["what's the latest on the election", "who won the game last night",
              "search for python tutorials", "what's happening today",
              "look up the stock price", "најнови вести"]:
        assert _LIVE_INFO_RE.search(q), q
    for q in ["tell me a joke", "how are you", "write a poem about cats"]:
        assert not _LIVE_INFO_RE.search(q), q


def test_cli_brain_borrows_tool_brain_for_live_info():
    r = _router("claude-code")
    client, model, borrowed = r._pick_brain("what's the latest news on AI")
    assert borrowed and model == "qwen3:30b" and client is r._tool_brain


def test_cli_brain_keeps_selected_model_for_normal_chat():
    r = _router("claude-code")
    client, model, borrowed = r._pick_brain("tell me a joke")
    assert not borrowed and model == "sonnet" and client is r._llm


def test_tool_capable_brain_never_borrows():
    # ollama/openai/anthropic already get MEDO's tools — no need to switch.
    r = _router("ollama")
    _, _, borrowed = r._pick_brain("what's the latest news on AI")
    assert not borrowed


def test_disabled_when_no_tool_brain_configured():
    r = _router("claude-code", tool_brain="")
    _, model, borrowed = r._pick_brain("what's the latest news")
    assert not borrowed and model == "sonnet"


def test_borrow_falls_back_when_ollama_down():
    r = _router("claude-code")
    r._tool_brain.is_available = lambda: False  # type: ignore[method-assign]
    _, model, borrowed = r._pick_brain("what's the latest news")
    assert not borrowed and model == "sonnet"  # gracefully use the selected brain
