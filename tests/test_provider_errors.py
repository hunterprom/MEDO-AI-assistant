"""Failure messages must name the REAL problem, and a cold vision model must
not read as a dead one.

Both came from live transcripts: a Claude Code agent that was installed, on
PATH and working still produced "make sure it is installed, logged in, and on
your PATH", and a first look after an idle spell answered "is Ollama running?"
while Ollama was running fine (the 3B vision model was mid-load).
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from core.config import load_settings
from core.events import EventBus
from core.router import OFFLINE_CLI_REPLY, Router
from llm.client import LLMUnavailableError, OllamaClient
from skills.base import SkillRegistry

_PNG = base64.b64encode(base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)).decode()


def _router(provider: str) -> Router:
    settings = load_settings()
    settings.llm.provider = provider
    r = Router(settings, SkillRegistry(), OllamaClient(settings.llm), EventBus())
    r.model = None
    return r


# --- CLI failure messages ----------------------------------------------------

def test_missing_cli_still_says_check_your_path():
    r = _router("claude-code")
    msg = r._failure_reply(LLMUnavailableError(
        "the claude-code CLI ('claude') is not installed or not on PATH"))
    assert msg == OFFLINE_CLI_REPLY.format(name="Claude Code")


def test_working_cli_that_errored_reports_the_real_reason():
    # Installed and on PATH, but the agent itself failed — don't send the user
    # hunting a PATH problem they don't have.
    r = _router("claude-code")
    msg = r._failure_reply(LLMUnavailableError(
        "claude-code CLI exited with 1: usage limit reached for this account"))
    assert "usage limit reached" in msg
    assert "on your PATH" not in msg


def test_cli_timeout_says_it_timed_out():
    r = _router("claude-code")
    msg = r._failure_reply(LLMUnavailableError("claude-code CLI timed out after 180s"))
    assert "too long" in msg.lower()
    assert "on your PATH" not in msg


def test_long_agent_error_is_trimmed_to_something_speakable():
    r = _router("claude-code")
    msg = r._failure_reply(LLMUnavailableError("claude-code CLI exited with 1: " + "x" * 500))
    assert len(msg) < 220


def test_non_cli_provider_falls_back_to_the_generic_message():
    r = _router("ollama")
    msg = r._failure_reply(LLMUnavailableError("connection refused"))
    assert msg == r._offline_reply


# --- vision cold-start retry -------------------------------------------------

def test_vision_retries_once_when_the_model_is_still_loading(monkeypatch):
    import httpx

    import skills.vision_skill as vs

    calls = {"n": 0}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "a cat"}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            calls["n"] += 1
            if calls["n"] == 1:                     # cold model refuses the first hit
                raise httpx.ConnectError("connection refused")
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    # capture the REAL sleep first: patching with a lambda that calls
    # asyncio.sleep would call the lambda (infinite recursion).
    _real_sleep = asyncio.sleep
    monkeypatch.setattr(vs.asyncio, "sleep", lambda *_a, **_k: _real_sleep(0))

    result = asyncio.run(vs._describe(load_settings(), _PNG, "what is this"))
    assert result.success is True and "cat" in result.speech
    assert calls["n"] == 2, "should have retried exactly once"


def test_vision_gives_up_after_the_retry(monkeypatch):
    import httpx

    import skills.vision_skill as vs

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            raise httpx.ConnectError("still down")

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    # capture the REAL sleep first: patching with a lambda that calls
    # asyncio.sleep would call the lambda (infinite recursion).
    _real_sleep = asyncio.sleep
    monkeypatch.setattr(vs.asyncio, "sleep", lambda *_a, **_k: _real_sleep(0))

    result = asyncio.run(vs._describe(load_settings(), _PNG, "what is this"))
    assert result.success is False
    assert "Ollama" in result.speech


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
