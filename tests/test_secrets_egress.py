"""Secrets redaction + cloud data-egress gate (S5).

Secrets never reach a log unscrubbed; local RAG/memory never rides along to a
cloud brain without a per-brain opt-in (verified through the real router: the
cloud tool list drops search_documents and local facts are withheld).
"""

from __future__ import annotations

import pytest

from core.config import load_settings
from core.events import EventBus
from core.router import Router
from security.egress import (
    LOCAL_CONTENT_TOOLS,
    is_cloud,
    local_context_allowed,
)
from security.secrets import Secrets
from skills.base import Skill, SkillRegistry, SkillResult


# -- redaction ----------------------------------------------------------------

def test_redact_scrubs_live_secret_values():
    s = load_settings()
    s.remote.token = "supersecrettoken123"
    red = Secrets(s).redact("connecting with token supersecrettoken123 now")
    assert "supersecrettoken123" not in red and "***" in red


def test_redact_ignores_trivial_values():
    s = load_settings()
    s.remote.token = "ab"                       # too short to scrub safely
    assert Secrets(s).redact("value ab here") == "value ab here"


def test_secrets_get_and_has():
    s = load_settings()
    s.remote.token = "tok_abcdef"
    sec = Secrets(s)
    assert sec.has("remote.token") and sec.get("remote.token") == "tok_abcdef"
    assert sec.get("nonexistent.key") == "" and sec.has("nonexistent.key") is False


# -- egress decision ----------------------------------------------------------

def test_local_provider_always_allowed():
    s = load_settings()
    assert is_cloud("ollama") is False
    assert local_context_allowed(s.security, "ollama") is True


def test_cloud_blocked_without_optin_allowed_with():
    s = load_settings()
    assert is_cloud("anthropic") is True
    assert local_context_allowed(s.security, "anthropic") is False
    s.security.cloud_egress_optin = {"anthropic": True}
    assert local_context_allowed(s.security, "anthropic") is True


# -- through the real router --------------------------------------------------

class _DocSkill(Skill):
    name = "search_documents"
    description = "search local documents"
    patterns = []

    async def execute(self, request):
        return SkillResult("local doc text")

    def tool_schema(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": {}, "required": []}}}


class _FakeLLM:
    def __init__(self):
        self.captured = {}

    async def chat(self, model, messages, tools=None, **kw):
        self.captured = {"messages": messages,
                         "tool_names": [t["function"]["name"] for t in (tools or [])]}
        return {"content": "hi", "tool_calls": []}


def _router(provider, optin=None):
    reg = SkillRegistry()
    reg.register(_DocSkill())
    s = load_settings()
    s.llm.provider = provider
    if optin:
        s.security.cloud_egress_optin = optin
    llm = _FakeLLM()
    r = Router(s, reg, llm, EventBus())
    r.model = "test-model"
    r.facts.relevant = lambda text, n: ["the user likes strong coffee"]
    return r, llm


@pytest.mark.asyncio
async def test_cloud_brain_gets_no_local_docs_or_memory_without_optin():
    r, llm = _router("anthropic")
    await r._llm_reply("what do my notes say", {})
    assert "search_documents" not in llm.captured["tool_names"]     # RAG withheld
    system = llm.captured["messages"][0]["content"]
    assert "strong coffee" not in system                           # memory withheld


@pytest.mark.asyncio
async def test_cloud_brain_gets_local_context_with_optin():
    r, llm = _router("anthropic", optin={"anthropic": True})
    await r._llm_reply("what do my notes say", {})
    assert "search_documents" in llm.captured["tool_names"]
    system = llm.captured["messages"][0]["content"]
    assert "strong coffee" in system


@pytest.mark.asyncio
async def test_local_brain_keeps_local_context():
    r, llm = _router("ollama")
    await r._llm_reply("what do my notes say", {})
    assert "search_documents" in llm.captured["tool_names"]
    assert "strong coffee" in llm.captured["messages"][0]["content"]


def test_local_content_tools_named():
    assert "search_documents" in LOCAL_CONTENT_TOOLS


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
