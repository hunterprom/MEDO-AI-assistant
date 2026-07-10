"""New brains (Claude API, Claude Code CLI, Codex CLI) + the MCP client.

Fully offline: HTTP goes through a fake httpx, CLI agents through a fake
subprocess, and MCP tools through a fake session — same approach as
test_llm_provider.py.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import llm.client as client_mod
from core.config import LLMConfig, MCPConfig, MCPServerConfig
from core.mcp import MCPManager, MCPToolSkill, _result_text, _sanitize
from llm.client import (
    LLMClient,
    LLMUnavailableError,
    _codex_final_message,
    _flatten_for_cli,
    _to_anthropic_messages,
)
from remote.server import VALID_PROVIDERS
from skills.base import SkillRegistry, SkillRequest

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_new_providers_accepted() -> None:
    for provider in ("anthropic", "claude-code", "codex"):
        assert LLMConfig(provider=provider).provider == provider
    assert set(VALID_PROVIDERS) == {"ollama", "openai", "anthropic", "claude-code", "codex"}


def test_unknown_provider_rejected() -> None:
    with pytest.raises(Exception):
        LLMConfig(provider="skynet")


# ---------------------------------------------------------------------------
# anthropic message mapping
# ---------------------------------------------------------------------------


def test_to_anthropic_messages_roundtrip() -> None:
    system, messages = _to_anthropic_messages(
        [
            {"role": "system", "content": "You are MEDO."},
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "weather", "arguments": {"city": "Skopje"}}}],
            },
            {"role": "tool", "name": "weather", "content": "Sunny, 30 degrees."},
            {"role": "assistant", "content": "It's sunny."},
        ]
    )
    assert system == "You are MEDO."
    assert messages[0] == {"role": "user", "content": "weather?"}
    tool_use = messages[1]["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["name"] == "weather"
    assert tool_use["input"] == {"city": "Skopje"}
    result = messages[2]["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == tool_use["id"]  # id matched by order
    assert messages[3] == {"role": "assistant", "content": "It's sunny."}


def test_consecutive_tool_results_share_one_user_turn() -> None:
    _, messages = _to_anthropic_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "a", "arguments": {}}},
                    {"function": {"name": "b", "arguments": {}}},
                ],
            },
            {"role": "tool", "name": "a", "content": "1"},
            {"role": "tool", "name": "b", "content": "2"},
        ]
    )
    assert len(messages) == 2  # one assistant turn + ONE user turn with both results
    assert [b["type"] for b in messages[1]["content"]] == ["tool_result", "tool_result"]


class _FakeAsyncClient:
    """Captures the request; returns a canned Anthropic response."""

    payload: dict[str, Any] = {}
    captured: dict[str, Any] = {}

    def __init__(self, *a: Any, **k: Any) -> None: ...
    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def post(self, url: str, json: Any = None, headers: Any = None) -> Any:
        _FakeAsyncClient.captured = {"url": url, "json": json, "headers": headers}

        class _R:
            def raise_for_status(self) -> None: ...
            def json(self) -> dict[str, Any]:
                return _FakeAsyncClient.payload

        return _R()


class _FakeHTTPError(Exception):
    pass


@pytest.mark.asyncio
async def test_chat_anthropic_maps_tools_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeAsyncClient.payload = {
        "content": [
            {"type": "text", "text": "On it."},
            {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "Skopje"}},
        ]
    }
    fake_httpx = SimpleNamespace(AsyncClient=_FakeAsyncClient, HTTPError=_FakeHTTPError)
    config = LLMConfig(provider="anthropic", anthropic_api_key="sk-ant-test")
    client = LLMClient(config)
    message = await client._chat_anthropic(
        fake_httpx,
        "claude-sonnet-4-5",
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "weather?"}],
        tools=[{"type": "function", "function": {"name": "weather", "description": "d",
                                                 "parameters": {"type": "object", "properties": {}}}}],
        temperature=None,
    )
    sent = _FakeAsyncClient.captured
    assert sent["url"].endswith("/v1/messages")
    assert sent["headers"]["x-api-key"] == "sk-ant-test"
    assert sent["json"]["system"] == "sys"
    assert sent["json"]["tools"][0]["input_schema"] == {"type": "object", "properties": {}}
    assert "max_tokens" in sent["json"]
    assert message["content"] == "On it."
    assert message["tool_calls"][0]["function"]["arguments"] == {"city": "Skopje"}


# ---------------------------------------------------------------------------
# CLI providers
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, out: bytes, err: bytes = b"", code: int = 0) -> None:
        self._out, self._err, self.returncode = out, err, code

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._out, self._err

    def kill(self) -> None: ...


@pytest.mark.asyncio
async def test_chat_cli_claude_code(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProc:
        captured["argv"] = list(argv)
        return _FakeProc(b"Hello there.\n")

    monkeypatch.setattr(client_mod.shutil, "which", lambda cmd: "/usr/local/bin/" + cmd)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    client = LLMClient(LLMConfig(provider="claude-code"))
    message = await client.chat("sonnet", [{"role": "user", "content": "hi"}])
    assert message == {"role": "assistant", "content": "Hello there."}
    assert captured["argv"][0] == "claude"
    assert "-p" in captured["argv"]
    assert ["--model", "sonnet"] == captured["argv"][-2:]


@pytest.mark.asyncio
async def test_chat_cli_default_model_omits_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProc:
        captured["argv"] = list(argv)
        return _FakeProc(b"ok")

    monkeypatch.setattr(client_mod.shutil, "which", lambda cmd: "/bin/" + cmd)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    client = LLMClient(LLMConfig(provider="codex"))
    await client.chat("default", [{"role": "user", "content": "hi"}])
    assert captured["argv"][:3] == ["codex", "exec", "--skip-git-repo-check"]
    assert "-m" not in captured["argv"]


@pytest.mark.asyncio
async def test_chat_cli_missing_binary_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_mod.shutil, "which", lambda cmd: None)
    client = LLMClient(LLMConfig(provider="codex"))
    with pytest.raises(LLMUnavailableError):
        await client.chat("default", [{"role": "user", "content": "hi"}])


def test_cli_availability_and_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_mod.shutil, "which", lambda cmd: "/bin/claude" if cmd == "claude" else None)
    claude = LLMClient(LLMConfig(provider="claude-code"))
    codex = LLMClient(LLMConfig(provider="codex"))
    assert claude.is_available() and "sonnet" in claude.list_models()
    assert not codex.is_available() and codex.list_models() == []


def test_flatten_for_cli_keeps_system_and_history() -> None:
    prompt = _flatten_for_cli(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "1"},
            {"role": "user", "content": "two"},
        ]
    )
    assert prompt.startswith("Be brief.")
    assert "Conversation so far:" in prompt and "User: one" in prompt
    assert prompt.rstrip().endswith("Reply with only your spoken answer, as plain text.")


def test_codex_final_message_strips_logs() -> None:
    out = "[2026-07-10] thinking\ntokens used: 123\nThe answer is 42.\nIt really is."
    assert _codex_final_message(out) == "The answer is 42.\nIt really is."


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------


def test_sanitize_tool_names() -> None:
    assert _sanitize("home assistant/turn.on") == "home_assistant_turn_on"
    assert _sanitize("---") == "tool"


def test_result_text_flattens_and_truncates() -> None:
    blocks = [SimpleNamespace(text="hello", type="text"), SimpleNamespace(text=None, type="image")]
    assert _result_text(SimpleNamespace(content=blocks)) == "hello\n[image]"
    long = SimpleNamespace(content=[SimpleNamespace(text="x" * 9000, type="text")])
    assert _result_text(long).endswith("…(truncated)")


class _FakeManager:
    def __init__(self, reply: str = "done", fail: bool = False) -> None:
        self.reply, self.fail = reply, fail
        self.calls: list[tuple[str, str, dict]] = []

    async def call_tool(self, server: str, tool: str, args: dict) -> str:
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append((server, tool, args))
        return self.reply


@pytest.mark.asyncio
async def test_mcp_tool_skill_schema_and_execute() -> None:
    tool = SimpleNamespace(
        name="play track",
        description="  Plays a\n track ",
        inputSchema={"type": "object", "properties": {"song": {"type": "string"}}},
    )
    manager = _FakeManager(reply="Playing.")
    skill = MCPToolSkill(manager, "spotify", tool)  # type: ignore[arg-type]
    assert skill.name == "spotify_play_track"
    schema = skill.tool_schema()
    assert schema["function"]["parameters"]["properties"]["song"]["type"] == "string"
    assert schema["function"]["description"].startswith("[spotify app]")
    result = await skill.execute(SkillRequest(text="", args={"song": "x"}))
    assert result.success and result.speech == "Playing."
    assert manager.calls == [("spotify", "play track", {"song": "x"})]


@pytest.mark.asyncio
async def test_mcp_tool_skill_failure_is_spoken_not_raised() -> None:
    tool = SimpleNamespace(name="t", description="", inputSchema=None)
    skill = MCPToolSkill(_FakeManager(fail=True), "app", tool)  # type: ignore[arg-type]
    result = await skill.execute(SkillRequest(text=""))
    assert not result.success and "failed" in result.speech


@pytest.mark.asyncio
async def test_mcp_manager_no_servers_is_a_noop() -> None:
    manager = MCPManager(MCPConfig(servers={}))
    registered = await manager.start(SkillRegistry())
    assert registered == 0
    await manager.stop()  # must not raise


def test_mcp_manager_status_states() -> None:
    config = MCPConfig(
        servers={
            "off": MCPServerConfig(enabled=False, command="x"),
            "broken": MCPServerConfig(command="x"),
        }
    )
    manager = MCPManager(config)
    manager._errors["broken"] = "spawn failed"
    states = {s["name"]: s["state"] for s in manager.status()}
    assert states == {"off": "disabled", "broken": "error"}
