"""strip_think: qwen3-style reasoning removal at the chat() choke point."""

from __future__ import annotations

from core.config import LLMConfig
from llm.client import LLMClient, strip_think


def test_plain_text_passes_through():
    assert strip_think("The time is noon.") == "The time is noon."


def test_open_tag_absent_last_close_wins():
    # Ollama templates inject the opening tag, so output starts mid-reasoning.
    assert strip_think("let me think about this</think>Answer.") == "Answer."


def test_multiple_blocks_keep_text_after_last_close():
    text = "<think>a</think>ignored<think>b</think>final"
    assert strip_think(text) == "final"


def test_unterminated_think_tail_dropped():
    assert strip_think("Sure.<think>never closed reasoning") == "Sure."


def test_empty_and_none_are_safe():
    assert strip_think("") == ""
    assert strip_think(None) == ""  # defensive: content may be missing


def test_whitespace_trimmed():
    assert strip_think("reason</think>\n\n  spoken reply  ") == "spoken reply"


async def test_chat_strips_content_but_keeps_tool_calls(monkeypatch):
    client = LLMClient(LLMConfig())

    async def fake_ollama(httpx, model, messages, tools, temperature):
        return {
            "content": "planning the call</think>",
            "tool_calls": [{"function": {"name": "volume", "arguments": {"action": "up"}}}],
        }

    monkeypatch.setattr(client, "_chat_ollama", fake_ollama)
    msg = await client.chat("qwen3:30b", [{"role": "user", "content": "louder"}])
    # Reasoning-only content strips to empty, but the tool call must survive.
    assert msg["content"] == ""
    assert msg["tool_calls"][0]["function"]["name"] == "volume"


async def test_chat_keep_alive_in_ollama_payload(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "ok</think>done"}}

    class _Client:
        def __init__(self, *a, **k): ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            captured.update(json or {})
            return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    client = LLMClient(LLMConfig(keep_alive="30m"))
    msg = await client.chat("m", [{"role": "user", "content": "hi"}])
    assert captured["keep_alive"] == "30m"
    assert msg["content"] == "done"  # strip applied on the real path
