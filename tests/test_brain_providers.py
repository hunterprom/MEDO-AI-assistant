"""S1 — the BrainProvider abstraction (interface + Ollama/OpenAI-compat adapters).

The providers delegate to the existing LLMClient, so these tests mostly pin the
CONTRACT (shape, local/cloud flags, availability, error surfacing) — the wire
behaviour is already covered by test_llm_provider.py. httpx is monkeypatched;
fully offline.
"""

from __future__ import annotations


import pytest

from core.config import load_settings
from llm.client import LLMUnavailableError
from llm.providers import (
    BrainProvider,
    KNOWN_OPENAI_COMPAT,
    OllamaProvider,
    OpenAICompatProvider,
)


# -- httpx fakes (client imports httpx lazily) --------------------------------

class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload, self.status_code = payload, status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("err", request=None, response=self)

    def json(self):
        return self._payload


def _mock_post(monkeypatch, *, payload=None, status=200, raises=None):
    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return False
        async def post(self, url, json=None, headers=None):
            if raises is not None:
                raise raises
            return FakeResponse(payload, status)
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)


def _openai_payload(content):
    return {"choices": [{"index": 0, "message": {
        "role": "assistant", "content": content}, "finish_reason": "stop"}]}


def cfg():
    return load_settings().llm


# -- interface conformance -----------------------------------------------------

def test_providers_conform_to_the_interface():
    local = OllamaProvider("qwen-local", "qwen3:30b", cfg())
    cloud = OpenAICompatProvider("deepseek", "deepseek-v4-flash",
                                 "https://api.deepseek.com", "sk-x", cfg())
    for p in (local, cloud):
        assert isinstance(p, BrainProvider)
        assert isinstance(p.name, str) and isinstance(p.model_id, str)
        assert isinstance(p.is_local, bool)
        assert hasattr(p, "generate") and hasattr(p, "available")


def test_local_and_cloud_flags():
    local = OllamaProvider("qwen-local", "qwen3:30b", cfg())
    cloud = OpenAICompatProvider("gemini", "gemini-2.5-flash",
                                 "https://x/", "key", cfg())
    assert local.is_local and not local.is_cloud and local.location == "local"
    assert cloud.is_cloud and not cloud.is_local and cloud.location == "cloud"


# -- OllamaProvider: unchanged local behaviour --------------------------------

@pytest.mark.asyncio
async def test_ollama_generate_delegates_and_returns_message(monkeypatch):
    _mock_post(monkeypatch, payload={
        "message": {"role": "assistant", "content": "It's 4."}})
    p = OllamaProvider("qwen-local", "qwen3:30b", cfg())
    msg = await p.generate([{"role": "user", "content": "2+2?"}])
    assert msg["content"] == "It's 4."


@pytest.mark.asyncio
async def test_ollama_strips_think_tags_like_before(monkeypatch):
    _mock_post(monkeypatch, payload={"message": {
        "role": "assistant", "content": "<think>hmm</think>Four."}})
    p = OllamaProvider("qwen-local", "qwen3:30b", cfg())
    msg = await p.generate([{"role": "user", "content": "2+2?"}])
    assert msg["content"] == "Four."          # single choke-point preserved


def test_ollama_available_checks_the_model_is_pulled(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse(
        {"models": [{"name": "qwen3:30b"}, {"name": "llama3.2:3b"}]}))
    assert OllamaProvider("qwen-local", "qwen3:30b", cfg()).available() is True
    assert OllamaProvider("missing", "nope:1b", cfg()).available() is False


def test_ollama_available_is_false_when_ollama_down(monkeypatch):
    import httpx
    def boom(*a, **k):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(httpx, "get", boom)
    assert OllamaProvider("qwen-local", "qwen3:30b", cfg()).available() is False


# -- OpenAICompatProvider: mocked endpoint (success/timeout/401/rate-limit) ----

@pytest.mark.asyncio
async def test_cloud_generate_success(monkeypatch):
    _mock_post(monkeypatch, payload=_openai_payload("Hello from the cloud."))
    p = OpenAICompatProvider("deepseek", "deepseek-v4-flash",
                             "https://api.deepseek.com", "sk-x", cfg())
    msg = await p.generate([{"role": "user", "content": "hi"}])
    assert msg["content"] == "Hello from the cloud."


@pytest.mark.asyncio
async def test_cloud_timeout_surfaces_as_unavailable(monkeypatch):
    import httpx
    _mock_post(monkeypatch, raises=httpx.ReadTimeout("timed out"))
    p = OpenAICompatProvider("kimi", "kimi-k2.6", "https://api.moonshot.ai/v1",
                             "sk-x", cfg())
    with pytest.raises(LLMUnavailableError):    # caller falls back to local (S4)
        await p.generate([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_cloud_401_surfaces_as_unavailable(monkeypatch):
    _mock_post(monkeypatch, payload={"error": "unauthorized"}, status=401)
    p = OpenAICompatProvider("glm", "glm-4.5", "https://open.bigmodel.cn/api/paas/v4/",
                             "bad-key", cfg())
    with pytest.raises(LLMUnavailableError):
        await p.generate([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_cloud_rate_limit_surfaces_as_unavailable(monkeypatch):
    _mock_post(monkeypatch, payload={"error": "rate limited"}, status=429)
    p = OpenAICompatProvider("gemini", "gemini-2.5-flash", "https://x/", "key", cfg())
    with pytest.raises(LLMUnavailableError):
        await p.generate([{"role": "user", "content": "hi"}])


# -- availability: cloud needs a key ------------------------------------------

def test_cloud_without_a_key_is_unavailable():
    assert OpenAICompatProvider("gemini", "gemini-2.5-flash", "https://x/", "",
                                cfg()).available() is False
    assert OpenAICompatProvider("gemini", "gemini-2.5-flash", "https://x/", "k",
                                cfg()).available() is True


# -- the verified reference endpoints -----------------------------------------

def test_known_endpoints_are_present_and_shaped():
    for key in ("openai", "deepseek", "gemini", "kimi", "glm"):
        ep = KNOWN_OPENAI_COMPAT[key]
        assert ep.base_url.startswith("https://") and ep.example_models
    # the gotchas worth encoding are recorded
    assert KNOWN_OPENAI_COMPAT["gemini"].base_url.endswith("/")   # trailing slash
    assert "deprecated" in KNOWN_OPENAI_COMPAT["deepseek"].note
