"""Multi-provider LLM tests: OpenAI normalization, live provider switching, and
the companion API's model/provider endpoints. Fully offline — httpx is
monkeypatched and the endpoints run against a fake llm + router."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from core.config import LLMConfig, Settings, load_settings
from core.events import AssistantState, EventBus, StateMachine
from llm.client import LLMClient, OllamaClient
from remote.server import RemoteServer

# ---------------------------------------------------------------------------
# httpx fakes (the client imports httpx lazily, so patching the module works)
# ---------------------------------------------------------------------------


class FakeResponse:
    """Just enough of httpx.Response for the client code paths."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"unexpected HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


def fake_httpx_post(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> dict[str, Any]:
    """Replace httpx.AsyncClient with a capture-only fake; returns the capture dict."""
    captured: dict[str, Any] = {}

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

        async def post(
            self, url: str, json: Any = None, headers: dict[str, str] | None = None
        ) -> FakeResponse:
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers or {}
            return FakeResponse(payload)

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return captured


def openai_chat_payload(message: dict[str, Any]) -> dict[str, Any]:
    return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}


# ---------------------------------------------------------------------------
# tool_calls normalization (OpenAI arguments arrive as a JSON string)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_tool_call_arguments_parsed_to_dict(monkeypatch: pytest.MonkeyPatch):
    payload = openai_chat_payload(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "Skopje"}'},
                }
            ],
        }
    )
    fake_httpx_post(monkeypatch, payload)
    client = LLMClient(LLMConfig(provider="openai", api_key="sk-test"))

    message = await client.chat("gpt-4o-mini", [{"role": "user", "content": "weather?"}])

    call = message["tool_calls"][0]["function"]
    assert call["name"] == "weather"
    assert call["arguments"] == {"city": "Skopje"}  # dict, not the JSON string
    assert message["content"] == ""  # null content normalized to str


@pytest.mark.asyncio
async def test_openai_unparseable_arguments_become_empty_dict(monkeypatch: pytest.MonkeyPatch):
    payload = openai_chat_payload(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {"name": "notes", "arguments": "{not valid json"},
                }
            ],
        }
    )
    fake_httpx_post(monkeypatch, payload)
    client = LLMClient(LLMConfig(provider="openai", api_key="sk-test"))

    message = await client.chat("gpt-4o-mini", [{"role": "user", "content": "note it"}])

    assert message["tool_calls"][0]["function"]["arguments"] == {}


# ---------------------------------------------------------------------------
# provider switching (read at call time from the shared config)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_switch_changes_target_and_headers(monkeypatch: pytest.MonkeyPatch):
    config = LLMConfig()  # provider defaults to ollama
    client = LLMClient(config)
    messages = [{"role": "user", "content": "hi"}]

    captured = fake_httpx_post(
        monkeypatch, {"message": {"role": "assistant", "content": "hello from ollama"}}
    )
    message = await client.chat("llama3.2:3b", messages)
    assert captured["url"].endswith("/api/chat")
    assert "Authorization" not in captured["headers"]
    assert message["content"] == "hello from ollama"

    # Mutate the SAME config object — the client must pick it up live.
    config.provider = "openai"
    config.api_key = "sk-live-test"
    captured = fake_httpx_post(
        monkeypatch,
        openai_chat_payload({"role": "assistant", "content": "hello from openai"}),
    )
    message = await client.chat("gpt-4o-mini", messages)
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-live-test"
    assert message["content"] == "hello from openai"


@pytest.mark.asyncio
async def test_openai_400_tools_rejection_retries_without_tools(monkeypatch: pytest.MonkeyPatch):
    """Models that reject tool schemas (Groq allam-2-7b) must still answer."""
    calls: list[dict[str, Any]] = []

    class _Client:
        def __init__(self, *a: Any, **k: Any) -> None: ...
        async def __aenter__(self): return self
        async def __aexit__(self, *exc: Any): return False

        async def post(self, url: str, json: Any = None, headers: Any = None) -> FakeResponse:
            calls.append(dict(json))  # snapshot: the client mutates the payload on retry
            if "tools" in json:
                return FakeResponse(
                    {"error": {"message": "`tool calling` is not supported with this model"}},
                    status_code=400,
                )
            return FakeResponse(openai_chat_payload({"role": "assistant", "content": "4"}))

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    client = LLMClient(LLMConfig(provider="openai", api_key="sk-test"))
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]

    message = await client.chat("allam-2-7b", [{"role": "user", "content": "2+2?"}], tools=tools)

    assert message["content"] == "4"
    assert len(calls) == 2 and "tools" in calls[0] and "tools" not in calls[1]


def test_is_available_openai_means_key_present():
    assert LLMClient(LLMConfig(provider="openai", api_key="sk-x")).is_available() is True
    assert LLMClient(LLMConfig(provider="openai", api_key="")).is_available() is False


def test_list_models_openai_sorted_ids(monkeypatch: pytest.MonkeyPatch):
    import httpx

    def fake_get(url: str, headers: dict[str, str] | None = None, timeout: float | None = None):
        assert url == "https://api.openai.com/v1/models"
        assert headers is not None and headers["Authorization"] == "Bearer sk-x"
        return FakeResponse({"data": [{"id": "gpt-b"}, {"id": "gpt-a"}]})

    monkeypatch.setattr(httpx, "get", fake_get)
    client = LLMClient(LLMConfig(provider="openai", api_key="sk-x"))
    assert client.list_models() == ["gpt-a", "gpt-b"]


def test_list_models_returns_empty_on_error(monkeypatch: pytest.MonkeyPatch):
    import httpx

    def boom(*args: Any, **kwargs: Any):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(httpx, "get", boom)
    client = LLMClient(LLMConfig(provider="openai", api_key="sk-x"))
    assert client.list_models() == []


def test_ollama_client_alias_is_llm_client():
    assert OllamaClient is LLMClient


# ---------------------------------------------------------------------------
# companion API endpoints (fake llm + router; no network, no Ollama)
# ---------------------------------------------------------------------------


class FakeLLM:
    """Stands in for LLMClient: fixed model list, always available."""

    def __init__(self, models: list[str]) -> None:
        self.models = models

    def is_available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return list(self.models)


class FakeRouter:
    """The slice of Router the endpoints touch: ``model`` and ``llm``."""

    def __init__(self, llm: FakeLLM) -> None:
        self._llm = llm
        self.model: str | None = "alpha"

    @property
    def llm(self) -> FakeLLM:
        return self._llm


@pytest_asyncio.fixture
async def api() -> AsyncIterator[tuple[TestClient, Settings, FakeRouter]]:
    settings = load_settings()
    router = FakeRouter(FakeLLM(["alpha", "beta"]))
    server = RemoteServer(settings, router, StateMachine(EventBus()))  # type: ignore[arg-type]

    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    yield client, settings, router
    await client.close()


@pytest.mark.asyncio
async def test_status_reports_provider_model_and_state(api):
    client, settings, router = api
    resp = await client.get("/status")
    assert resp.status == 200
    body = await resp.json()
    assert body["ok"] is True
    assert body["name"] == settings.personality.name
    assert body["provider"] == settings.llm.provider
    assert body["model"] == router.model
    assert body["models"] == ["alpha", "beta"]
    assert body["state"] == AssistantState.IDLE.value


@pytest.mark.asyncio
async def test_models_endpoint(api):
    client, _, _ = api
    resp = await client.get("/models")
    assert resp.status == 200
    assert (await resp.json())["models"] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_set_model(api):
    client, _, router = api
    resp = await client.post("/model", json={"name": "beta"})
    assert resp.status == 200
    body = await resp.json()
    assert body == {"ok": True, "model": "beta"}
    assert router.model == "beta"


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"name": ""}, {"name": "   "}])
async def test_set_model_rejects_missing_or_empty_name(api, payload):
    client, _, router = api
    resp = await client.post("/model", json=payload)
    assert resp.status == 400
    assert router.model == "alpha"  # unchanged


@pytest.mark.asyncio
async def test_provider_switch_mutates_settings_and_never_leaks_key(api):
    client, settings, router = api
    resp = await client.post(
        "/provider",
        json={"provider": "openai", "api_key": "sk-secret-123", "model": "gpt-4o-mini"},
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["ok"] is True
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o-mini"
    assert body["models"] == ["alpha", "beta"]
    # The key is stored but never echoed back — not even under another name.
    assert "sk-secret-123" not in await resp.text()
    assert "api_key" not in body
    assert settings.llm.provider == "openai"
    assert settings.llm.api_key == "sk-secret-123"
    assert router.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_provider_switch_defaults_model_to_first_listed(api):
    client, settings, router = api
    router.model = None
    resp = await client.post("/provider", json={"provider": "ollama"})
    assert resp.status == 200
    body = await resp.json()
    assert body["model"] == "alpha"
    assert router.model == "alpha"
    assert settings.llm.provider == "ollama"


@pytest.mark.asyncio
async def test_provider_switch_sets_base_url(api):
    client, settings, _ = api
    resp = await client.post(
        "/provider",
        json={"provider": "openai", "api_key": "sk-x", "base_url": "https://api.groq.com/openai/v1"},
    )
    assert resp.status == 200
    assert settings.llm.openai_base_url == "https://api.groq.com/openai/v1"


@pytest.mark.asyncio
async def test_provider_rejects_unknown_provider(api):
    client, settings, _ = api
    before = settings.llm.provider
    resp = await client.post("/provider", json={"provider": "skynet"})
    assert resp.status == 400
    assert settings.llm.provider == before


@pytest.mark.asyncio
async def test_cors_headers_on_ping_and_errors(api):
    client, _, _ = api
    resp = await client.get("/ping")
    assert resp.headers["Access-Control-Allow-Origin"] == "*"

    resp = await client.post("/model", json={})  # 400 must still carry CORS
    assert resp.status == 400
    assert resp.headers["Access-Control-Allow-Origin"] == "*"


@pytest.mark.asyncio
async def test_options_preflight_returns_204(api):
    client, _, _ = api
    resp = await client.options("/provider")
    assert resp.status == 204
    assert resp.headers["Access-Control-Allow-Origin"] == "*"
    assert "POST" in resp.headers["Access-Control-Allow-Methods"]
