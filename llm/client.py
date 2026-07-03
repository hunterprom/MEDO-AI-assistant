"""Provider-agnostic async LLM client.

Talks to either a local Ollama server or any online OpenAI-compatible API
(chat completions + model listing). The active provider is read from the shared
:class:`~core.config.LLMConfig` *at call time*, so mutating that config (e.g.
via the companion API's ``POST /provider`` endpoint) switches providers live —
no reconstruction needed.

``httpx`` is imported lazily so the fast path — and the whole app — still runs
on a machine where the LLM deps or the servers aren't present. The API key is
only ever placed in request headers; it is never logged.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when the configured LLM provider can't be reached."""


def _normalize_openai_message(message: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI assistant message to the internal (Ollama-style) shape.

    OpenAI encodes ``function.arguments`` as a JSON *string*; internally the
    router expects a dict. Unparseable or non-object arguments become ``{}``.
    Tool-call ids are preserved so multi-round tool calling can echo them back.
    """
    normalized: dict[str, Any] = {
        "role": message.get("role", "assistant"),
        "content": message.get("content") or "",
    }
    tool_calls: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
            except ValueError:
                parsed = {}
        elif isinstance(args, dict):
            parsed = args
        else:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        entry: dict[str, Any] = {
            "function": {"name": fn.get("name", ""), "arguments": parsed}
        }
        if call.get("id"):
            entry["id"] = call["id"]
        tool_calls.append(entry)
    if tool_calls:
        normalized["tool_calls"] = tool_calls
    return normalized


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-encode internal chat history for the OpenAI wire format.

    Internally, assistant ``tool_calls`` carry dict arguments and tool-result
    messages have no ``tool_call_id``. OpenAI-compatible servers require string
    arguments, an ``id``/``type`` on each call, and a matching ``tool_call_id``
    on every tool result — this restores all three without mutating the input.
    """
    out: list[dict[str, Any]] = []
    id_queue: list[str] = []  # pending call ids, consumed by tool results in order
    for index, original in enumerate(messages):
        msg = dict(original)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            fixed: list[dict[str, Any]] = []
            for i, call in enumerate(msg["tool_calls"]):
                fn = dict(call.get("function") or {})
                if isinstance(fn.get("arguments"), dict):
                    fn["arguments"] = json.dumps(fn["arguments"])
                call_id = str(call.get("id") or f"call_{index}_{i}")
                fixed.append({"id": call_id, "type": "function", "function": fn})
                id_queue.append(call_id)
            msg["tool_calls"] = fixed
        elif msg.get("role") == "tool" and "tool_call_id" not in msg and id_queue:
            msg["tool_call_id"] = id_queue.pop(0)
        out.append(msg)
    return out


class LLMClient:
    """Chat + model discovery over Ollama or an OpenAI-compatible HTTP API."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    # -- provider plumbing ---------------------------------------------------

    @property
    def _provider(self) -> str:
        """The active provider, read fresh on every call (live switching)."""
        return self._config.provider

    def _ollama_host(self) -> str:
        return self._config.host.rstrip("/")

    def _openai_base(self) -> str:
        return self._config.openai_base_url.rstrip("/")

    def _openai_headers(self) -> dict[str, str]:
        # Carries the API key — never log this dict.
        return {"Authorization": f"Bearer {self._config.api_key}"}

    # -- availability / model discovery -------------------------------------

    def is_available(self) -> bool:
        """True if the active provider looks usable. Never raises."""
        if self._provider == "openai":
            # No probe request: a key present means we can at least try.
            return bool(self._config.api_key)
        try:
            import httpx

            resp = httpx.get(f"{self._ollama_host()}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception as exc:  # server down, httpx missing, DNS, etc.
            logger.debug("ollama not available: %s", exc)
            return False

    def list_models(self) -> list[str]:
        """Sorted model names offered by the active provider (``[]`` on error)."""
        try:
            import httpx

            if self._provider == "openai":
                resp = httpx.get(
                    f"{self._openai_base()}/models",
                    headers=self._openai_headers(),
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()
                return sorted(str(m["id"]) for m in data.get("data", []))
            resp = httpx.get(f"{self._ollama_host()}/api/tags", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            return sorted(m["name"] for m in data.get("models", []))
        except Exception as exc:
            logger.debug("could not list models: %s", exc)
            return []

    # -- inference ----------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion and return the assistant *message* dict.

        The message has ``content`` and, when the model decides to call tools,
        a ``tool_calls`` list whose ``function.arguments`` is always a dict —
        whichever provider is active. Raises :class:`LLMUnavailableError` if the
        provider can't be reached so the caller can degrade gracefully.
        """
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - deps not installed
            raise LLMUnavailableError("httpx is not installed") from exc

        if self._provider == "openai":
            return await self._chat_openai(httpx, model, messages, tools, temperature)
        return await self._chat_ollama(httpx, model, messages, tools, temperature)

    async def _chat_ollama(
        self,
        httpx: Any,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        """Chat against a local Ollama server (``POST /api/chat``)."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": (
                    self._config.temperature if temperature is None else temperature
                ),
                "num_ctx": self._config.num_ctx,
            },
        }
        if tools:
            payload["tools"] = tools
        try:
            async with httpx.AsyncClient(timeout=self._config.request_timeout_s) as client:
                resp = await client.post(f"{self._ollama_host()}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(str(exc)) from exc

        return data.get("message", {}) or {}

    async def _chat_openai(
        self,
        httpx: Any,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        """Chat against an OpenAI-compatible API (``POST /chat/completions``)."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": _to_openai_messages(messages),
            "temperature": (
                self._config.temperature if temperature is None else temperature
            ),
        }
        if tools:
            payload["tools"] = tools
        try:
            async with httpx.AsyncClient(timeout=self._config.request_timeout_s) as client:
                resp = await client.post(
                    f"{self._openai_base()}/chat/completions",
                    json=payload,
                    headers=self._openai_headers(),
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            # str(exc) contains the URL at most — never the Authorization header.
            raise LLMUnavailableError(str(exc)) from exc

        choices = data.get("choices") or []
        message = (choices[0].get("message") if choices else None) or {}
        return _normalize_openai_message(message)


#: Backwards-compatible alias — the router, main.py, and tests import this name.
OllamaClient = LLMClient
