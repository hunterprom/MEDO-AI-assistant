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
import re
from typing import Any

from core.config import LLMConfig

logger = logging.getLogger(__name__)

#: A complete <think>…</think> block anywhere in a reply.
_THINK_PAIR = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove a thinking model's chain-of-thought from a reply.

    qwen3-style models emit reasoning that ends in ``</think>`` — the opening
    tag is usually template-injected, so it never appears in the output. Keep
    only what follows the LAST ``</think>``, then drop any stray complete
    pairs and an unterminated trailing ``<think>`` block (mid-reasoning
    truncation). The result is safe for TTS, conversation memory, and the
    yes/no confirmation gate.
    """
    t = text or ""
    close = t.rfind("</think>")
    if close != -1:
        t = t[close + len("</think>"):]
    t = _THINK_PAIR.sub("", t)
    open_idx = t.find("<think>")
    if open_idx != -1:
        t = t[:open_idx]
    return t.strip()


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

    async def warmup(self, model: str | None) -> None:
        """Preload a local model so the first real question doesn't stall.

        A cold qwen3:30b takes ~27 s just to load on this GPU — users read that
        as "no answer". Fired as a background task at startup and after a
        provider/model switch; a 1-token generation forces the load and
        ``keep_alive`` then keeps it resident. No-op for online providers,
        never raises.
        """
        if self._provider != "ollama" or not model:
            return
        try:
            import httpx

            async with httpx.AsyncClient(timeout=300.0) as client:
                await client.post(
                    f"{self._ollama_host()}/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                        "keep_alive": self._config.keep_alive,
                        "options": {"num_ctx": self._config.num_ctx, "num_predict": 1},
                    },
                )
            logger.info("warmed up local model %s", model)
        except Exception:
            logger.debug("model warmup failed (non-fatal)", exc_info=True)

    # -- inference ----------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        on_delta: Any = None,
    ) -> dict[str, Any]:
        """Send a chat completion and return the assistant *message* dict.

        The message has ``content`` and, when the model decides to call tools,
        a ``tool_calls`` list whose ``function.arguments`` is always a dict —
        whichever provider is active. Raises :class:`LLMUnavailableError` if the
        provider can't be reached so the caller can degrade gracefully.

        ``on_delta`` (a ``Callable[[str], None]``) switches to the streaming
        wire protocol: content chunks are pushed to it as they generate — the
        voice loop speaks completed sentences while the rest is still being
        written — and the SAME final message dict is returned. Tool-call
        rounds produce no content deltas, so streaming is safe on every round.
        """
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - deps not installed
            raise LLMUnavailableError("httpx is not installed") from exc

        if self._provider == "openai":
            if on_delta is not None:
                message = await self._chat_openai_stream(httpx, model, messages,
                                                         tools, temperature, on_delta)
            else:
                message = await self._chat_openai(httpx, model, messages, tools, temperature)
        else:
            if on_delta is not None:
                message = await self._chat_ollama_stream(httpx, model, messages,
                                                         tools, temperature, on_delta)
            else:
                message = await self._chat_ollama(httpx, model, messages, tools, temperature)
        # One choke point for reasoning removal: everything downstream (router,
        # conversation memory, HUD, TTS, confirmation gate) sees clean text.
        if isinstance(message.get("content"), str):
            message["content"] = strip_think(message["content"])
        return message

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
            # Keep the model warm between turns; the default 5m unload costs a
            # 15–30s reload for a partially-offloaded 30B on this GPU.
            "keep_alive": self._config.keep_alive,
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

    async def _chat_ollama_stream(
        self,
        httpx: Any,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        on_delta: Any,
    ) -> dict[str, Any]:
        """Streaming Ollama chat: push content chunks, return the final message.

        Newer Ollama emits qwen3-style reasoning in a separate ``thinking``
        field, so content deltas are clean; a ``<think`` guard stops emission
        anyway if an older template leaks tags inline.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": self._config.keep_alive,
            "options": {
                "temperature": (
                    self._config.temperature if temperature is None else temperature
                ),
                "num_ctx": self._config.num_ctx,
            },
        }
        if tools:
            payload["tools"] = tools
        content = ""
        tool_calls: list[dict[str, Any]] = []
        emit_ok = True
        try:
            async with httpx.AsyncClient(timeout=self._config.request_timeout_s) as client:
                async with client.stream(
                    "POST", f"{self._ollama_host()}/api/chat", json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        msg = data.get("message") or {}
                        delta = msg.get("content") or ""
                        if delta:
                            content += delta
                            if "<think" in content:
                                emit_ok = False  # inline reasoning: stop speaking it
                            if emit_ok:
                                on_delta(delta)
                        tool_calls.extend(msg.get("tool_calls") or [])
                        if data.get("done"):
                            break
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(str(exc)) from exc
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    async def _chat_openai_stream(
        self,
        httpx: Any,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        on_delta: Any,
    ) -> dict[str, Any]:
        """Streaming OpenAI-compatible chat (SSE); returns the final message.

        Tool-call deltas are merged by index (OpenAI fragments name/arguments
        across chunks). On any non-2xx (e.g. Groq's 400 for models that reject
        tools) it falls back to the non-streaming path, which owns that retry.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": _to_openai_messages(messages),
            "temperature": (
                self._config.temperature if temperature is None else temperature
            ),
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        content = ""
        calls: dict[int, dict[str, Any]] = {}
        emit_ok = True
        try:
            async with httpx.AsyncClient(timeout=self._config.request_timeout_s) as client:
                async with client.stream(
                    "POST",
                    f"{self._openai_base()}/chat/completions",
                    json=payload,
                    headers=self._openai_headers(),
                ) as resp:
                    if resp.status_code >= 400:
                        await resp.aread()  # drain; the non-stream path handles retries
                        return await self._chat_openai(httpx, model, messages,
                                                       tools, temperature)
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if not chunk or chunk == "[DONE]":
                            continue
                        choices = json.loads(chunk).get("choices") or []
                        delta = (choices[0].get("delta") if choices else None) or {}
                        piece = delta.get("content") or ""
                        if piece:
                            content += piece
                            if "<think" in content:
                                emit_ok = False
                            if emit_ok:
                                on_delta(piece)
                        for tc in delta.get("tool_calls") or []:
                            slot = calls.setdefault(
                                int(tc.get("index") or 0),
                                {"id": "", "function": {"name": "", "arguments": ""}},
                            )
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                slot["function"]["arguments"] += fn["arguments"]
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(str(exc)) from exc
        raw: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            raw["tool_calls"] = [
                {"id": c["id"], "type": "function", "function": c["function"]}
                for _, c in sorted(calls.items())
            ]
        return _normalize_openai_message(raw)

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
        url = f"{self._openai_base()}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self._config.request_timeout_s) as client:
                resp = await client.post(url, json=payload, headers=self._openai_headers())
                if resp.status_code == 400 and tools:
                    # Some hosted models (e.g. Groq's allam-2-7b) reject tool
                    # schemas outright with a 400. Retry once without tools so
                    # plain questions still get answered — tool skills simply
                    # won't run through this model.
                    detail = ""
                    try:
                        detail = str(resp.json().get("error", {}).get("message", ""))
                    except Exception:
                        pass
                    if "tool" in detail.lower() or "function" in detail.lower():
                        logger.info("model rejected tool schemas (%s); retrying without tools",
                                    detail[:100])
                        payload.pop("tools", None)
                        resp = await client.post(url, json=payload, headers=self._openai_headers())
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
