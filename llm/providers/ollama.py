"""Local brain — an Ollama model on THIS machine (the default, always private).

A thin adapter: it wraps the existing :class:`llm.client.LLMClient` pinned to
the ``ollama`` provider and delegates. That means the current qwen path is
byte-for-byte unchanged — same streaming, same ``keep_alive``/``num_ctx``, same
think-tag stripping — and this class adds only the :class:`BrainProvider`
surface on top. Nothing about today's behaviour moves.
"""

from __future__ import annotations

from typing import Any

from core.config import LLMConfig
from llm.client import LLMClient
from llm.providers.base import BrainProvider


class OllamaProvider(BrainProvider):
    """A local Ollama model. ``is_local`` is always True — it never leaves the box."""

    is_local = True

    def __init__(self, name: str, model: str, config: LLMConfig,
                 client: LLMClient | None = None) -> None:
        self.name = name
        self.model_id = model
        # Pin a copy of the config to Ollama so every other field (host,
        # keep_alive, num_ctx, request_timeout) is inherited unchanged. `client`
        # is injectable for tests.
        self._client = client or LLMClient(config.model_copy(
            update={"provider": "ollama"}))

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        on_delta: Any = None,
    ) -> dict[str, Any]:
        return await self._client.chat(
            self.model_id, messages,
            tools=tools, temperature=temperature, on_delta=on_delta)

    def available(self) -> bool:
        """Usable when the model is actually pulled into Ollama.

        ``list_models`` returns ``[]`` when Ollama is down, so this is False
        then too — never raises.
        """
        try:
            return self.model_id in set(self._client.list_models())
        except Exception:      # noqa: BLE001 - availability must never raise
            return False
