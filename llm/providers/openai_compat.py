"""Cloud brain — any OpenAI-compatible ``/chat/completions`` endpoint.

This ONE adapter covers most clouds, differing only by config (base_url + key +
model): DeepSeek, Kimi/Moonshot, GLM/Zhipu, Gemini (via its OpenAI-compat
endpoint), and OpenAI itself. It wraps the existing
:class:`llm.client.LLMClient` pinned to the ``openai`` provider, so streaming,
tool-call normalization, the 400-without-tools retry, and think-tag stripping
are the exact same, single code path the local brain uses.

``is_local`` is always False: choosing one of these means the prompt leaves the
machine, which the privacy model (S3/S4) makes explicit and visible.

:data:`KNOWN_OPENAI_COMPAT` records the endpoints VERIFIED from each provider's
live docs (July 2026) as sane config defaults — but they are only reference
data here; the real registry is config-driven (S2), and the user sets/updates
the model id and key. Model IDs move fast, so these are examples to verify, not
hardcoded truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.config import LLMConfig
from llm.client import LLMClient
from llm.providers.base import BrainProvider


@dataclass(frozen=True)
class KnownEndpoint:
    """A verified OpenAI-compatible cloud, as a config starting point."""

    base_url: str
    example_models: tuple[str, ...]
    note: str = ""


#: Verified from each provider's live OpenAI-compatibility docs (July 2026).
#: Reference only — S2 puts the real registry in config.yaml, and model IDs are
#: expected to be checked/edited by the user (they change often).
KNOWN_OPENAI_COMPAT: dict[str, KnownEndpoint] = {
    "openai": KnownEndpoint(
        "https://api.openai.com/v1", ("gpt-4o", "gpt-4o-mini")),
    "deepseek": KnownEndpoint(
        "https://api.deepseek.com", ("deepseek-v4-flash", "deepseek-v4-pro"),
        "deepseek-chat/deepseek-reasoner deprecated 2026-07-24 -> v4-flash/pro"),
    "gemini": KnownEndpoint(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        ("gemini-2.5-flash", "gemini-2.5-pro"),
        "trailing slash on the base_url is required (404 without it)"),
    "kimi": KnownEndpoint(
        "https://api.moonshot.ai/v1", ("kimi-k2.6",),
        "China endpoint is https://api.moonshot.cn/v1"),
    "glm": KnownEndpoint(
        "https://open.bigmodel.cn/api/paas/v4/", ("glm-4.5", "glm-4.5-flash"),
        "international endpoint is https://api.z.ai/api"),
}


class OpenAICompatProvider(BrainProvider):
    """A cloud model behind an OpenAI-compatible endpoint. Never local."""

    is_local = False

    def __init__(self, name: str, model: str, base_url: str, api_key: str,
                 config: LLMConfig, client: LLMClient | None = None) -> None:
        self.name = name
        self.model_id = model
        self._base_url = base_url
        self._has_key = bool(api_key)
        # Reuse the existing OpenAI path, pinned to THIS brain's endpoint + key.
        # The key lives only in this in-memory config copy (loaded from the
        # git-ignored secrets file at wiring time, S2) — never logged.
        self._client = client or LLMClient(config.model_copy(update={
            "provider": "openai",
            "openai_base_url": base_url,
            "api_key": api_key,
        }))

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
        """Usable when a key is present.

        A cloud brain with no key is simply unavailable (and S3 says so). The
        endpoint being *reachable* is confirmed at call time, where a failure
        falls back to the local default and is announced (S4) — probing every
        cloud on every availability check would make the HUD picker slow and
        leak traffic the user didn't ask for.
        """
        return self._has_key
