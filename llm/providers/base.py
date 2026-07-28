"""The `BrainProvider` interface — one selectable LLM "brain".

MEDO is model-agnostic behind this interface: the router and skills call
:meth:`BrainProvider.generate` and never care whether the answer came from a
local Ollama model or a cloud endpoint. That indirection is what lets a brain
be added as a *config entry* (S2) rather than code — and, crucially, it is the
one place the privacy model can be enforced, because every provider knows
whether it :attr:`is_local`.

The contract each provider must honour, so callers stay provider-agnostic:

* ``generate`` returns the SAME assistant-message dict shape as the existing
  :meth:`llm.client.LLMClient.chat` — ``{"content": str, "tool_calls"?: [...]}``
  — with think-tag reasoning already stripped and the same ``on_delta``
  streaming protocol. Providers achieve this by delegating to that one client
  (composition), so the reasoning-strip / streaming choke-point stays SINGLE.
* ``available`` is a cheap, never-raising "is this usable right now" check.

The privacy principle lives on the flags, not in scattered ifs:

* ``is_local`` True  → data never leaves the machine; this is the default and
  the fallback, always available.
* ``is_local`` False → a cloud brain; using it means data leaves the machine,
  so its selection is explicit, per-session confirmed, and visibly indicated
  (S3), and local RAG/memory context is not attached unless opted in (S4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BrainProvider(ABC):
    """One LLM brain the router can select. Local or cloud, one interface."""

    #: Registry id the user selects by, e.g. ``"qwen-local"`` / ``"deepseek"``.
    name: str
    #: True when the model runs on THIS machine (no data leaves). The whole
    #: privacy model keys off this: local is default + fallback and always
    #: available; cloud is opt-in, confirmed, and indicated.
    is_local: bool
    #: The concrete model id handed to the backend (e.g. ``"qwen3:30b"``).
    model_id: str

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        on_delta: Any = None,
    ) -> dict[str, Any]:
        """Chat completion → assistant message ``{content, tool_calls?}``.

        Identical contract to :meth:`llm.client.LLMClient.chat`: think-tag
        stripping applied, ``tool_calls`` arguments always a dict, and ``on_delta``
        (``Callable[[str], None]``) streaming content chunks when provided.
        Raises :class:`llm.client.LLMUnavailableError` when the backend can't be
        reached, so the caller can fall back to local (S4).
        """

    @abstractmethod
    def available(self) -> bool:
        """Cheap, never-raises: is this brain usable right now?

        Local → the model is present in Ollama. Cloud → an API key is present
        (endpoint reachability is confirmed at call time, with graceful
        fallback to local — see S4).
        """

    # -- shared conveniences (never overridden) ------------------------------

    @property
    def is_cloud(self) -> bool:
        return not self.is_local

    @property
    def location(self) -> str:
        """``"local"`` / ``"cloud"`` — the word the HUD badge and the spoken
        privacy notice use."""
        return "local" if self.is_local else "cloud"

    def describe(self) -> str:
        return f"{self.name} · {self.location} · {self.model_id}"

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<{type(self).__name__} {self.describe()}>"
