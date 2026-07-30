"""Cloud data-egress gate (S5): local context never leaves for a cloud brain
without an explicit per-brain opt-in.

MEDO is local-first. When the selected brain is a CLOUD provider, MEDO must not
send your local RAG documents, memory/facts, or file content to it unless you
opted THAT brain in (`security.cloud_egress_optin`). Default: withhold — a
cloud brain gets your question, not your private local context.
"""

from __future__ import annotations

from typing import Any

#: Providers that send text off this machine. Ollama is local → never egress.
CLOUD_PROVIDERS = frozenset({"openai", "anthropic", "claude-code", "codex"})

#: Tools whose OUTPUT is local private content — not offered to a cloud brain
#: unless egress is opted in (RAG documents, file listings/contents).
LOCAL_CONTENT_TOOLS = frozenset({
    "search_documents", "files", "read_file", "open_in_editor", "import_file",
})


def is_cloud(provider: str) -> bool:
    return provider in CLOUD_PROVIDERS


def local_context_allowed(security_config: Any, provider: str) -> bool:
    """May local RAG/memory/file context be sent to this brain? Always yes for a
    local provider; for a cloud one, only if it's opted in."""
    if not is_cloud(provider):
        return True
    optin = getattr(security_config, "cloud_egress_optin", None) or {}
    return bool(optin.get(provider))
