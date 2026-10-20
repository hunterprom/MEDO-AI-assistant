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


def _is_loopback(host: str) -> bool:
    h = (host or "").strip().lower()
    if not h:
        return True                       # unset host defaults to local Ollama
    return "127.0.0.1" in h or "localhost" in h or "::1" in h


def sends_off_machine(provider: str, host: str = "") -> bool:
    """True if this brain sends text OFF this machine — a cloud provider, OR an
    Ollama pointed at a non-loopback host (a LAN/remote Ollama is not local)."""
    if is_cloud(provider):
        return True
    return provider == "ollama" and not _is_loopback(host)


def local_context_allowed(security_config: Any, provider: str,
                          host: str = "") -> bool:
    """May local RAG/memory/file context be sent to this brain? Yes only when the
    brain runs on THIS machine; for anything off-box (cloud OR a remote Ollama),
    only with an opt-in (by provider name or by the exact host)."""
    if not sends_off_machine(provider, host):
        return True
    optin = getattr(security_config, "cloud_egress_optin", None) or {}
    return bool(optin.get(provider) or (host and optin.get(host)))
