"""MCP client — connect any application that speaks the Model Context Protocol.

Servers are declared in ``config.yaml`` under ``mcp.servers`` (a local command
for stdio transport, or a URL for Streamable HTTP). At startup
:class:`MCPManager` connects to each enabled server, lists its tools, and wraps
every tool as a :class:`~skills.base.Skill` registered into the normal
:class:`~skills.base.SkillRegistry` — so MCP tools are callable by the LLM path
exactly like built-in skills and plugins, with zero per-app code.

Example config::

    mcp:
      enabled: true
      servers:
        filesystem:
          command: "npx"
          args: ["-y", "@modelcontextprotocol/server-filesystem", "~/Documents"]
        home-assistant:
          url: "http://homeassistant.local:8123/mcp"

Design notes:

* The ``mcp`` package is imported lazily; if it isn't installed (or no servers
  are configured) MEDO runs exactly as before.
* A server that fails to connect is logged and skipped — never fatal.
* Tool names are namespaced ``<server>_<tool>`` and sanitized to the
  ``[a-zA-Z0-9_]`` charset function-calling requires.
* Sessions live on the app's asyncio loop inside one ``AsyncExitStack``;
  ``stop()`` unwinds them all.
"""

from __future__ import annotations

import asyncio
import logging
import re
from contextlib import AsyncExitStack
from typing import Any

from core.config import MCPConfig, MCPServerConfig
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

#: Function-calling tool names must match this; anything else is replaced by _.
_NAME_SANITIZER = re.compile(r"[^a-zA-Z0-9_]+")

#: Cap what a tool result contributes to the conversation (they can be huge).
MAX_RESULT_CHARS = 4000


def _sanitize(name: str) -> str:
    return _NAME_SANITIZER.sub("_", name).strip("_") or "tool"


def _result_text(result: Any) -> str:
    """Flatten an MCP CallToolResult into plain text for the LLM/TTS."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
        elif getattr(block, "type", "") == "image":
            parts.append("[image]")
    text = "\n".join(parts).strip() or "(the tool returned no text)"
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + " …(truncated)"
    return text


class MCPToolSkill(Skill):
    """One MCP tool exposed as a MEDO skill (LLM path only — no regex)."""

    patterns: list[re.Pattern[str]] = []

    def __init__(self, manager: "MCPManager", server: str, tool: Any) -> None:
        self._manager = manager
        self._server = server
        self._tool_name = tool.name
        self.name = _sanitize(f"{server}_{tool.name}")
        raw_desc = (tool.description or "").strip() or tool.name
        # One line, bounded: descriptions land in every LLM request.
        self.description = f"[{server} app] " + " ".join(raw_desc.split())[:300]
        schema = getattr(tool, "inputSchema", None)
        self._parameters: dict[str, Any] = (
            schema if isinstance(schema, dict) else {"type": "object", "properties": {}}
        )

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._parameters,
            },
        }

    async def execute(self, request: SkillRequest) -> SkillResult:
        try:
            text = await self._manager.call_tool(
                self._server, self._tool_name, request.args
            )
        except Exception as exc:
            logger.warning("MCP tool %s failed: %s", self.name, exc)
            return SkillResult(
                f"The {self._server} tool failed: {str(exc)[:120]}", success=False
            )
        return SkillResult(text)


class MCPManager:
    """Owns the MCP client sessions and the skills wrapped around their tools."""

    def __init__(self, config: MCPConfig) -> None:
        self._config = config
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, Any] = {}
        self._tools: dict[str, list[Any]] = {}   # server -> ListTools result
        self._errors: dict[str, str] = {}        # server -> why it's not connected

    # -- lifecycle -----------------------------------------------------------

    async def start(self, registry: SkillRegistry) -> int:
        """Connect every enabled server and register its tools as skills.

        Returns the number of tools registered. Never raises: a bad server (or
        the ``mcp`` package missing entirely) degrades to a logged warning.
        """
        servers = {
            name: cfg
            for name, cfg in (self._config.servers or {}).items()
            if cfg.enabled and (cfg.command or cfg.url)
        }
        if not self._config.enabled or not servers:
            return 0
        try:
            import mcp  # noqa: F401
        except ImportError:
            logger.warning(
                "mcp servers are configured but the 'mcp' package is not "
                "installed — run: pip install mcp"
            )
            self._errors.update({n: "mcp package not installed" for n in servers})
            return 0

        self._stack = AsyncExitStack()
        registered = 0
        for name, cfg in servers.items():
            try:
                session = await asyncio.wait_for(
                    self._connect(name, cfg), timeout=self._config.connect_timeout_s
                )
                listing = await asyncio.wait_for(
                    session.list_tools(), timeout=self._config.connect_timeout_s
                )
                self._sessions[name] = session
                self._tools[name] = list(listing.tools)
            except Exception as exc:
                self._errors[name] = str(exc)[:200]
                logger.warning("MCP server %r unavailable: %s", name, exc)
                continue
            for tool in self._tools[name]:
                skill = MCPToolSkill(self, name, tool)
                try:
                    registry.register(skill)
                    registered += 1
                except ValueError:  # duplicate name (two servers, same tool)
                    logger.warning("skipping duplicate MCP tool name %r", skill.name)
            logger.info(
                "MCP server %r connected: %d tool(s)", name, len(self._tools[name])
            )
        return registered

    async def _connect(self, name: str, cfg: MCPServerConfig) -> Any:
        """Open one session (stdio or Streamable HTTP) on the shared stack."""
        from mcp import ClientSession

        assert self._stack is not None
        if cfg.command:
            import os

            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=cfg.command,
                args=list(cfg.args),
                env={**os.environ, **cfg.env} if cfg.env else None,
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        else:
            from mcp.client.streamable_http import streamablehttp_client

            # Optional bearer auth for a protected HTTP server.
            headers = ({"Authorization": f"Bearer {cfg.auth_token}"}
                       if cfg.auth_token else None)
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(cfg.url, headers=headers)
            )
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def stop(self) -> None:
        """Close every session/transport (no-op when nothing connected)."""
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:  # transports may already be gone at shutdown
                logger.debug("error closing MCP sessions", exc_info=True)
            self._stack = None
        self._sessions.clear()

    # -- calls & status ------------------------------------------------------

    async def call_tool(self, server: str, tool: str, args: dict[str, Any]) -> str:
        session = self._sessions.get(server)
        if session is None:
            raise RuntimeError(f"MCP server {server!r} is not connected")
        result = await asyncio.wait_for(
            session.call_tool(tool, args or {}), timeout=self._config.call_timeout_s
        )
        return _result_text(result)

    def status(self) -> list[dict[str, Any]]:
        """Connection summary for UIs (HUD CONFIG tab)."""
        out: list[dict[str, Any]] = []
        for name, cfg in (self._config.servers or {}).items():
            if not cfg.enabled:
                out.append({"name": name, "state": "disabled", "tools": []})
            elif name in self._sessions:
                out.append(
                    {
                        "name": name,
                        "state": "connected",
                        "tools": [t.name for t in self._tools.get(name, [])],
                    }
                )
            else:
                out.append(
                    {
                        "name": name,
                        "state": "error",
                        "error": self._errors.get(name, "not connected"),
                        "tools": [],
                    }
                )
        return out
