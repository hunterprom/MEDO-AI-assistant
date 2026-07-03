"""Glue between the skill registry and Ollama function calling.

The same skills that serve the fast path expose a ``tool_schema``; here we collect
those into the ``tools`` list Ollama expects, and dispatch a model's tool call
back to the owning skill's ``execute`` — so there's genuinely one implementation
behind both routes.
"""

from __future__ import annotations

import logging
from typing import Any

from skills.base import SkillRegistry, SkillRequest, SkillResult

logger = logging.getLogger(__name__)


def build_tools(registry: SkillRegistry) -> list[dict[str, Any]]:
    """All non-empty tool schemas, ready to hand to Ollama's ``tools`` param."""
    tools: list[dict[str, Any]] = []
    for skill in registry.all():
        schema = skill.tool_schema()
        if schema and schema.get("function", {}).get("name"):
            tools.append(schema)
    return tools


async def dispatch_tool(
    registry: SkillRegistry,
    name: str,
    arguments: dict[str, Any],
    context: dict[str, Any],
) -> SkillResult:
    """Run the skill named by a model tool call with its parsed arguments."""
    skill = registry.get(name)
    if skill is None:
        return SkillResult(f"I don't have a tool called {name}.", success=False)
    # Reconstruct a natural utterance so pattern-driven skills still behave, and
    # pass the structured args the LLM chose. ``via=tool`` lets a skill know it's
    # being called by the model (e.g. web search returns raw results to summarize).
    request = SkillRequest(
        text=_synthesize_text(name, arguments),
        args=arguments,
        context={**context, "via": "tool"},
    )
    logger.info("tool call: %s(%s)", name, arguments)
    return await skill.execute(request)


def _synthesize_text(name: str, arguments: dict[str, Any]) -> str:
    """Best-effort natural phrasing of a tool call for pattern-based skills."""
    if not arguments:
        return name.replace("_", " ")
    parts = " ".join(str(v) for v in arguments.values())
    return f"{name.replace('_', ' ')} {parts}".strip()
