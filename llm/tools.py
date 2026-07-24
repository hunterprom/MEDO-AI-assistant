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


def coerce_args(skill, arguments: dict[str, Any]) -> dict[str, Any]:
    """Coerce model values for STRING-typed params to ``str``.

    Models sometimes emit a number/bool where a string is declared — a PID as
    ``1234``, a path as a number — and skills call ``.strip()`` on the value,
    which would raise ``AttributeError`` instead of answering. This coerces
    only params the skill's own schema declares ``"string"``, so genuinely
    numeric params (e.g. browser scroll ``amount``: integer) are left alone.
    Idempotent and safe to call more than once.
    """
    if not arguments or skill is None:
        return arguments
    try:
        props = skill.tool_schema()["function"]["parameters"].get("properties", {})
    except Exception:      # a skill with an unusual schema — don't touch its args
        return arguments
    out = dict(arguments)
    for key, val in arguments.items():
        if (props.get(key, {}).get("type") == "string"
                and val is not None and not isinstance(val, str)):
            out[key] = str(val)
    return out


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
    arguments = coerce_args(skill, arguments)
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
