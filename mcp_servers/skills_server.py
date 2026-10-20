"""MEDO's skills, offered to other local AI tools over MCP.

MEDO already speaks MCP as a *client* (``core/mcp.py``): it connects to other
applications' servers and turns their tools into skills. This is the other
direction. Point Claude Desktop, an editor's assistant, or anything else that
speaks the protocol at this server and MEDO's own capabilities — the weather,
the news, the specialist council, the resistor decoder — become tools those
tools can call.

    # claude_desktop_config.json
    "medo": {
      "command": "C:/…/MEDO-main/.venv/Scripts/python.exe",
      "args": ["-m", "mcp_servers.skills_server"]
    }

**Read-only by default, and that default is the point.** Anything that speaks to
this server is, by construction, software MEDO's owner did not write, driven by
a model whose prompt they cannot see. Handing that "shut down the computer" is
not a feature. So the exposed set is filtered to skills that only ANSWER:

* nothing that declares an actuation capability,
* nothing that sets ``controls_pc``,
* nothing that requires confirmation — a confirmation prompt sent down an MCP
  pipe has nobody to answer it, and a caller that could answer it would be
  answering on the user's behalf,
* nothing on the explicit ``never`` list, which catches skills that are
  read-only in the capability sense but leak in the practical one.

Widening it is possible (``mcp.server_expose``) and deliberately awkward: you
name each skill individually, and the reason to is that you trust the specific
program on the other end.

The server runs in its own process and builds its own registry, so it works
whether or not MEDO itself is running. Both processes share the SQLite
databases, which is safe because they are opened in WAL mode (``core/db.py``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from security.capabilities import ACTUATION_CAPS, effective_capabilities
from skills.base import Skill, SkillRegistry, SkillRequest

logger = logging.getLogger(__name__)

#: Never exposed, whatever their capability declarations say.
#:
#: The capability vocabulary answers "does this actuate the machine?", which is
#: the right question for the PC-control switch and the wrong one here. Auditing
#: what the plain filter actually let through turned up three kinds of skill
#: that declare no actuation and should still never be handed to another
#: program:
#:
#: * **Writes to the owner's own memory** — remember_fact, forget_fact, notes,
#:   obsidian_note. A caller that can plant a fact in MEDO's long-term memory
#:   can change what MEDO tells its owner tomorrow. That is prompt injection
#:   with persistence, and it costs one tool call.
#: * **Reveals what is in the room** — see_screen, see_camera, what_is_this,
#:   see_bench. Reading is not acting, but a webcam frame and a screenshot are
#:   the two most private things this machine holds, and "read-only" is a poor
#:   reason to put them on a pipe to someone else's model.
#: * **Reveals the conversation** — repeat, second_opinion, recall_facts,
#:   recall_conversation. These read back what the owner said to MEDO.
#:
#: The rest (quit, modes, dictate, lion_mode, self_dev, security) end or reshape
#: MEDO's own session. Any of these can still be exposed by naming it in
#: ``mcp.server_expose``, which is the point of that setting being per-skill.
NEVER_EXPOSE = frozenset({
    # session control. These are registered CONDITIONALLY (lion mode, config
    # flags), so they are absent from a default registry — listed by their real
    # registry names anyway, because a deny list has to be right for the build
    # where the skill exists, not for the one where it doesn't.
    "quit", "session_mode", "dictation", "lion_mode", "self_dev",
    # writes to the owner's memory / files
    "remember_fact", "forget_fact", "notes", "obsidian_note",
    "make_document", "make_app", "projects",
    # writes a file containing the owner's RECENT CONVERSATION. Not read-only
    # on either count, and there is no reading of "another AI tool files MEDO's
    # bug reports on the owner's behalf" that makes sense. The MCP path already
    # marks every call untrusted and the skill refuses those, so exposing it
    # would only advertise a tool that always says no — noise in every caller's
    # tool list, and a needless hint that the owner's conversation is on disk.
    "bug_report",
    # names everything the owner has designed, and its one actuating branch
    # opens a folder on their screen. The three skills that BUILD are already
    # excluded (they declare controls_pc), but listing the results is the same
    # kind of disclosure as `projects` and `obsidian_search` above: what someone
    # is working on is theirs to share.
    "studio_projects",
    # reveals the room
    "see_screen", "see_camera", "what_is_this", "see_bench",
    # reveals the conversation
    "repeat", "second_opinion", "recall_facts", "recall_conversation",
    # reveals the owner's private notes and their day
    "obsidian_search", "briefing",
    # sets state that makes MEDO speak later, unprompted, in the owner's room
    "timers",
    # delegates a whole multi-step task. It declares no capabilities of its own
    # — each STEP is gated instead — so the read-only filter waves it through,
    # which is precisely the case this list exists for. Two reasons it must not
    # be here: the approval that makes Cowork safe is a spoken conversation, and
    # there is nobody on the MCP path to have it; and a task delegated by other
    # software is exactly the "content became an instruction" the trust boundary
    # forbids. CoworkSkill refuses untrusted callers itself as well.
    "cowork",
    # local reconnaissance: listening ports, running processes, the firewall
    # profile, installed updates, file permissions. Every one is read-only and
    # every one is the first thing you would want if you were mapping this
    # machine from the outside.
    "security_check", "explain_process", "firewall_audit", "security_updates",
    "explain_permissions",
})

#: The prefix every exposed tool gets, so a caller's tool list makes it obvious
#: where these came from and two servers can't collide on a name.
TOOL_PREFIX = "medo_"


def is_read_only(skill: Skill) -> bool:
    """Can this skill only ANSWER, never act?

    Deliberately conservative and layered: a skill has to pass every check, and
    a skill that has declared nothing at all is treated as unsafe by the
    ``controls_pc`` bridge in :func:`effective_capabilities` rather than assumed
    harmless.
    """
    if skill.name in NEVER_EXPOSE:
        return False
    if getattr(skill, "requires_confirmation", False):
        return False
    if getattr(skill, "controls_pc", False):
        return False
    return not (effective_capabilities(skill) & ACTUATION_CAPS)


def exposed_skills(registry: SkillRegistry, *, extra: list[str] | None = None,
                   allow_actuation: bool = False) -> list[Skill]:
    """The skills this server offers, in registry order.

    ``extra`` names skills to expose even though they are not read-only — each
    one individually, never a wildcard. ``allow_actuation`` drops the filter
    entirely and exists for a single case: an operator who has read this
    docstring and is pointing the server at something they wrote themselves.
    """
    named = {n.strip() for n in (extra or []) if n.strip()}
    out = []
    for skill in registry.all():
        if allow_actuation:
            out.append(skill)
        elif is_read_only(skill) or skill.name in named:
            out.append(skill)
    return out


def tool_definition(skill: Skill) -> dict[str, Any]:
    """One skill as an MCP tool definition."""
    schema = skill.tool_schema()
    function = schema.get("function", {})
    return {
        "name": TOOL_PREFIX + skill.name,
        "description": (function.get("description") or skill.description
                        or skill.name),
        "inputSchema": function.get("parameters")
                       or {"type": "object", "properties": {}},
    }


async def call_skill(registry: SkillRegistry, tool_name: str,
                     arguments: dict[str, Any], *,
                     allowed: set[str] | None = None) -> str:
    """Run one exposed skill and return its speech as text.

    The allowlist is re-checked HERE, not only when the tool list was built. A
    client is free to call a name it was never offered, and "we didn't advertise
    it" is not an access control.
    """
    name = tool_name[len(TOOL_PREFIX):] if tool_name.startswith(TOOL_PREFIX) \
        else tool_name
    if allowed is not None and name not in allowed:
        return f"MEDO does not expose a tool called {tool_name!r}."
    skill = registry.get(name)
    if skill is None:
        return f"MEDO has no skill called {name!r}."
    request = SkillRequest(text=name.replace("_", " "),
                           args=dict(arguments or {}),
                           # Marked as coming from outside so the trust boundary
                           # treats anything it returns as untrusted content,
                           # exactly like a fetched web page.
                           context={"source": "mcp", "untrusted": True})
    try:
        result = await skill.execute(request)
    except Exception as exc:  # noqa: BLE001 - report, never kill the server
        logger.warning("MCP call to %s failed", name, exc_info=True)
        return f"That skill failed: {exc}"
    if result.needs_confirmation:
        # Unreachable through the filter, and refused anyway: a confirmation
        # sent down a pipe would be answered by the caller, not by the user.
        return ("That action needs the owner to confirm it out loud, so it "
                "can't be run over MCP.")
    return result.speech


def build_server(registry: SkillRegistry, skills: list[Skill]):
    """An ``mcp.server.Server`` exposing ``skills``. Imported lazily."""
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    allowed = {s.name for s in skills}
    server = Server("medo")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [Tool(**tool_definition(s)) for s in skills]

    @server.call_tool()
    async def handle_call(name: str, arguments: dict) -> list[TextContent]:
        text = await call_skill(registry, name, arguments, allowed=allowed)
        return [TextContent(type="text", text=text)]

    return server


async def _serve() -> None:  # pragma: no cover - needs a live stdio peer
    from mcp.server.stdio import stdio_server

    from core.config import load_settings
    from main import Announcer, build_registry

    settings = load_settings()
    registry = build_registry(settings, Announcer())
    skills = exposed_skills(
        registry,
        extra=getattr(settings.mcp, "server_expose", []),
        allow_actuation=getattr(settings.mcp, "server_allow_actuation", False))
    logger.info("MEDO MCP server: offering %d read-only skill(s)", len(skills))
    server = build_server(registry, skills)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def _main(argv=None) -> int:
    """``python -m mcp_servers.skills_server`` — serve, or ``--list`` and exit."""
    import argparse

    from core.config import load_settings
    from main import Announcer, build_registry

    parser = argparse.ArgumentParser(description="MEDO's skills, over MCP")
    parser.add_argument("--list", action="store_true",
                        help="print what would be exposed, and exit")
    args = parser.parse_args(argv)

    if args.list:
        settings = load_settings()
        registry = build_registry(settings, Announcer())
        skills = exposed_skills(
            registry,
            extra=getattr(settings.mcp, "server_expose", []),
            allow_actuation=getattr(settings.mcp, "server_allow_actuation", False))
        offered = {s.name for s in skills}
        print(f"Exposed ({len(skills)}):")
        for skill in skills:
            print(f"  {TOOL_PREFIX}{skill.name:<24} {skill.description[:60]}")
        held_back = [s for s in registry.all() if s.name not in offered]
        print(f"\nWithheld ({len(held_back)}) — not read-only:")
        for skill in held_back:
            print(f"  {skill.name}")
        return 0

    logging.basicConfig(level=logging.INFO)
    asyncio.run(_serve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
