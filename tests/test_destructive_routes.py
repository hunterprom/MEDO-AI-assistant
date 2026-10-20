"""P0-1c: no route reaches a destructive skill without an explicit yes.

MEDO has a lot of front doors — a regex, an embedding match, a model's tool
call, an MCP server's tool, a dropped-in plugin, an app connector, a paired
device's capability, the companion API, a scheduled routine, a user-defined
trigger phrase. Each of those was individually careful. What was missing was a
statement that holds across all of them at once, and a test that keeps it true
when the eleventh front door is added.

The instrument is :class:`_Landmine`: a skill that declares
``requires_confirmation`` and then, deliberately, does NOT check
``context["confirmed"]``. It acts the instant it is executed. Every test below
drives it down one route and asserts ``fired == 0``. If a route ever executes a
declared-destructive skill without confirmation, the mine goes off and the test
says which door was open — no test needs to guess at the wording of a prompt or
the shape of a result.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from core.config import load_settings
from core.events import EventBus
from core.router import NON_INTERACTIVE_SOURCES, RoutePath, Router
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult


class _Landmine(Skill):
    """Destructive, and careless: it acts as soon as anyone executes it."""

    name = "landmine"
    description = "erase the whole disk"
    routing_phrases = ["erase the whole disk", "wipe my computer"]
    patterns = [re.compile(r"\berase the disk\b", re.IGNORECASE)]
    requires_confirmation = True

    def __init__(self) -> None:
        self.fired = 0
        self.confirmed_when_fired: list[bool] = []

    async def execute(self, request: SkillRequest) -> SkillResult:
        self.fired += 1
        self.confirmed_when_fired.append(bool(request.context.get("confirmed")))
        return SkillResult("Erased.")

    def tool_schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": {"type": "object", "properties": {}}}}


class _Clock(Skill):
    name = "clock"
    description = "tell the time"
    patterns = [re.compile(r"\bwhat time is it\b", re.IGNORECASE)]

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("It's noon.")


class _ToolCallingLLM:
    """A model that always calls the tool it is told to call."""

    model = "m"

    def __init__(self, *names: str) -> None:
        self._names = names or ("landmine",)

    async def chat(self, model, messages, tools=None, **kw):
        if any(m.get("role") == "tool" for m in messages):
            return {"content": "all done"}
        return {"content": "", "tool_calls": [
            {"function": {"name": n, "arguments": {}}} for n in self._names]}


def _router(*skills: Skill, llm=None, **flags) -> Router:
    settings = load_settings()
    settings.safety.confirm_destructive = True
    settings.safety.pc_control_enabled = True
    settings.security.yolo = False
    for key, value in flags.items():
        section, _, field = key.partition("__")
        setattr(getattr(settings, section), field, value)
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    router = Router(settings, registry, llm or _ToolCallingLLM(), EventBus())
    if llm is None:
        router.model = None          # no LLM unless a test asked for one
    return router


# --- tier 1: the regex fast path ---------------------------------------------


@pytest.mark.asyncio
async def test_the_fast_path_does_not_fire_it():
    mine = _Landmine()
    router = _router(mine)
    await router.route("erase the disk")
    assert mine.fired == 0
    assert router.awaiting_confirmation


@pytest.mark.asyncio
async def test_the_fast_path_fires_it_only_after_yes():
    mine = _Landmine()
    router = _router(mine)
    await router.route("erase the disk")
    result = await router.route("yes")
    assert mine.fired == 1
    assert mine.confirmed_when_fired == [True]
    assert "Erased" in result.speech


@pytest.mark.asyncio
async def test_no_cancels_it_outright():
    mine = _Landmine()
    router = _router(mine)
    await router.route("erase the disk")
    await router.route("no")
    assert mine.fired == 0
    assert not router.awaiting_confirmation


@pytest.mark.asyncio
async def test_an_ambiguous_reply_does_not_fire_it():
    """"Maybe" is not a yes. The pending action is dropped, not guessed at."""
    mine = _Landmine()
    router = _router(mine, _Clock())
    await router.route("erase the disk")
    await router.route("what time is it")
    assert mine.fired == 0
    assert not router.awaiting_confirmation


@pytest.mark.asyncio
async def test_a_custom_trigger_phrase_cannot_skip_the_gate():
    """``skills.triggers`` adds wording, never permission (core/triggers.py)."""
    from core.triggers import apply_triggers

    mine = _Landmine()
    router = _router(mine)
    apply_triggers(router.registry, {"landmine": ["tidy up my machine"]})
    await router.route("tidy up my machine")
    assert mine.fired == 0
    assert router.awaiting_confirmation


# --- tier 2: the semantic (embedding) tier -----------------------------------


def test_the_semantic_tier_refuses_to_index_it():
    assert Router._semantic_safe(_Landmine()) is False


@pytest.mark.asyncio
async def test_the_semantic_tier_never_routes_to_it():
    """Even with the tier on and an embedder that says it is a perfect match."""
    mine = _Landmine()
    router = _router(mine, router__semantic_enabled=True,
                     router__semantic_shadow=False)
    router._route_index = None
    match = await router._semantic_route("wipe my computer completely")
    assert match is None or match[0] is not mine
    result = await router.route("wipe my computer completely")
    assert mine.fired == 0
    assert result.path is not RoutePath.SEMANTIC or result.skill_name != mine.name


@pytest.mark.asyncio
async def test_a_semantic_clarify_choice_cannot_offer_it():
    """The either/or tie-break only ever offers skills the tier could route to."""
    mine = _Landmine()
    router = _router(mine, _Clock(), router__semantic_enabled=True)
    cands = await router._semantic_clarify("wipe my computer")
    assert mine not in (cands or [])


# --- tier 3: the LLM tool loop -----------------------------------------------


@pytest.mark.asyncio
async def test_a_model_tool_call_does_not_fire_it():
    mine = _Landmine()
    router = _router(mine, llm=_ToolCallingLLM("landmine"))
    await router.route("clean things up for me")
    assert mine.fired == 0
    assert router.awaiting_confirmation


@pytest.mark.asyncio
async def test_a_model_tool_call_fires_it_only_after_yes():
    mine = _Landmine()
    router = _router(mine, llm=_ToolCallingLLM("landmine"))
    await router.route("clean things up for me")
    await router.route("yes")
    assert mine.confirmed_when_fired == [True]


@pytest.mark.asyncio
async def test_hiding_it_mid_batch_does_not_fire_it():
    """A model that calls a harmless tool first must not smuggle the mine past."""
    mine = _Landmine()
    router = _router(mine, _Clock(), llm=_ToolCallingLLM("clock", "landmine"))
    result = await router.route("tell me the time and clean up")
    assert mine.fired == 0
    assert "noon" in result.speech          # what did run is still reported


@pytest.mark.asyncio
async def test_calling_it_twice_in_one_batch_does_not_fire_it():
    mine = _Landmine()
    router = _router(mine, llm=_ToolCallingLLM("landmine", "landmine"))
    await router.route("clean up, twice")
    assert mine.fired == 0


# --- the other registries: MCP, plugins, connectors, devices -----------------


@pytest.mark.asyncio
async def test_an_mcp_tool_marked_destructive_is_gated():
    """MCP tools are ordinary skills, so they inherit the same choke point."""
    from core.mcp import MCPToolSkill

    fired: list[str] = []

    class _Manager:
        async def call_tool(self, server, name, args):
            fired.append(name)
            return "done"

    tool = type("T", (), {"name": "rm_rf", "description": "delete a tree",
                          "inputSchema": {"type": "object", "properties": {}}})()
    skill = MCPToolSkill(_Manager(), "files", tool)
    skill.requires_confirmation = True
    router = _router(skill, llm=_ToolCallingLLM(skill.name))
    await router.route("clear the build directory")
    assert fired == []
    assert router.awaiting_confirmation


@pytest.mark.asyncio
async def test_a_plugin_skill_marked_destructive_is_gated():
    """A dropped-in plugin gets no privileges a built-in skill doesn't have."""

    class _Plugin(_Landmine):
        name = "plugin_wipe"
        patterns = [re.compile(r"\bplugin wipe\b", re.IGNORECASE)]

    plugin = _Plugin()
    router = _router(plugin)
    await router.route("plugin wipe")
    assert plugin.fired == 0


@pytest.mark.asyncio
async def test_a_connector_action_marked_destructive_is_gated():
    from software.registry import ConnectorActionSkill

    calls: list[str] = []

    class _Action:
        name = "send_message"
        description = "send the message"
        capabilities = frozenset()
        requires_confirmation = True
        confirm_prompt = "Send it?"
        phrases = [r"\bsend the message\b"]

    class _Connector:
        app_id = "chat"
        display_name = "Chat"

        def detect(self):
            calls.append("detect")
            return type("D", (), {"installed": True})()

        def is_running(self):
            calls.append("is_running")
            return True

    skill = ConnectorActionSkill(_Connector(), _Action())
    router = _router(skill)
    result = await router.route("send the message")
    assert "is_running" not in calls        # never got as far as acting
    assert result.speech == "Send it?"
    assert router.awaiting_confirmation


@pytest.mark.asyncio
async def test_a_paired_device_capability_is_gated():
    from link.registry import DeviceCapabilitySkill

    dispatched: list[str] = []

    class _Link:
        async def dispatch(self, device_id, name, args):
            dispatched.append(name)
            return SkillResult("sent")

    skill = DeviceCapabilitySkill(
        _Link(), "phone1", "the phone",
        {"name": "factory_reset", "description": "wipe the phone",
         "requires_confirmation": True, "fast_patterns": [r"\breset my phone\b"]})
    router = _router(skill)
    await router.route("reset my phone")
    assert dispatched == []
    assert router.awaiting_confirmation


# --- the other callers: the companion API, routines --------------------------


@pytest.mark.asyncio
async def test_a_remote_ask_does_not_fire_it():
    mine = _Landmine()
    router = _router(mine)
    await router.route("erase the disk", context={"source": "remote"})
    assert mine.fired == 0


@pytest.mark.asyncio
async def test_a_remote_yes_cannot_answer_a_voice_confirmation():
    """Channel isolation: only the channel that armed it may confirm it."""
    mine = _Landmine()
    router = _router(mine)
    await router.route("erase the disk", context={"source": "voice"})
    await router.route("yes", context={"source": "remote"})
    assert mine.fired == 0


@pytest.mark.parametrize("source", sorted(NON_INTERACTIVE_SOURCES))
@pytest.mark.asyncio
async def test_an_unattended_source_is_refused_not_queued(source):
    """Nobody is listening on a timer, so there is no yes to wait for."""
    mine = _Landmine()
    router = _router(mine)
    result = await router.route("erase the disk", context={"source": source})
    assert mine.fired == 0
    assert not router.awaiting_confirmation      # nothing left armed
    assert "confirm" in result.speech.lower()


@pytest.mark.asyncio
async def test_a_routine_cannot_talk_itself_through_the_prompt():
    """A routine of ["erase the disk", "yes"] must not add up to permission."""
    from core.routines import RoutineScheduler

    mine = _Landmine()
    router = _router(mine)
    said: list[str] = []

    async def announce(text):
        said.append(text)

    routine = type("R", (), {"name": "cleanup", "enabled": True,
                             "at": "03:00", "days": [],
                             "ask": ["erase the disk", "yes"]})()
    await RoutineScheduler([routine], router, announce).fire(routine)
    assert mine.fired == 0


@pytest.mark.asyncio
async def test_an_unattended_source_is_refused_for_a_tool_call_too():
    mine = _Landmine()
    router = _router(mine, llm=_ToolCallingLLM("landmine"))
    await router.route("clean up", context={"source": "routine"})
    assert mine.fired == 0
    assert not router.awaiting_confirmation


# --- the escape hatches, and what they do and don't turn off -----------------


@pytest.mark.asyncio
async def test_turning_confirmations_off_runs_it_deliberately():
    """``safety.confirm_destructive: false`` means "act, don't ask" — not
    "ask into the void". This is the documented behaviour, asserted so it can't
    silently become a dead end again (it was one, once)."""
    mine = _Landmine()
    router = _router(mine)
    router._settings.safety.confirm_destructive = False
    await router.route("erase the disk")
    assert mine.confirmed_when_fired == [True]


@pytest.mark.asyncio
async def test_pc_control_off_beats_confirmation_entirely():
    """A PC-controlling destructive skill is refused before it is even offered."""

    class _PcMine(_Landmine):
        name = "pc_landmine"
        patterns = [re.compile(r"\bformat my drive\b", re.IGNORECASE)]
        controls_pc = True

    mine = _PcMine()
    router = _router(mine)
    router._settings.safety.pc_control_enabled = False
    await router.route("format my drive")
    assert mine.fired == 0
    assert not router.awaiting_confirmation


@pytest.mark.asyncio
async def test_pc_control_switched_off_mid_confirmation_still_refuses():
    class _PcMine(_Landmine):
        name = "pc_landmine"
        patterns = [re.compile(r"\bformat my drive\b", re.IGNORECASE)]
        controls_pc = True

    mine = _PcMine()
    router = _router(mine)
    await router.route("format my drive")
    router._settings.safety.pc_control_enabled = False
    await router.route("yes")
    assert mine.fired == 0


# --- the invariant over the LIVE registry ------------------------------------


def _live_registry():
    """Every skill MEDO actually ships, wired exactly as main.py wires it."""
    from main import Announcer, build_registry

    return build_registry(load_settings(), Announcer())


def test_every_shipped_destructive_skill_answers_the_confirmation_contract():
    """Declaring ``requires_confirmation`` obliges a skill to say what it will
    ask. The default prompt is serviceable but generic ("Do you want me to
    operate screen?"); a shipped skill should name the thing it is about to do.
    """
    registry = _live_registry()
    destructive = [s for s in registry.all()
                   if getattr(s, "requires_confirmation", False)]
    assert destructive, "the live registry should contain destructive skills"
    for skill in destructive:
        request = SkillRequest(text="do the thing", context={})
        prompt = skill.confirmation_prompt(request)
        assert isinstance(prompt, str) and prompt.strip(), skill.name
        assert asyncio.iscoroutinefunction(skill.preflight), skill.name


def test_no_shipped_destructive_skill_is_reachable_by_meaning():
    """The semantic tier has no confirmation step, so it must not index one."""
    registry = _live_registry()
    for skill in registry.all():
        if getattr(skill, "requires_confirmation", False):
            assert Router._semantic_safe(skill) is False, skill.name


def test_no_shipped_pc_controlling_skill_is_reachable_by_meaning():
    registry = _live_registry()
    for skill in registry.all():
        if getattr(skill, "controls_pc", False):
            assert Router._semantic_safe(skill) is False, skill.name


# --- a preflight that ASKS ----------------------------------------------------


class _NeedsATarget(Skill):
    """Destructive, and it cannot even ask "are you sure?" until it knows what.

    The realistic shape of a preflight question: the browser agent's "what
    should I do on the page?" is the same thing.
    """

    name = "shred"
    description = "shred a document"
    patterns = [re.compile(r"\bshred\b", re.IGNORECASE)]
    requires_confirmation = True

    def __init__(self) -> None:
        self.fired = 0

    def _target(self, request: SkillRequest) -> str:
        if request.args.get("target"):
            return str(request.args["target"]).strip()
        text = request.text.strip()
        _, marker, tail = text.partition("shred")
        return (tail if marker else text).strip()

    async def preflight(self, request: SkillRequest) -> SkillResult | None:
        if not self._target(request):
            return SkillResult("What should I shred?", await_reply=True,
                               success=False)
        return None

    def confirmation_prompt(self, request: SkillRequest) -> str:
        return f"Shall I shred {self._target(request)}?"

    async def execute(self, request: SkillRequest) -> SkillResult:
        self.fired += 1
        return SkillResult("Shredded.")


@pytest.mark.asyncio
async def test_a_preflight_question_captures_its_answer():
    """A question asked BEFORE the confirmation still gets its reply.

    _run_skill arms the reply capture when execute() asks something; the
    preflight hook is the same question asked one step earlier, and dropping
    the flag there routed the answer as a fresh request — losing both it and
    the request it belonged to.
    """
    skill = _NeedsATarget()
    router = _router(skill)
    first = await router.route("shred")
    assert "What should I shred" in first.speech
    second = await router.route("the old invoices")
    assert skill.fired == 0                       # still nothing destroyed
    assert router.awaiting_confirmation           # NOW it asks
    assert "old invoices" in second.speech


@pytest.mark.asyncio
async def test_the_answer_to_a_preflight_question_still_needs_a_yes():
    skill = _NeedsATarget()
    router = _router(skill)
    await router.route("shred")
    await router.route("the old invoices")
    assert skill.fired == 0
    await router.route("yes")
    assert skill.fired == 1
