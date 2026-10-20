"""P3-1: MEDO's skills over MCP, and everything that must NOT go over it.

The feature is small; the filter is the feature. Whatever connects to this
server is software the owner did not write, driven by a model whose prompt they
cannot see, so almost every test here is about something being withheld.

The auditing test at the bottom is the important one: it runs the real filter
over the real registry and fails if a future skill lands on the exposed side
without anyone deciding it should.
"""

from __future__ import annotations

import pytest

from core.config import load_settings
from mcp_servers.skills_server import (
    NEVER_EXPOSE,
    TOOL_PREFIX,
    call_skill,
    exposed_skills,
    is_read_only,
    tool_definition,
)
from security.capabilities import Capability
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult


class _Answers(Skill):
    name = "answers"
    description = "answer a question"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(self, request: SkillRequest) -> SkillResult:
        self.calls.append(dict(request.args or {}))
        return SkillResult("Forty-two.")


class _Acts(Skill):
    name = "acts"
    description = "do something to the machine"
    controls_pc = True

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("Done.")


class _Confirms(Skill):
    name = "confirms"
    description = "needs a yes"
    requires_confirmation = True

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("Sure?", needs_confirmation=True)


class _WritesFiles(Skill):
    name = "writes"
    description = "writes a file"
    capabilities = frozenset({Capability.WRITE_FILES})

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("Written.")


def _registry(*skills) -> SkillRegistry:
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


# --- the filter ---------------------------------------------------------------


def test_an_answering_skill_is_exposed():
    assert is_read_only(_Answers())


def test_a_pc_controlling_skill_is_not():
    assert not is_read_only(_Acts())


def test_a_skill_that_needs_confirmation_is_not():
    """A prompt down a pipe would be answered by the caller, not the owner."""
    assert not is_read_only(_Confirms())


def test_a_skill_that_writes_files_is_not():
    assert not is_read_only(_WritesFiles())


def test_the_never_list_wins_over_the_capability_check():
    class _Named(_Answers):
        name = "quit"

    assert not is_read_only(_Named())


def test_only_the_read_only_skills_are_offered():
    registry = _registry(_Answers(), _Acts(), _Confirms(), _WritesFiles())
    assert [s.name for s in exposed_skills(registry)] == ["answers"]


def test_a_skill_can_be_exposed_by_name():
    registry = _registry(_Answers(), _Acts())
    names = [s.name for s in exposed_skills(registry, extra=["acts"])]
    assert names == ["answers", "acts"]


def test_allow_actuation_drops_the_filter_entirely():
    registry = _registry(_Answers(), _Acts(), _Confirms())
    assert len(exposed_skills(registry, allow_actuation=True)) == 3


# --- the tool definitions -----------------------------------------------------


def test_a_tool_is_named_after_its_skill_with_a_prefix():
    definition = tool_definition(_Answers())
    assert definition["name"] == f"{TOOL_PREFIX}answers"
    assert definition["description"] == "answer a question"
    assert definition["inputSchema"]["type"] == "object"


def test_a_skill_with_no_parameters_still_gets_a_schema():
    assert tool_definition(_Answers())["inputSchema"]["properties"] == {}


# --- calling ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_call_reaches_the_skill():
    answers = _Answers()
    registry = _registry(answers)
    text = await call_skill(registry, "medo_answers", {"q": "life"},
                            allowed={"answers"})
    assert text == "Forty-two."
    assert answers.calls == [{"q": "life"}]


@pytest.mark.asyncio
async def test_the_prefix_is_optional_on_the_way_in():
    registry = _registry(_Answers())
    assert await call_skill(registry, "answers", {}, allowed={"answers"}) \
        == "Forty-two."


@pytest.mark.asyncio
async def test_a_skill_that_was_never_offered_cannot_be_called():
    """Not advertising something is not access control; the check is re-run."""
    registry = _registry(_Answers(), _Acts())
    text = await call_skill(registry, "medo_acts", {}, allowed={"answers"})
    assert "does not expose" in text


@pytest.mark.asyncio
async def test_an_unknown_tool_is_reported_not_raised():
    registry = _registry(_Answers())
    assert "no skill called" in await call_skill(registry, "medo_nope", {})


@pytest.mark.asyncio
async def test_a_raising_skill_does_not_kill_the_server():
    class _Boom(Skill):
        name = "boom"
        description = "raises"

        async def execute(self, request):
            raise RuntimeError("bang")

    registry = _registry(_Boom())
    assert "failed" in await call_skill(registry, "medo_boom", {})


@pytest.mark.asyncio
async def test_a_confirmation_is_refused_rather_than_auto_answered():
    registry = _registry(_Confirms())
    text = await call_skill(registry, "medo_confirms", {}, allowed={"confirms"})
    assert "confirm it out loud" in text


@pytest.mark.asyncio
async def test_a_call_is_marked_untrusted():
    """Whatever comes back is content from outside, not an instruction."""
    seen = {}

    class _Ctx(Skill):
        name = "ctx"
        description = "reads context"

        async def execute(self, request):
            seen.update(request.context or {})
            return SkillResult("ok")

    await call_skill(_registry(_Ctx()), "medo_ctx", {}, allowed={"ctx"})
    assert seen["source"] == "mcp"
    assert seen["untrusted"] is True


# --- the audit over the LIVE registry -----------------------------------------


def _live_exposed():
    from main import Announcer, build_registry

    settings = load_settings()
    registry = build_registry(settings, Announcer())
    return registry, {s.name for s in exposed_skills(registry)}


def test_nothing_that_acts_on_the_machine_is_exposed():
    registry, offered = _live_exposed()
    for skill in registry.all():
        if skill.name in offered:
            assert not getattr(skill, "controls_pc", False), skill.name
            assert not getattr(skill, "requires_confirmation", False), skill.name


def test_the_exposed_set_is_the_one_that_was_reviewed():
    """A tripwire, not a style check.

    The capability flags answer "does this actuate?", which is not the same
    question as "is it safe to hand to another program" — auditing the plain
    filter turned up skills that write to the owner's memory, read the webcam,
    and enumerate listening ports, all of them technically read-only. This test
    pins the reviewed answer, so a NEW skill that lands on the exposed side has
    to be looked at by a person rather than appearing there quietly.
    """
    _registry_, offered = _live_exposed()
    reviewed = {
        "datetime", "resistor_colors", "convene_council", "ask_specialist",
        "list_council", "circuit_help", "locate_app", "system_info", "dice",
        "weather", "news", "web_fetch", "web_search", "smalltalk",
    }
    assert offered == reviewed, (
        "the set of skills exposed over MCP changed. Anything newly listed here "
        "is now callable by other software on this machine — check it reveals "
        "nothing private and changes nothing, then update `reviewed`.")


@pytest.mark.parametrize("name", sorted({
    "remember_fact", "forget_fact", "notes", "obsidian_note", "obsidian_search",
    "see_screen", "see_camera", "what_is_this", "recall_conversation",
    "recall_facts", "repeat", "timers", "quit", "self_dev", "make_app",
}))
def test_the_sensitive_skills_stay_withheld(name):
    """Named individually so a regression says WHICH one came unstuck."""
    _registry_, offered = _live_exposed()
    assert name not in offered


def test_every_never_expose_entry_names_a_real_skill():
    """A typo in the deny list is a silent hole, not an error.

    Names are checked against every skill CLASS in ``skills/``, not against a
    built registry: several of the entries are registered conditionally (lion
    mode, a config flag) and are absent from a default build — which is exactly
    the build where a typo would go unnoticed. This test found three real ones
    ("modes" and "dictate" are called `session_mode` and `dictation`; "security"
    is not a skill at all).
    """
    import importlib
    import pkgutil

    import plugins as plugins_pkg
    import skills as skills_pkg
    from skills.base import Skill as _Skill

    live: set[str] = set()
    # plugins/ as well as skills/: the Obsidian skills ship as a plugin, and a
    # deny-list entry for one has to be checkable too.
    packages = [("skills", skills_pkg), ("plugins", plugins_pkg)]
    modules = [(name, m.name) for name, pkg in packages
               for m in pkgutil.iter_modules(pkg.__path__)]
    for package, module in modules:
        try:
            mod = importlib.import_module(f"{package}.{module}")
        except Exception:                      # optional dependency missing
            continue
        for value in vars(mod).values():
            if isinstance(value, type) and issubclass(value, _Skill):
                name = getattr(value, "name", "")
                if isinstance(name, str) and name:
                    live.add(name)
    stale = {name for name in NEVER_EXPOSE if name not in live}
    assert stale == set(), f"deny list names skills that don't exist: {stale}"
