"""P2-3: chaining skills in YAML, and the things a routine is not allowed to do.

A routine becomes a real :class:`~skills.base.Skill`, which is the whole point —
it inherits the fast path, the tool schema and the policy gate rather than
reimplementing them. That also means a routine could, if nothing stopped it,
become a way to reach a destructive skill without the confirmation that
protects it. Stopping that is the most important thing tested here.
"""

from __future__ import annotations

import re

import pytest

from core.composer import (
    MAX_STEPS,
    Routine,
    RoutineError,
    RoutineSkill,
    Step,
    load_routine_files,
    parse_routine,
    register_routines,
    render,
    render_args,
    validate_steps,
)
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult


class _Echo(Skill):
    name = "echo"
    description = "say something back"

    def __init__(self, name: str = "echo", reply: str = "echoed") -> None:
        self.name = name
        self._reply = reply
        self.calls: list[dict] = []

    async def execute(self, request: SkillRequest) -> SkillResult:
        self.calls.append(dict(request.args or {}))
        return SkillResult(self._reply)


class _Fails(Skill):
    name = "flaky"
    description = "always fails"

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("no good", success=False)


class _Explodes(Skill):
    name = "explodes"
    description = "raises"

    async def execute(self, request: SkillRequest) -> SkillResult:
        raise RuntimeError("boom")


class _Destructive(Skill):
    name = "wipe"
    description = "delete everything"
    requires_confirmation = True

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("Are you sure?", needs_confirmation=True)


def _registry(*skills: Skill) -> SkillRegistry:
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


# --- templating ---------------------------------------------------------------


def test_a_placeholder_is_substituted():
    assert render("the weather in {city}", {"city": "Skopje"}) == \
        "the weather in Skopje"


def test_an_unknown_placeholder_is_left_visible():
    """Blanking it would send the weather skill a request for nowhere."""
    assert render("in {city}", {}) == "in {city}"


def test_templating_does_not_evaluate_anything():
    """A config file that can compute can be made to compute something else."""
    assert render("{__import__}", {}) == "{__import__}"
    assert render("{1+1}", {}) == "{1+1}"


def test_a_none_value_renders_empty():
    assert render("x{y}z", {"y": None}) == "xz"


def test_args_are_rendered_including_nested_ones():
    out = render_args({"city": "{city}", "n": 3,
                       "deep": {"who": "{who}"}, "list": ["{who}", 1]},
                      {"city": "Skopje", "who": "Matej"})
    assert out == {"city": "Skopje", "n": 3,
                   "deep": {"who": "Matej"}, "list": ["Matej", 1]}


# --- parsing ------------------------------------------------------------------


def _yaml(**over) -> dict:
    data = {"name": "morning briefing", "steps": [{"skill": "echo"}]}
    data.update(over)
    return data


def test_a_minimal_routine_parses():
    routine = parse_routine(_yaml())
    assert routine.name == "morning briefing"
    assert routine.skill_name == "routine_morning_briefing"
    assert len(routine.steps) == 1


def test_a_bare_string_step_is_a_skill_name():
    routine = parse_routine(_yaml(steps=["weather", "news"]))
    assert [s.skill for s in routine.steps] == ["weather", "news"]


def test_a_routine_without_a_name_is_refused():
    with pytest.raises(RoutineError, match="needs a 'name'"):
        parse_routine({"steps": ["echo"]})


def test_a_routine_without_steps_is_refused():
    with pytest.raises(RoutineError, match="non-empty 'steps'"):
        parse_routine({"name": "empty"})


def test_a_step_that_is_neither_skill_nor_say_is_refused():
    with pytest.raises(RoutineError, match="needs either 'skill' or 'say'"):
        parse_routine(_yaml(steps=[{"into": "x"}]))


def test_too_many_steps_are_refused():
    with pytest.raises(RoutineError, match="past the"):
        parse_routine(_yaml(steps=["echo"] * (MAX_STEPS + 1)))


def test_a_bad_trigger_time_is_refused():
    with pytest.raises(RoutineError, match="HH:MM"):
        parse_routine(_yaml(trigger={"at": "8am"}))


def test_a_single_phrase_may_be_a_bare_string():
    routine = parse_routine(_yaml(trigger={"phrase": "good morning"}))
    assert routine.trigger.phrases == ("good morning",)


def test_days_are_normalised_to_three_letters():
    routine = parse_routine(_yaml(trigger={"at": "07:30",
                                           "days": ["Monday", "TUE"]}))
    assert routine.trigger.days == ("mon", "tue")


def test_the_error_names_the_file_it_came_from():
    with pytest.raises(RoutineError, match="broken.yaml"):
        parse_routine({"steps": []}, source="broken.yaml")


# --- loading from disk --------------------------------------------------------


def test_routine_files_are_loaded(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "name: one\nsteps:\n  - skill: echo\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text(
        "name: two\nsteps:\n  - say: hello\n", encoding="utf-8")
    routines, errors = load_routine_files(tmp_path)
    assert {r.name for r in routines} == {"one", "two"}
    assert errors == []


def test_one_broken_file_does_not_stop_the_others(tmp_path):
    (tmp_path / "good.yaml").write_text(
        "name: good\nsteps:\n  - skill: echo\n", encoding="utf-8")
    (tmp_path / "bad.yaml").write_text("name: bad\n", encoding="utf-8")
    routines, errors = load_routine_files(tmp_path)
    assert [r.name for r in routines] == ["good"]
    assert len(errors) == 1 and "bad" in errors[0]


def test_unparseable_yaml_is_reported_not_raised(tmp_path):
    (tmp_path / "bad.yaml").write_text("name: [unclosed\n", encoding="utf-8")
    routines, errors = load_routine_files(tmp_path)
    assert routines == [] and len(errors) == 1


def test_a_missing_directory_is_not_an_error(tmp_path):
    assert load_routine_files(tmp_path / "nope") == ([], [])


def test_an_empty_file_is_skipped(tmp_path):
    (tmp_path / "empty.yaml").write_text("\n", encoding="utf-8")
    assert load_routine_files(tmp_path) == ([], [])


# --- registration and safety --------------------------------------------------


def test_a_routine_registers_as_an_ordinary_skill():
    registry = _registry(_Echo())
    routine = parse_routine(_yaml(trigger={"phrase": ["good morning"]}))
    count, problems, _warnings = register_routines(registry, [routine])
    assert (count, problems) == (1, [])
    skill = registry.get("routine_morning_briefing")
    assert skill is not None
    assert skill.match("good morning") is not None
    assert skill.tool_schema()["function"]["name"] == "routine_morning_briefing"


def test_a_routine_naming_a_skill_that_does_not_exist_is_refused():
    registry = _registry(_Echo())
    routine = parse_routine(_yaml(steps=[{"skill": "teleport"}]))
    count, problems, _warnings = register_routines(registry, [routine])
    assert count == 0
    assert "no skill called 'teleport'" in problems[0]


def test_a_routine_cannot_call_a_skill_that_needs_confirmation():
    """The headline safety rule: routines compose capability, never grant it."""
    registry = _registry(_Echo(), _Destructive())
    routine = parse_routine(_yaml(steps=[{"skill": "echo"}, {"skill": "wipe"}]))
    count, problems, _warnings = register_routines(registry, [routine])
    assert count == 0
    assert "step 2" in problems[0] and "wipe" in problems[0]
    assert registry.get("routine_morning_briefing") is None


def test_the_shipped_example_routines_are_all_valid():
    """They are documentation; documentation that does not load is worse than none."""
    from core.config import PROJECT_ROOT, load_settings
    from main import Announcer, build_registry

    routines, errors = load_routine_files(PROJECT_ROOT / "routines")
    assert errors == []
    assert len(routines) >= 2
    registry = build_registry(load_settings(), Announcer())
    for routine in routines:
        assert validate_steps(routine, registry) == [], routine.name


def test_a_duplicate_routine_name_is_refused():
    registry = _registry(_Echo())
    routine = parse_routine(_yaml())
    register_routines(registry, [routine])
    count, problems, _warnings = register_routines(registry, [routine])
    assert count == 0 and "already registered" in problems[0]


def test_phrases_are_matched_literally_not_as_regex():
    """A config file must not be able to inject a pattern into the hot path."""
    registry = _registry(_Echo())
    routine = parse_routine(_yaml(trigger={"phrase": ["what's up?"]}))
    register_routines(registry, [routine])
    skill = registry.get("routine_morning_briefing")
    assert skill.match("what's up?") is not None
    assert skill.match("whats up") is None       # the ? was not a quantifier


def test_a_catastrophic_pattern_in_a_phrase_is_inert():
    registry = _registry(_Echo())
    routine = parse_routine(_yaml(trigger={"phrase": ["(a+)+$"]}))
    register_routines(registry, [routine])
    skill = registry.get("routine_morning_briefing")
    assert skill.match("aaaaaaaaaaaaaaaaaaaaaaaaaaaa!") is None
    assert skill.match("(a+)+$") is not None


# --- running ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_routine_runs_its_steps_in_order():
    first, second = _Echo("first", "one"), _Echo("second", "two")
    registry = _registry(first, second)
    routine = parse_routine(_yaml(steps=[{"skill": "first"}, {"skill": "second"}]))
    skill = RoutineSkill(routine, registry)
    result = await skill.execute(SkillRequest(text="go"))
    assert result.speech == "one two"
    assert result.success


@pytest.mark.asyncio
async def test_a_say_step_speaks_a_literal_line():
    registry = _registry(_Echo())
    routine = parse_routine(_yaml(steps=[{"say": "Good morning."},
                                         {"skill": "echo"}]))
    result = await RoutineSkill(routine, registry).execute(SkillRequest(text="go"))
    assert result.speech == "Good morning. echoed"


@pytest.mark.asyncio
async def test_params_reach_a_step_as_arguments():
    echo = _Echo()
    registry = _registry(echo)
    routine = parse_routine(_yaml(
        params={"city": "Skopje"},
        steps=[{"skill": "echo", "args": {"city": "{city}"}}]))
    await RoutineSkill(routine, registry).execute(SkillRequest(text="go"))
    assert echo.calls == [{"city": "Skopje"}]


@pytest.mark.asyncio
async def test_a_tool_argument_overrides_a_param_default():
    echo = _Echo()
    registry = _registry(echo)
    routine = parse_routine(_yaml(
        params={"city": "Skopje"},
        steps=[{"skill": "echo", "args": {"city": "{city}"}}]))
    await RoutineSkill(routine, registry).execute(
        SkillRequest(text="go", args={"city": "Bitola"}))
    assert echo.calls == [{"city": "Bitola"}]


@pytest.mark.asyncio
async def test_an_undeclared_tool_argument_is_ignored():
    """The model may only fill in parameters the routine actually declares."""
    echo = _Echo()
    registry = _registry(echo)
    routine = parse_routine(_yaml(
        params={"city": "Skopje"},
        steps=[{"skill": "echo", "args": {"city": "{city}", "evil": "{evil}"}}]))
    await RoutineSkill(routine, registry).execute(
        SkillRequest(text="go", args={"evil": "rm -rf"}))
    assert echo.calls == [{"city": "Skopje", "evil": "{evil}"}]


@pytest.mark.asyncio
async def test_a_step_can_feed_the_next_one():
    registry = _registry(_Echo("first", "sunny"), _Echo("second", "ok"))
    routine = parse_routine(_yaml(steps=[
        {"skill": "first", "into": "forecast"},
        {"say": "It is {forecast} out."}]))
    result = await RoutineSkill(routine, registry).execute(SkillRequest(text="go"))
    assert "It is sunny out." in result.speech


@pytest.mark.asyncio
async def test_previous_holds_the_last_step_speech():
    registry = _registry(_Echo("first", "sunny"))
    routine = parse_routine(_yaml(steps=[{"skill": "first"},
                                         {"say": "({previous})"}]))
    result = await RoutineSkill(routine, registry).execute(SkillRequest(text="go"))
    assert "(sunny)" in result.speech


@pytest.mark.asyncio
async def test_a_failing_step_does_not_stop_the_routine():
    registry = _registry(_Fails(), _Echo("after", "still here"))
    routine = parse_routine(_yaml(steps=[{"skill": "flaky"}, {"skill": "after"}]))
    result = await RoutineSkill(routine, registry).execute(SkillRequest(text="go"))
    assert "still here" in result.speech
    assert "couldn't do flaky" in result.speech      # and it says what it missed
    assert result.success is False


@pytest.mark.asyncio
async def test_a_raising_step_does_not_kill_the_routine():
    registry = _registry(_Explodes(), _Echo("after", "still here"))
    routine = parse_routine(_yaml(steps=[{"skill": "explodes"}, {"skill": "after"}]))
    result = await RoutineSkill(routine, registry).execute(SkillRequest(text="go"))
    assert "still here" in result.speech
    assert result.data["failed"] == ["explodes"]


@pytest.mark.asyncio
async def test_a_step_that_asks_to_confirm_is_refused_at_runtime():
    """Belt to validate_steps' braces, for a registry that changed underneath."""
    registry = _registry(_Destructive())
    routine = Routine(name="sneaky", description="",
                      steps=(Step(skill="wipe"),))
    result = await RoutineSkill(routine, registry).execute(SkillRequest(text="go"))
    assert result.success is False
    assert "wipe" in result.speech


@pytest.mark.asyncio
async def test_the_step_inherits_the_turn_context():
    """So an unattended routine's steps are still unattended."""
    seen = {}

    class _Ctx(Skill):
        name = "ctx"
        description = "reads context"

        async def execute(self, request):
            seen.update(request.context or {})
            return SkillResult("ok")

    registry = _registry(_Ctx())
    routine = parse_routine(_yaml(steps=[{"skill": "ctx"}]))
    await RoutineSkill(routine, registry).execute(
        SkillRequest(text="go", context={"source": "routine", "language": "mk"}))
    assert seen["source"] == "routine"
    assert seen["language"] == "mk"


@pytest.mark.asyncio
async def test_a_routine_is_reachable_through_the_router():
    """End to end: the fast path routes a trigger phrase to the routine."""
    from core.config import load_settings
    from core.events import EventBus
    from core.router import Router

    echo = _Echo()
    registry = _registry(echo)
    routine = parse_routine(_yaml(trigger={"phrase": ["good morning"]}))
    register_routines(registry, [routine])
    router = Router(load_settings(), registry, None, EventBus())
    router.model = None
    result = await router.route("good morning")
    assert result.skill_name == "routine_morning_briefing"
    assert echo.calls == [{}]


def test_the_routine_skill_name_is_a_valid_tool_name():
    """Tool names reach the model's function-calling schema; keep them plain."""
    routine = parse_routine(_yaml(name="Morning Briefing! (v2)"))
    assert re.fullmatch(r"[a-z0-9_]+", routine.skill_name)


# --- a routine inherits its steps' privileges --------------------------------
#
# Regression: RoutineSkill declared `controls_pc = False` — true of the wrapper,
# false of what it runs. With the PC-control master switch OFF, calling
# `open_website` directly was refused and the identical call inside a routine
# went through. A config file could silently re-enable actuation.

class _Actuates(Skill):
    name = "open_website"
    description = "open a website"
    controls_pc = True

    def __init__(self) -> None:
        self.ran = 0

    async def execute(self, request: SkillRequest) -> SkillResult:
        self.ran += 1
        return SkillResult("Opened.")


def _routine_skill(steps, *skills):
    registry = _registry(*skills)
    routine = parse_routine({"name": "r", "trigger": {"phrase": ["good morning"]},
                             "steps": steps})
    register_routines(registry, [routine])
    return registry, registry.get("routine_r")


def test_a_routine_declares_its_steps_actuation():
    _reg, skill = _routine_skill([{"skill": "open_website"}], _Actuates())
    assert skill.controls_pc is True


def test_a_harmless_routine_declares_nothing():
    _reg, skill = _routine_skill([{"skill": "echo"}, {"say": "hi"}], _Echo())
    assert skill.controls_pc is False


def test_one_actuating_step_taints_the_whole_routine():
    _reg, skill = _routine_skill(
        [{"skill": "echo"}, {"skill": "open_website"}], _Echo(), _Actuates())
    assert skill.controls_pc is True


def test_a_routine_inherits_its_steps_capabilities():
    from security.capabilities import ACTUATION_CAPS

    _reg, skill = _routine_skill([{"skill": "open_website"}], _Actuates())
    assert skill.capabilities & ACTUATION_CAPS


@pytest.mark.asyncio
async def test_a_routine_cannot_slip_past_the_pc_control_switch():
    """The headline: the switch means the same thing however the skill is reached."""
    from core.config import load_settings
    from core.events import EventBus
    from core.router import Router

    actuator = _Actuates()
    registry, _skill = _routine_skill([{"skill": "open_website"}], actuator)
    settings = load_settings()
    settings.safety.pc_control_enabled = False
    router = Router(settings, registry, None, EventBus())
    router.model = None

    result = await router.route("good morning")
    assert actuator.ran == 0
    assert "PC control" in result.speech


@pytest.mark.asyncio
async def test_a_routine_still_runs_when_the_switch_is_on():
    from core.config import load_settings
    from core.events import EventBus
    from core.router import Router

    actuator = _Actuates()
    registry, _skill = _routine_skill([{"skill": "open_website"}], actuator)
    settings = load_settings()
    settings.safety.pc_control_enabled = True
    router = Router(settings, registry, None, EventBus())
    router.model = None

    await router.route("good morning")
    assert actuator.ran == 1


@pytest.mark.asyncio
async def test_a_harmless_routine_is_not_over_gated():
    """Inheriting privileges must not gate routines that need none."""
    from core.config import load_settings
    from core.events import EventBus
    from core.router import Router

    echo = _Echo()
    registry, _skill = _routine_skill([{"skill": "echo"}], echo)
    settings = load_settings()
    settings.safety.pc_control_enabled = False
    router = Router(settings, registry, None, EventBus())
    router.model = None

    result = await router.route("good morning")
    assert "PC control" not in result.speech
    assert echo.calls == [{}]


# --- a trigger phrase another skill already owns -----------------------------
#
# Regression: registration order sets fast-path precedence and routines register
# LAST, so a phrase an existing skill matches never reaches the routine. Both
# shipped examples hit it — "focus session" went to the window-focus skill and
# answered "I couldn't find session", "добро утро" went to the briefing. The
# routine looked simply broken, with nothing anywhere saying why.

def test_a_shadowed_phrase_is_reported():
    from core.composer import phrase_conflicts

    # `echo` too: the routine's default step calls it, and a missing step skill
    # would be REFUSED before the phrase check this test is about.
    registry = _registry(_Echo(), _Echo("clock", "Noon."))
    registry.get("clock").patterns = [re.compile(r"\bgood morning\b", re.I)]
    routine = parse_routine(_yaml(trigger={"phrase": ["good morning"]}))
    clashes = phrase_conflicts(routine, registry)
    assert len(clashes) == 1
    assert "'clock'" in clashes[0] and "good morning" in clashes[0]


def test_a_free_phrase_is_not_reported():
    from core.composer import phrase_conflicts

    registry = _registry(_Echo())
    routine = parse_routine(_yaml(trigger={"phrase": ["utterly unique phrase"]}))
    assert phrase_conflicts(routine, registry) == []


def test_a_shadowed_phrase_warns_but_still_registers():
    """The routine keeps its other phrases and its LLM tool — warn, don't refuse."""
    # `echo` too: the routine's default step calls it, and a missing step skill
    # would be REFUSED before the phrase check this test is about.
    registry = _registry(_Echo(), _Echo("clock", "Noon."))
    registry.get("clock").patterns = [re.compile(r"\bgood morning\b", re.I)]
    routine = parse_routine(_yaml(
        trigger={"phrase": ["good morning", "utterly unique phrase"]}))
    count, refused, warnings = register_routines(registry, [routine])
    assert (count, refused) == (1, [])
    assert len(warnings) == 1 and "good morning" in warnings[0]
    assert registry.get("routine_morning_briefing") is not None


def test_the_conflict_check_does_not_flag_the_routine_against_itself():
    registry = _registry(_Echo())
    routine = parse_routine(_yaml(trigger={"phrase": ["utterly unique phrase"]}))
    register_routines(registry, [routine])
    from core.composer import phrase_conflicts

    assert phrase_conflicts(routine, registry) == []


def test_the_shipped_examples_have_no_shadowed_phrases():
    """They are documentation; one that silently does the wrong thing is worse
    than none. Checked against the registry main.py actually builds."""
    from core.composer import phrase_conflicts
    from core.config import PROJECT_ROOT, load_settings
    from main import Announcer, build_registry

    registry = build_registry(load_settings(), Announcer())
    routines, _errors = load_routine_files(PROJECT_ROOT / "routines")
    clashes = [c for r in routines for c in phrase_conflicts(r, registry)]
    assert clashes == [], clashes
