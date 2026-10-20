"""Cowork S1: a plan may only contain things MEDO can actually do.

The planner's job is not to be clever, it is to be honest. Every assertion here
is some form of the same property: what the MODEL said is a proposal, and what
the live registry says is the truth — about whether a capability exists, about
whether it is destructive, and about whether the policy engine will allow it.
"""

from __future__ import annotations

import pytest

from cowork.plan import PENDING, Plan, PlanLocked, Step, parse_edit
from cowork.planner import Planner, ability_for, catalogue, language_of
from security.capabilities import Capability
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult


class _Fake(Skill):
    """A skill with exactly the surface the planner reads."""

    def __init__(self, name, description="does a thing", *, params=None,
                 required=(), caps=frozenset(), confirm=False,
                 controls_pc=False, reply="done"):
        self.name = name
        self.description = description
        self.capabilities = frozenset(caps)
        self.requires_confirmation = confirm
        self.controls_pc = controls_pc
        self.patterns = []
        self._params = params or {}
        self._required = list(required)
        self._reply = reply

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult(self._reply)

    def tool_schema(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object",
                           "properties": {k: {"type": "string", "description": v}
                                          for k, v in self._params.items()},
                           "required": self._required}}}


def _registry(*skills) -> SkillRegistry:
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


def _brain(payload: str):
    async def think(prompt: str) -> str:
        think.prompt = prompt
        return payload
    think.prompt = ""
    return think


WEATHER = dict(name="weather", description="Current weather for a city",
               params={"city": "the city"}, required=["city"])
OBS = dict(name="obs_record", description="Start recording in OBS",
           caps={Capability.RUN_COMMAND}, confirm=True, controls_pc=True)
NOTES = dict(name="notes", description="Write a note")


# --- a task maps to REAL capabilities -----------------------------------------


@pytest.mark.asyncio
async def test_a_plan_binds_every_step_to_a_registered_capability():
    registry = _registry(_Fake(**WEATHER), _Fake(**NOTES))
    planner = Planner(registry, _brain(
        '{"steps": [{"description": "check the weather", "capability": "weather",'
        ' "args": {"city": "Skopje"}},'
        ' {"description": "note it down", "capability": "notes"}]}'))
    plan = await planner.plan("check the weather and note it down")
    assert [s.capability for s in plan.steps] == ["weather", "notes"]
    assert plan.steps[0].args == {"city": "Skopje"}
    assert plan.refusal == "" and all(s.status == PENDING for s in plan.steps)


@pytest.mark.asyncio
async def test_the_catalogue_is_the_live_registry_not_a_hardcoded_list():
    registry = _registry(_Fake(**WEATHER))
    assert [a.name for a in catalogue(registry)] == ["weather"]
    registry.register(_Fake(**NOTES))
    assert [a.name for a in catalogue(registry)] == ["weather", "notes"]


@pytest.mark.asyncio
async def test_cowork_and_quit_can_never_be_planned():
    """A plan that plans, or that closes the thing running the plan."""
    registry = _registry(_Fake("cowork"), _Fake("quit"), _Fake(**NOTES))
    assert [a.name for a in catalogue(registry)] == ["notes"]


# --- an impossible step is FLAGGED, not faked ---------------------------------


@pytest.mark.asyncio
async def test_a_step_with_no_capability_is_kept_and_explained():
    registry = _registry(_Fake(**NOTES))
    planner = Planner(registry, _brain(
        '{"steps": [{"description": "order a pizza", "capability": null},'
        ' {"description": "note it", "capability": "notes"}]}'))
    plan = await planner.plan("order a pizza and note it")
    assert len(plan.steps) == 2
    assert plan.steps[0].capability == "" and plan.steps[0].problem
    assert plan.steps[0].possible is False
    assert "order a pizza" in plan.summary()
    assert "no skill" in plan.summary().lower()


@pytest.mark.asyncio
async def test_an_invented_capability_is_not_swapped_for_a_real_one():
    """The dangerous failure: "delete_files" isn't registered, so the planner
    must NOT quietly bind the step to the nearest thing that is."""
    registry = _registry(_Fake("files", "Read files"), _Fake(**NOTES))
    planner = Planner(registry, _brain(
        '{"steps": [{"description": "delete the folder",'
        ' "capability": "delete_files"}]}'))
    plan = await planner.plan("delete the folder")
    assert plan.steps[0].capability == ""
    assert plan.steps[0].problem


@pytest.mark.asyncio
async def test_a_plan_of_only_impossible_steps_never_asks_to_go_ahead():
    registry = _registry(_Fake(**NOTES))
    planner = Planner(registry, _brain(
        '{"steps": [{"description": "book a flight", "capability": null}]}'))
    plan = await planner.plan("book a flight")
    assert "go ahead" not in plan.summary().lower()
    assert "can't do that" in plan.summary().lower()


# --- risk flags come from the SKILL, never from the model ---------------------


@pytest.mark.asyncio
async def test_side_effect_and_confirmation_are_read_from_the_registry():
    registry = _registry(_Fake(**OBS), _Fake(**WEATHER))
    planner = Planner(registry, _brain(
        '{"steps": [{"description": "start recording", "capability":'
        ' "obs_record", "side_effect": false, "requires_confirmation": false},'
        ' {"description": "weather", "capability": "weather",'
        ' "args": {"city": "Skopje"}, "side_effect": true}]}'))
    plan = await planner.plan("record and tell me the weather")
    recording, weather = plan.steps
    # The model said harmless. The skill says otherwise, and the skill wins.
    assert recording.side_effect and recording.requires_confirmation
    assert recording.high_impact                      # RUN_COMMAND
    # …and the reverse: the model called a sensing skill destructive.
    assert not weather.side_effect and not weather.requires_confirmation


def test_ability_reads_risk_from_declared_capabilities():
    assert ability_for(_Fake(**OBS)).side_effect is True
    assert ability_for(_Fake(**WEATHER)).side_effect is False
    # An un-migrated skill that only sets controls_pc is still a side effect.
    assert ability_for(_Fake("typer", controls_pc=True)).side_effect is True


# --- bounded ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_over_long_task_is_refused_not_truncated():
    registry = _registry(_Fake(**NOTES))
    steps = ", ".join('{"description": "step", "capability": "notes"}'
                      for _ in range(9))
    planner = Planner(registry, _brain(f'{{"steps": [{steps}]}}'), max_steps=6)
    plan = await planner.plan("do nine things")
    assert plan.steps == []
    assert "split" in plan.refusal.lower() and "9" in plan.refusal


@pytest.mark.asyncio
async def test_a_plan_at_the_cap_is_fine():
    registry = _registry(_Fake(**NOTES))
    steps = ", ".join('{"description": "step", "capability": "notes"}'
                      for _ in range(6))
    planner = Planner(registry, _brain(f'{{"steps": [{steps}]}}'), max_steps=6)
    plan = await planner.plan("do six things")
    assert len(plan.steps) == 6 and not plan.refusal


# --- one clarifying question --------------------------------------------------


@pytest.mark.asyncio
async def test_a_missing_required_argument_asks_exactly_one_question():
    registry = _registry(_Fake(**WEATHER),
                         _Fake("email", "Send an email",
                               params={"to": "recipient"}, required=["to"]))
    planner = Planner(registry, _brain(
        '{"steps": [{"description": "weather", "capability": "weather"},'
        ' {"description": "email it", "capability": "email"}]}'))
    plan = await planner.plan("check the weather and email it")
    assert plan.question                     # something was missing
    assert plan.question.count("?") == 1     # ONE question, not an interrogation


@pytest.mark.asyncio
async def test_the_models_own_question_wins_when_it_asked_one():
    registry = _registry(_Fake(**WEATHER))
    planner = Planner(registry, _brain(
        '{"steps": [{"description": "weather", "capability": "weather"}],'
        ' "question": "Which city?"}'))
    plan = await planner.plan("what's the weather")
    assert plan.question == "Which city?"


@pytest.mark.asyncio
async def test_no_question_when_nothing_is_missing():
    registry = _registry(_Fake(**WEATHER))
    planner = Planner(registry, _brain(
        '{"steps": [{"description": "weather", "capability": "weather",'
        ' "args": {"city": "Ohrid"}}]}'))
    plan = await planner.plan("weather in Ohrid")
    assert plan.question == ""


@pytest.mark.asyncio
async def test_an_invented_argument_name_is_dropped_not_passed_through():
    registry = _registry(_Fake(**WEATHER))
    planner = Planner(registry, _brain(
        '{"steps": [{"description": "weather", "capability": "weather",'
        ' "args": {"city": "Ohrid", "force": true, "rm": "-rf"}}]}'))
    plan = await planner.plan("weather in Ohrid")
    assert plan.steps[0].args == {"city": "Ohrid"}


# --- the policy engine has a say at planning time -----------------------------


class _DenyAll:
    class _D:
        reason = "PC control is switched off"
        code = "pc_control_off"

        def denied(self):
            return True
    def gate_skill(self, skill, **kw):
        return self._D()


@pytest.mark.asyncio
async def test_a_step_the_policy_denies_right_now_is_marked_impossible():
    """Better to say "PC control is off" while planning than to fail at step 3."""
    registry = _registry(_Fake(**OBS))
    planner = Planner(registry, _brain(
        '{"steps": [{"description": "record", "capability": "obs_record"}]}'),
        policy=_DenyAll())
    plan = await planner.plan("start recording")
    assert plan.steps[0].possible is False
    assert "PC control" in plan.steps[0].problem


# --- degrading honestly -------------------------------------------------------


@pytest.mark.asyncio
async def test_no_brain_means_no_plan_and_says_so():
    planner = Planner(_registry(_Fake(**NOTES)), None)
    plan = await planner.plan("do a thing")
    assert plan.steps == [] and "offline" in plan.refusal.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["", "I think we should start by...", "{}",
                                   '{"steps": []}', "null"])
async def test_an_unusable_model_reply_refuses_instead_of_inventing(reply):
    planner = Planner(_registry(_Fake(**NOTES)), _brain(reply))
    plan = await planner.plan("do a thing")
    assert plan.steps == [] and plan.refusal


@pytest.mark.asyncio
async def test_json_wrapped_in_prose_is_still_read():
    """Small local models narrate. The plan is still in there."""
    planner = Planner(_registry(_Fake(**NOTES)), _brain(
        'Sure! Here is the plan:\n{"steps": [{"description": "note",'
        ' "capability": "notes"}]}\nHope that helps.'))
    plan = await planner.plan("note something")
    assert [s.capability for s in plan.steps] == ["notes"]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected", [
    # a markdown fence, which the prompt forbids and models add anyway
    ('```json\n{"steps": [{"description": "d", "capability": "notes"}]}\n```',
     "notes"),
    # a draft, then the plan it settled on — the LAST one with steps wins
    ('{"steps": []}\nActually:\n{"steps": [{"description": "d",'
     ' "capability": "notes"}]}', "notes"),
    # a brace INSIDE a string: first-brace-to-last-brace parses as neither
    ('{"steps": [{"description": "say {hi}", "capability": "notes"}]}',
     "notes"),
])
async def test_the_plan_is_found_in_what_a_real_model_actually_writes(
        raw, expected):
    plan = await Planner(_registry(_Fake(**NOTES)), _brain(raw)).plan("do it")
    assert [s.capability for s in plan.steps] == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("language,named", [("mk", "Macedonian"),
                                            ("en", "English")])
async def test_the_prompt_names_the_language_it_must_write_in(language, named):
    """"in the user's language" is not an instruction a model can follow — it
    has no idea which that is. Asked in Macedonian, qwen3:30b read the plan back
    as "Report today's date"."""
    think = _brain('{"steps": [{"description": "d", "capability": "notes"}]}')
    await Planner(_registry(_Fake(**NOTES)), think).plan("do it",
                                                         language=language)
    assert f"written in {named}" in think.prompt


@pytest.mark.asyncio
async def test_a_brain_that_ran_out_of_time_says_that_and_not_something_else(
        monkeypatch):
    """Measured against qwen3:30b: a nine-part task never finished inside 300 s.
    "I couldn't work out a plan" is true and useless — the remedy is to split
    the task, and that sentence doesn't hint at it."""
    import cowork.planner as mod

    clock = iter([0.0, 301.0])
    monkeypatch.setattr(mod, "_now", lambda: next(clock))
    plan = await Planner(_registry(_Fake(**NOTES)), _brain("")).plan(
        "back up everything, email it, compress it, restart, then report")
    assert "too big" in plan.refusal and "splitting" in plan.refusal


@pytest.mark.asyncio
async def test_a_quick_empty_reply_is_not_reported_as_a_timeout(monkeypatch):
    """The brain answering nothing in two seconds is a different failure."""
    import cowork.planner as mod

    clock = iter([0.0, 2.0])
    monkeypatch.setattr(mod, "_now", lambda: next(clock))
    plan = await Planner(_registry(_Fake(**NOTES)), _brain("")).plan("do it")
    assert "too big" not in plan.refusal and plan.refusal


@pytest.mark.asyncio
async def test_a_slow_but_ANSWERED_plan_is_still_a_plan(monkeypatch):
    """Slow is not a failure. Two minutes with a good plan at the end is the
    normal case on this hardware."""
    import cowork.planner as mod

    clock = iter([0.0, 200.0])
    monkeypatch.setattr(mod, "_now", lambda: next(clock))
    plan = await Planner(_registry(_Fake(**NOTES)), _brain(
        '{"steps": [{"description": "note", "capability": "notes"}]}')).plan(
        "note it")
    assert plan.refusal == "" and [s.capability for s in plan.steps] == ["notes"]


@pytest.mark.asyncio
async def test_a_brain_that_raises_does_not_raise_out_of_planning():
    async def broken(prompt):
        raise RuntimeError("model on fire")

    plan = await Planner(_registry(_Fake(**NOTES)), broken).plan("do a thing")
    assert plan.steps == [] and plan.refusal


@pytest.mark.asyncio
async def test_an_empty_task_asks_what_to_do():
    plan = await Planner(_registry(_Fake(**NOTES)), _brain("{}")).plan("   ")
    assert plan.refusal and plan.steps == []


# --- Macedonian ---------------------------------------------------------------


def test_language_is_taken_from_the_task():
    assert language_of("prep my demo") == "en"
    assert language_of("подготви ја презентацијата") == "mk"


@pytest.mark.asyncio
async def test_a_macedonian_task_gets_a_macedonian_plan():
    registry = _registry(_Fake(**WEATHER))
    planner = Planner(registry, _brain(
        '{"steps": [{"description": "провери го времето", "capability":'
        ' "weather", "args": {"city": "Скопје"}}]}'))
    task = "кажи ми го времето во Скопје"
    plan = await planner.plan(task, language=language_of(task))
    summary = plan.summary()
    assert summary.startswith("Еве го планот")
    assert "Да продолжам?" in summary
    assert "Here's my plan" not in summary


@pytest.mark.asyncio
async def test_a_macedonian_refusal_is_macedonian():
    registry = _registry(_Fake(**NOTES))
    steps = ", ".join('{"description": "чекор", "capability": "notes"}'
                      for _ in range(9))
    planner = Planner(registry, _brain(f'{{"steps": [{steps}]}}'), max_steps=6)
    plan = await planner.plan("направи девет работи", language="mk")
    assert "Подели" in plan.refusal


@pytest.mark.asyncio
async def test_a_macedonian_impossible_step_is_explained_in_macedonian():
    planner = Planner(_registry(_Fake(**NOTES)), _brain(
        '{"steps": [{"description": "нарачај пица", "capability": null}]}'))
    plan = await planner.plan("нарачај пица", language="mk")
    assert "вештина" in plan.steps[0].problem


# --- the plan object itself ---------------------------------------------------


def test_the_summary_is_numbered_because_edits_are_by_number():
    plan = Plan(task="t", steps=[Step("open OBS", "obs"), Step("record", "rec")])
    assert "1. open OBS" in plan.summary() and "2. record" in plan.summary()


def test_an_approved_plan_cannot_gain_a_step():
    """The trust boundary, structurally: nothing seen mid-task can add work."""
    plan = Plan(task="t", steps=[Step("open OBS", "obs")])
    plan.lock()
    with pytest.raises(PlanLocked):
        plan.add_step(Step("delete the folder", "files"))
    with pytest.raises(PlanLocked):
        plan.remove_step(1)
    with pytest.raises(PlanLocked):
        plan.set_arg(1, "path", "/")
    assert len(plan.steps) == 1


def test_edits_before_approval_are_allowed():
    plan = Plan(task="t", steps=[Step("a", "x"), Step("b", "y")])
    assert plan.remove_step(2) is not None
    plan.add_step(Step("c", "z"))
    assert plan.set_arg(1, "city", "Ohrid") is True
    assert [s.description for s in plan.steps] == ["a", "c"]
    assert plan.steps[0].args == {"city": "Ohrid"}


def test_removing_a_step_that_is_not_there_is_not_an_error():
    plan = Plan(task="t", steps=[Step("a", "x")])
    assert plan.remove_step(7) is None
    assert plan.set_arg(7, "k", "v") is False


@pytest.mark.parametrize("text,expected", [
    ("skip step 3", ("skip", 3)),
    ("drop 2", ("skip", 2)),
    ("прескокни го чекор 2", ("skip", 2)),
    ("no", None),
    ("yes go ahead", None),
])
def test_spoken_plan_edits_are_understood_or_refused(text, expected):
    got = parse_edit(text)
    if expected is None:
        assert got is None
    else:
        assert got[0] == expected[0] and got[1]["number"] == expected[1]


def test_the_hud_payload_is_plain_data():
    plan = Plan(task="t", steps=[Step("a", "x", {"city": "Ohrid"})])
    payload = plan.as_dict()
    assert payload["steps"][0]["n"] == 1
    assert payload["steps"][0]["status"] == PENDING
    assert payload["steps"][0]["args"] == {"city": "Ohrid"}
