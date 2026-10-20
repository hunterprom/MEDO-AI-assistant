"""Cowork S4: the skill wrapper, the config gate, dry run, and saving a plan.

These are the tests about MEDO rather than about the plan: does the feature
appear only when it is switched on, does it grab phrases that belong to other
skills (it must not), does a task from a timer get refused, and does "save that
as demo prep" produce a routine that actually works.
"""

from __future__ import annotations

import pytest

from cowork.plan import Plan, Step
from cowork.save import clean_name, plan_to_routine, save_plan_as_routine, slug
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult
from skills.cowork_skill import CoworkSkill


class _Config:
    enabled = True
    max_steps = 6
    allow_unattended = False
    model = ""
    narrate = True
    dry_run = False
    save_routines = True

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)


class _Fake(Skill):
    def __init__(self, name, *, confirm=False):
        self.name = name
        self.description = name
        self.patterns = []
        self.requires_confirmation = confirm
        self.calls = []

    async def execute(self, request: SkillRequest) -> SkillResult:
        self.calls.append(request)
        return SkillResult(f"{self.name} done")


def _registry(*skills):
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


def _brain(payload):
    async def think(prompt):
        return payload
    return think


ONE_STEP = ('{"steps": [{"description": "open OBS", "capability": "obs"}]}')


def _skill(config=None, registry=None, think=ONE_STEP, tmp_path=None):
    return CoworkSkill(config or _Config(), registry or _registry(_Fake("obs")),
                       think=_brain(think) if think else None,
                       routines_dir=tmp_path)


def _request(text, **context):
    match = None
    for pattern in CoworkSkill.patterns:
        match = pattern.search(text)
        if match:
            break
    return SkillRequest(text=text, match=match, context=context)


# --- the config gate ----------------------------------------------------------


@pytest.mark.asyncio
async def test_switched_off_it_does_nothing_and_says_so():
    skill = _skill(_Config(enabled=False))
    result = await skill.execute(_request("cowork: open OBS"))
    assert result.success is False and "switched off" in result.speech


def test_it_is_not_registered_at_all_when_disabled():
    """Disabled means the tool does not exist — MEDO can't offer what it hasn't."""
    from core.config import load_settings
    from main import Announcer, build_registry

    settings = load_settings()
    settings.cowork.enabled = False
    assert build_registry(settings, Announcer()).get("cowork") is None


def test_it_is_registered_when_enabled():
    from core.config import load_settings
    from main import Announcer, build_registry

    settings = load_settings()
    settings.cowork.enabled = True
    registry = build_registry(settings, Announcer())
    skill = registry.get("cowork")
    assert skill is not None
    # It declares nothing of its own: each STEP is gated live instead.
    assert not getattr(skill, "capabilities", frozenset())
    assert skill.requires_confirmation is False


# --- routing: take the tasks meant for it, and nothing else -------------------


@pytest.mark.parametrize("text,task", [
    ("cowork: open OBS and start recording", "open OBS and start recording"),
    ("plan and do this: open OBS then record", "open OBS then record"),
    ("take care of this: open the report and email it",
     "open the report and email it"),
    ("испланирај и направи: отвори го извештајот", "отвори го извештајот"),
])
def test_the_delegation_phrases_route_here(text, task):
    request = _request(text)
    assert request.match is not None
    assert CoworkSkill(_Config(), _registry())._task_of(request) == task


@pytest.mark.parametrize("text", [
    "do this for me",                       # the screen agent owns this
    "operate my screen",
    "save this as a Word document",         # make_document owns this
    "that's a bug",
    "what's the weather in Skopje",
    "play some music",
])
def test_it_never_grabs_another_skills_phrase(text):
    assert all(p.search(text) is None for p in CoworkSkill.patterns), text


@pytest.mark.asyncio
async def test_reached_by_meaning_the_whole_utterance_is_the_task():
    """The semantic tier hands over a bare request with no match and no args."""
    skill = _skill()
    request = SkillRequest(text="open OBS and start recording", context={})
    assert skill._task_of(request) == "open OBS and start recording"


# --- nobody there to approve --------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["routine", "webhook", "schedule",
                                    "trigger", "briefing"])
async def test_a_task_from_an_unattended_source_is_refused(source):
    obs = _Fake("obs")
    skill = _skill(registry=_registry(obs))
    result = await skill.execute(_request("cowork: open OBS", source=source))
    assert obs.calls == []
    assert result.success is False and "schedule" in result.speech
    assert result.await_reply is False       # nothing is waiting for an answer


@pytest.mark.asyncio
@pytest.mark.parametrize("context", [{"untrusted": True},
                                     {"provenance": "untrusted"}])
async def test_a_task_that_came_from_content_is_refused(context):
    """The trust boundary: a "task" found in a document, a web page or another
    program's call is content, and content never becomes a plan."""
    obs = _Fake("obs")
    skill = _skill(registry=_registry(obs))
    result = await skill.execute(_request("cowork: open OBS", **context))
    assert obs.calls == [] and result.success is False
    assert "directly" in result.speech


def test_cowork_is_not_offered_over_mcp():
    """It declares no capabilities of its own — each step is gated instead — so
    the read-only filter would wave it through. There is nobody on the MCP path
    to approve a plan, and a task delegated by other software is content."""
    from mcp_servers.skills_server import exposed_skills, is_read_only

    from core.config import load_settings
    from main import Announcer, build_registry

    settings = load_settings()
    settings.cowork.enabled = True
    registry = build_registry(settings, Announcer())
    assert registry.get("cowork") is not None          # it IS registered…
    assert is_read_only(registry.get("cowork")) is False
    assert "cowork" not in {s.name for s in exposed_skills(registry)}


@pytest.mark.asyncio
async def test_the_same_task_from_a_person_is_planned():
    skill = _skill()
    result = await skill.execute(_request("cowork: open OBS", source="voice"))
    assert "Here's my plan" in result.speech and result.await_reply is True


@pytest.mark.asyncio
async def test_unattended_can_be_allowed_deliberately():
    skill = _skill(_Config(allow_unattended=True))
    result = await skill.execute(_request("cowork: open OBS", source="routine"))
    assert "Here's my plan" in result.speech


# --- the conversation ---------------------------------------------------------


@pytest.mark.asyncio
async def test_the_captured_reply_continues_the_same_plan():
    obs = _Fake("obs")
    skill = _skill(registry=_registry(obs))
    first = await skill.execute(_request("cowork: open OBS"))
    assert obs.calls == [] and first.await_reply is True
    assert first.reply_is_open is True        # "skip step 2" must reach us too
    second = await skill.execute(
        SkillRequest(text="yes", context={"captured_reply": True}))
    assert len(obs.calls) == 1 and "Task finished" in second.speech


@pytest.mark.asyncio
async def test_a_reply_with_no_plan_in_flight_starts_a_new_task_instead():
    skill = _skill()
    result = await skill.execute(
        SkillRequest(text="cowork: open OBS", match=None,
                     context={"captured_reply": True}))
    assert "Here's my plan" in result.speech


@pytest.mark.asyncio
@pytest.mark.parametrize("text,expected", [
    ("cowork: open OBS", "Let me work out a plan."),
    ("испланирај: отвори го OBS", "Само да составам план."),
])
async def test_it_says_something_before_the_long_planning_wait(text, expected):
    """Measured: 60-120 s on qwen3:30b. Silence that long reads as ignored."""
    said = []

    async def announce(line):
        said.append(line)

    skill = CoworkSkill(_Config(), _registry(_Fake("obs")),
                        think=_brain(ONE_STEP), announce=announce)
    await skill.execute(_request(text))
    assert said and said[0] == expected


@pytest.mark.asyncio
async def test_a_broken_announcer_never_stops_a_task():
    async def explode(line):
        raise RuntimeError("no voice")

    skill = CoworkSkill(_Config(), _registry(_Fake("obs")),
                        think=_brain(ONE_STEP), announce=explode)
    result = await skill.execute(_request("cowork: open OBS"))
    assert "Here's my plan" in result.speech


@pytest.mark.asyncio
async def test_an_empty_task_asks_what_to_do():
    skill = _skill()
    result = await skill.execute(SkillRequest(text="", context={}))
    assert result.success is False and "?" in result.speech


# --- dry run ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_spoken_dry_run_plans_and_executes_nothing():
    obs = _Fake("obs")
    skill = _skill(registry=_registry(obs))
    result = await skill.execute(_request("cowork: dry run, open OBS"))
    assert obs.calls == []
    assert "Dry run" in result.speech
    assert result.await_reply is False       # there is nothing to approve


@pytest.mark.asyncio
async def test_dry_run_can_be_the_configured_default():
    obs = _Fake("obs")
    skill = _skill(_Config(dry_run=True), registry=_registry(obs))
    result = await skill.execute(_request("cowork: open OBS"))
    assert obs.calls == [] and "Dry run" in result.speech


# --- no brain -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_a_brain_it_says_so_rather_than_guessing_a_plan():
    obs = _Fake("obs")
    skill = _skill(registry=_registry(obs), think=None)
    result = await skill.execute(_request("cowork: open OBS"))
    assert obs.calls == [] and "offline" in result.speech.lower()


# --- saving a plan as a routine -----------------------------------------------


def test_a_plan_becomes_routine_yaml():
    plan = Plan(task="prep my demo", steps=[
        Step("open OBS", "obs"),
        Step("weather", "weather", {"city": "Skopje"}),
        Step("order a pizza", "", problem="I have no skill for that"),
    ])
    data = plan_to_routine(plan, "demo prep")
    assert data["name"] == "demo prep"
    assert data["trigger"]["phrase"] == ["demo prep"]
    # The impossible step is not a routine step — a routine cannot "not do" it.
    assert data["steps"] == [{"skill": "obs"},
                             {"skill": "weather", "args": {"city": "Skopje"}}]
    assert "prep my demo" in data["description"]


def test_the_saved_routine_is_written_and_registered_live(tmp_path):
    registry = _registry(_Fake("obs"))
    plan = Plan(task="prep my demo", steps=[Step("open OBS", "obs")])
    name, error = save_plan_as_routine(plan, "demo prep", registry, tmp_path)
    assert error == "" and name == "demo prep"
    assert (tmp_path / "demo-prep.yaml").exists()
    # …and usable NOW, not after the next restart.
    assert registry.get("routine_demo_prep") is not None
    assert registry.find_match("demo prep") is not None


def test_a_plan_with_a_confirming_step_is_refused_with_the_reason(tmp_path):
    """Composer's rule, not a second copy of it: a routine may not contain a
    step that needs a spoken yes, because on a timer nobody can give one."""
    registry = _registry(_Fake("power", confirm=True))
    plan = Plan(task="shut down", steps=[Step("shut down", "power")])
    name, error = save_plan_as_routine(plan, "bedtime", registry, tmp_path)
    assert name == "" and "confirmation" in error
    assert list(tmp_path.iterdir()) == []


def test_a_plan_with_nothing_doable_is_not_saved(tmp_path):
    plan = Plan(task="order pizza", steps=[
        Step("order a pizza", "", problem="I have no skill for that")])
    name, error = save_plan_as_routine(plan, "dinner", _registry(), tmp_path)
    assert name == "" and "nothing" in error


def test_an_existing_routine_file_is_never_overwritten(tmp_path):
    (tmp_path / "demo-prep.yaml").write_text("name: something else\n",
                                             encoding="utf-8")
    registry = _registry(_Fake("obs"))
    plan = Plan(task="t", steps=[Step("open OBS", "obs")])
    name, error = save_plan_as_routine(plan, "demo prep", registry, tmp_path)
    assert name == "" and "already" in error
    assert "something else" in (tmp_path / "demo-prep.yaml").read_text(
        encoding="utf-8")


@pytest.mark.parametrize("spoken,expected", [
    ("demo prep", "demo prep"),
    ("'demo prep' routine", "demo prep"),
    ("demo prep routine, please", "demo prep"),
    ("  Demo   Prep  ", "Demo Prep"),
])
def test_the_spoken_name_is_cleaned_up(spoken, expected):
    assert clean_name(spoken) == expected


@pytest.mark.parametrize("name,expected", [
    ("demo prep", "demo-prep"),
    ("Demo/../prep", "demo-prep"),          # cannot escape the folder
    ("подготовка", "podgotovka"),           # transliterated, not dropped
])
def test_the_filename_is_a_safe_slug(name, expected):
    assert slug(name) == expected


def test_a_macedonian_name_keeps_its_name_and_gets_a_latin_filename(tmp_path):
    """Dropping the Cyrillic would leave a file called ".yaml" — a routine the
    user could never find again."""
    registry = _registry(_Fake("obs"))
    plan = Plan(task="t", steps=[Step("open OBS", "obs")])
    name, error = save_plan_as_routine(plan, "подготовка", registry, tmp_path)
    assert error == "" and name == "подготовка"
    assert (tmp_path / "podgotovka.yaml").exists()
    assert registry.find_match("подготовка") is not None


def test_a_name_with_no_letters_at_all_is_refused(tmp_path):
    registry = _registry(_Fake("obs"))
    plan = Plan(task="t", steps=[Step("open OBS", "obs")])
    name, error = save_plan_as_routine(plan, "!!!", registry, tmp_path)
    assert name == "" and error
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_saving_with_no_plan_says_there_is_nothing_to_save(tmp_path):
    skill = _skill(tmp_path=tmp_path)
    result = await skill.execute(_request("save that as a demo prep routine"))
    assert result.success is False and "no plan" in result.speech.lower()


@pytest.mark.asyncio
async def test_a_finished_plan_can_be_saved_by_voice(tmp_path):
    obs = _Fake("obs")
    registry = _registry(obs)
    skill = _skill(registry=registry, tmp_path=tmp_path)
    await skill.execute(_request("cowork: open OBS"))
    await skill.execute(SkillRequest(text="yes",
                                     context={"captured_reply": True}))
    result = await skill.execute(_request("save that as a demo prep routine"))
    assert "demo prep" in result.speech
    assert (tmp_path / "demo-prep.yaml").exists()
    assert registry.get("routine_demo_prep") is not None


@pytest.mark.asyncio
async def test_saving_can_be_switched_off(tmp_path):
    skill = _skill(_Config(save_routines=False), tmp_path=tmp_path)
    await skill.execute(_request("cowork: open OBS"))
    result = await skill.execute(_request("save that as a demo prep routine"))
    assert result.success is False
    assert list(tmp_path.iterdir()) == []


# --- config -------------------------------------------------------------------


def test_the_config_defaults_are_the_bounded_ones():
    from core.config import CoworkConfig

    config = CoworkConfig()
    assert config.enabled is False           # opt in, like every acting feature
    assert config.max_steps == 6
    assert config.allow_unattended is False
    # Planning is a build, not a spoken turn: it gets more than the
    # conversation's 120 s, or a long plan dies mid-generation.
    assert config.plan_timeout_s > 120


@pytest.mark.parametrize("field,value,fatal", [
    ("max_steps", 0, True),
    ("max_steps", 40, False),
    ("allow_unattended", True, False),
])
def test_the_config_check_catches_a_nonsense_setting(field, value, fatal):
    from core.config import load_settings
    from core.config_check import check_settings

    settings = load_settings()
    settings.cowork.enabled = True
    setattr(settings.cowork, field, value)
    problems = [p for p in check_settings(settings)
                if p.key.startswith("cowork.")]
    assert problems and any(p.fatal is fatal for p in problems)


def test_a_disabled_cowork_is_never_complained_about():
    from core.config import load_settings
    from core.config_check import check_settings

    settings = load_settings()
    settings.cowork.enabled = False
    settings.cowork.max_steps = 0
    assert [p for p in check_settings(settings)
            if p.setting.startswith("cowork.")] == []
