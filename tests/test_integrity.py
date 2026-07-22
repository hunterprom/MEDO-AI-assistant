"""Whole-app integrity checks — the sweep, made permanent.

Most bugs this project has actually shipped were not logic errors inside a
skill. They were wiring: a module that stopped importing, a name used but
never imported, a schema whose ``required`` listed a property it never
declared, two skills quietly claiming the same name. Unit tests miss all of
those because nothing imports the broken thing until a user speaks.

So this file asserts the *shape* of the app rather than any behaviour.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

from core.config import load_settings
from core.modes import SessionModes
from main import Announcer, build_registry

#: Every package whose modules must import cleanly. vision/ is excluded: it
#: runs in a separate venv and needs mediapipe, which the main venv lacks by
#: design (the numpy<2 pin is quarantined there).
APP_PACKAGES = ("core", "skills", "llm", "remote", "ui", "voice")


@pytest.fixture(scope="module")
def registry():
    settings = load_settings()
    settings.browser.enabled = True          # exercise the browser skills too

    async def stub(*args, **kwargs):
        return "x"

    return build_registry(settings, Announcer(), modes=SessionModes(),
                          expert=stub, synthesize_council=stub)


@pytest.mark.parametrize("package", APP_PACKAGES)
def test_every_module_imports(package):
    """A module that stops importing is invisible until something needs it."""
    broken = []
    for module in pkgutil.iter_modules([package]):
        name = f"{package}.{module.name}"
        try:
            importlib.import_module(name)
        except Exception as exc:                       # noqa: BLE001 - report all
            broken.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not broken, "modules failed to import:\n  " + "\n  ".join(broken)


def test_skill_names_are_unique(registry):
    """Duplicate names silently shadow a skill on the LLM tool path."""
    names = [skill.name for skill in registry.all()]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"duplicate skill names: {duplicates}"


def test_every_pattern_is_compiled(registry):
    for skill in registry.all():
        for pattern in skill.patterns:
            assert hasattr(pattern, "search"), f"{skill.name}: pattern not compiled"


def test_every_tool_schema_is_well_formed(registry):
    """A malformed schema is rejected by the provider, not by us — and the
    symptom is the model silently losing one tool."""
    problems = []
    for skill in registry.all():
        schema = skill.tool_schema()
        function = schema.get("function", {})
        if schema.get("type") != "function" or not function.get("name"):
            problems.append(f"{skill.name}: missing type/name")
            continue
        params = function.get("parameters")
        if not isinstance(params, dict):
            problems.append(f"{skill.name}: parameters is not an object")
            continue
        properties = params.get("properties", {})
        for required in params.get("required", []):
            if required not in properties:
                problems.append(
                    f"{skill.name}: required {required!r} is not a declared property")
    assert not problems, "bad tool schemas:\n  " + "\n  ".join(problems)


def test_tool_name_matches_skill_name(registry):
    """dispatch_tool looks the skill up by the name the model was given."""
    for skill in registry.all():
        assert skill.tool_schema()["function"]["name"] == skill.name


@pytest.mark.parametrize("probe", ["", "   ", ".", "?"])
def test_no_skill_claims_empty_input(registry, probe):
    """A catch-all pattern would swallow every utterance ahead of the LLM."""
    greedy = [s.name for s in registry.all() if s.match(probe)]
    assert not greedy, f"{greedy} match {probe!r}"


def test_actuating_skills_declare_controls_pc(registry):
    """The HUD switch and Lion Mode both key off this flag; a skill that acts
    on the machine without it bypasses the only gate there is."""
    acts_on_pc = {
        "apps", "files", "edit_file", "open_in_editor", "write_in_app",
        "open_website", "site_search", "play_media", "browser_control",
        "browser_task", "operate_screen", "type_text", "press_keys",
        "screenshot", "power", "volume", "media", "clipboard", "dictation",
    }
    for name in acts_on_pc:
        skill = registry.get(name)
        if skill is not None:
            assert skill.controls_pc, f"{name} acts on the PC but isn't gated"


#: Skills whose degenerate-input path reaches the network, a camera, or a
#: model. Excluded to keep the suite deterministic and fast.
NEEDS_THE_WORLD = {"briefing", "see_camera", "see_screen", "see_bench",
                   "point_at", "weather", "news", "web_search", "documents"}


@pytest.mark.asyncio
async def test_every_answering_skill_survives_degenerate_input(registry):
    """Call every NON-actuating skill with no match and no args.

    A correct skill answers "what did you mean?"; a buggy one raises — KeyError
    on a regex group that never matched, AttributeError on a None match,
    TypeError from a signature that drifted. That is the exact path taken when
    the LLM calls a tool with arguments it invented, so it is worth sweeping.

    ``controls_pc`` skills are deliberately NOT executed here. Calling
    ``execute()`` directly bypasses the router, which is the only thing that
    enforces the PC-control switch — an earlier version of this test swept
    every skill and suspended the machine, because PowerSkill inferred "sleep"
    from empty input and slept without confirmation. A test must never be the
    thing that acts on the user's computer. That those skills are gated at all
    is covered by ``test_actuating_skills_declare_controls_pc``.
    """
    from skills.base import SkillRequest

    broken = []
    for skill in registry.all():
        if skill.name in NEEDS_THE_WORLD or skill.controls_pc:
            continue
        try:
            result = await skill.execute(SkillRequest(text="", args={}, context={}))
        except Exception as exc:                       # noqa: BLE001 - report all
            broken.append(f"{skill.name}: {type(exc).__name__}: {exc}")
            continue
        if result is None or not hasattr(result, "speech"):
            broken.append(f"{skill.name}: returned {type(result).__name__}")
    assert not broken, "skills that crash on empty input:\n  " + "\n  ".join(broken)
