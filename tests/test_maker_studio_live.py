"""The two making features as they behave with the REAL libraries installed.

``test_studio_domains.py`` pins the domains with the heavy CAD libraries faked
away. This file pins what only showed up once cadquery and schemdraw were
actually installed and a real request was put through — every one of these is a
regression test for a defect that made a shipped feature unusable rather than
merely imperfect:

* CadQuery's OpenCascade bindings segfault while UNLOADING on Windows, so a
  perfect watertight STL came back as "I tried to model that but couldn't".
* schemdraw's matplotlib backend ends a drawing with ``plt.show()``, which
  blocks forever in a subprocess with no display — every schematic ran to the
  sandbox timeout.
* ``make_app``'s fast path had no room for an adjective, so "build me a simple
  counter app" matched nothing, and its lazy ``called (?P<name>...)`` group
  captured a single letter and dropped the spec.
* ``circuit_help`` registers first and claimed *"design a circuit for an ESP32
  blinking an LED"* — the phrase the Maker Studio docs tell you to say.
"""

from __future__ import annotations

import asyncio
import importlib.util as _iu
import struct

import pytest

from core.config import load_settings
from skills.app_builder_skill import _MAKE_APP, _MAKE_APP_MK, _kind_as_spec
from skills.circuit import CircuitSkill
from studio.domains.model3d import Model3DDomain
from studio.sandbox import SandboxResult, _scrubbed_env


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _pretend_installed(monkeypatch, *libs):
    real = _iu.find_spec
    monkeypatch.setattr(_iu, "find_spec",
                        lambda n: object() if n in libs else real(n))


def _cube_stl(path, s=10.0):
    """A real, complete, watertight binary STL."""
    v = [(0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0),
         (0, 0, s), (s, 0, s), (s, s, s), (0, s, s)]
    f = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 5, 1), (0, 4, 5),
         (1, 6, 2), (1, 5, 6), (2, 7, 3), (2, 6, 7), (3, 4, 0), (3, 7, 4)]
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(f)))
        for (a, b, c) in f:
            fh.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for idx in (a, b, c):
                fh.write(struct.pack("<3f", *v[idx]))
            fh.write(struct.pack("<H", 0))


class CrashingSandbox:
    """A run that produces its files and THEN dies on the way out."""

    def __init__(self, result: SandboxResult, *, write_stl=True, stl=None):
        self._result = result
        self._write_stl = write_stl
        self._stl = stl

    async def run_python(self, script, workdir):
        from pathlib import Path
        wd = Path(workdir)
        if self._write_stl:
            if self._stl is None:
                _cube_stl(wd / "model.stl")
            else:
                (wd / "model.stl").write_bytes(self._stl)
            (wd / "model.step").write_text("ISO-10303-21;", encoding="utf-8")
        return self._result


#: The real thing, measured on Windows with cadquery 2.8.0 + OCP 7.9.3:
#: STATUS_ACCESS_VIOLATION, raised while unloading, with an empty stderr.
_SEGFAULT = SandboxResult(ok=False, returncode=3221225477, stdout="", stderr="")


# -- the 3D teardown crash ----------------------------------------------------

def test_a_model_that_exported_before_the_crash_is_kept(tmp_path, monkeypatch):
    """The whole point: a complete mesh is a success, whatever the exit code.

    Reproducible on a bare ``import cadquery`` with no geometry at all, so
    WITHOUT this every 3D request on this machine failed while a perfect model
    sat in the folder — and the engine spent its entire retry budget asking the
    brain to fix a script that was never broken.
    """
    _pretend_installed(monkeypatch, "cadquery")
    out = _run(Model3DDomain().execute("code", tmp_path,
                                       CrashingSandbox(_SEGFAULT)))
    assert out.ok is True
    assert set(out.artifacts) == {"model.stl", "model.step"}
    assert out.preview == "model.stl"
    # Kept, not hidden: the crash is still on the record.
    assert "3221225477" in out.stderr and "shutting down" in out.stderr


def test_a_clean_run_says_nothing_about_a_crash(tmp_path, monkeypatch):
    _pretend_installed(monkeypatch, "cadquery")
    ok = SandboxResult(ok=True, returncode=0, stdout="", stderr="")
    out = _run(Model3DDomain().execute("code", tmp_path, CrashingSandbox(ok)))
    assert out.ok is True and "shutting down" not in (out.stderr or "")


def test_a_crash_with_no_model_is_still_a_failure(tmp_path, monkeypatch):
    _pretend_installed(monkeypatch, "cadquery")
    out = _run(Model3DDomain().execute(
        "code", tmp_path, CrashingSandbox(_SEGFAULT, write_stl=False)))
    assert out.ok is False


def test_a_half_written_stl_is_not_salvaged(tmp_path, monkeypatch):
    """The salvage must not accept a truncated export.

    The binary STL reader demands the header's triangle count match the file
    length exactly, so a script killed mid-export is rejected — which is what
    keeps this from becoming "declare success whenever a file exists".
    """
    _pretend_installed(monkeypatch, "cadquery")
    truncated = b"\0" * 80 + struct.pack("<I", 500) + b"\x01" * 40
    out = _run(Model3DDomain().execute(
        "code", tmp_path, CrashingSandbox(_SEGFAULT, stl=truncated)))
    assert out.ok is False


def test_a_timeout_is_never_salvaged(tmp_path, monkeypatch):
    """A timeout KILLED the script; a teardown crash FOLLOWED it. Different
    things, and only the second one leaves output worth trusting."""
    _pretend_installed(monkeypatch, "cadquery")
    killed = SandboxResult(ok=False, returncode=-1, stderr="the code timed out",
                           timed_out=True)
    out = _run(Model3DDomain().execute("code", tmp_path,
                                       CrashingSandbox(killed)))
    assert out.ok is False and "timed out" in out.stderr


def test_a_policy_refusal_is_never_salvaged(tmp_path, monkeypatch):
    _pretend_installed(monkeypatch, "cadquery")
    denied = SandboxResult(ok=False, returncode=-1, stderr="not authorised",
                           denied="not authorised")
    out = _run(Model3DDomain().execute("code", tmp_path,
                                       CrashingSandbox(denied)))
    assert out.ok is False


def test_a_salvaged_model_still_faces_the_printability_check(tmp_path, monkeypatch):
    """Salvaging the FILE is not the same as calling it print-ready.

    A non-watertight mesh is complete enough to keep, and ``check`` still
    returns the hard 'fail' advisory that stops the engine claiming it's ready.
    """
    _pretend_installed(monkeypatch, "cadquery")
    one_triangle = (b"\0" * 80 + struct.pack("<I", 1)
                    + struct.pack("<3f", 0.0, 0.0, 0.0)
                    + struct.pack("<3f", 0.0, 0.0, 0.0)
                    + struct.pack("<3f", 1.0, 0.0, 0.0)
                    + struct.pack("<3f", 0.0, 1.0, 0.0)
                    + struct.pack("<H", 0))
    dom = Model3DDomain()
    out = _run(dom.execute("code", tmp_path,
                           CrashingSandbox(_SEGFAULT, stl=one_triangle)))
    assert out.ok is True
    assert any(a.level == "fail" for a in dom.check(out, tmp_path))


# -- the schematic hang -------------------------------------------------------

def test_generated_code_runs_headless(tmp_path):
    """Without this, schemdraw's mpl backend calls plt.show() and blocks
    forever — every schematic ran until the sandbox timeout killed it."""
    assert _scrubbed_env(tmp_path / "home")["MPLBACKEND"] == "Agg"


def test_the_headless_setting_did_not_cost_the_secret_scrub(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")
    monkeypatch.setenv("MEDO_HOME", "keep-me")
    env = _scrubbed_env(tmp_path / "home")
    assert "OPENAI_API_KEY" not in env
    assert env.get("MEDO_HOME") == "keep-me"


# -- make_app: the phrasings that missed --------------------------------------

def _app_spec(text):
    """(name, spec) exactly as the skill computes them, or None for no match."""
    m = _MAKE_APP.search(text) or _MAKE_APP_MK.search(text)
    if m is None:
        return None
    gd = m.groupdict()
    return ((gd.get("name") or "").strip(),
            (gd.get("spec") or "").strip() or _kind_as_spec(gd))


@pytest.mark.parametrize("text, spec", [
    # An adjective between the article and the noun — the shape that missed.
    ("So, can you build me a simple counter app?", "simple counter app"),
    ("build me a simple counter app", "simple counter app"),
    ("make me a to-do app", "to-do app"),
    ("make a small budgeting app", "small budgeting app"),
    ("make a simple calculator program", "simple calculator program"),
    ("can you make me a python script that renames files", "renames files"),
    ("write me a quick utility that resizes images", "resizes images"),
    # ...and the ones that already worked, which must keep working.
    ("make an app that tracks my water intake", "tracks my water intake"),
    ("create a web app for tracking runs", "tracking runs"),
    # Macedonian puts its adjective in the same place.
    ("направи ми една проста апликација за броење", "броење"),
    ("направи апликација за прогноза", "прогноза"),
    ("направи ми игра за помнење", "помнење"),
])
def test_the_app_brief_survives_an_adjective(text, spec):
    got = _app_spec(text)
    assert got is not None, f"{text!r} matched nothing"
    assert got[1] == spec


@pytest.mark.parametrize("text, name, spec", [
    ("build a tool called budgeter that logs expenses", "budgeter",
     "logs expenses"),
    ("build me a tool called budgeter that logs expenses", "budgeter",
     "logs expenses"),
    ("make an app called Ledger that tracks expenses", "Ledger",
     "tracks expenses"),
])
def test_a_named_app_keeps_its_whole_name_and_its_spec(text, name, spec):
    """A LAZY group followed by an OPTIONAL one matches as little as it can.

    So "a tool called budgeter that logs expenses" named the app **"b"** and
    reported no spec at all — every "called X" phrasing, in the skill's own
    docstring example, built nothing and asked what the app should do.
    """
    assert _app_spec(text) == (name, spec)


@pytest.mark.parametrize("text", [
    "make me an app",
    "make me a simple web app",
    "build a little game",
    "направи ми мала игра",
])
def test_a_kind_without_a_purpose_still_asks(text):
    """"a simple web app" says what SORT of thing, never what it does.

    Guessing a brief there would confidently build the wrong thing, so these
    keep asking — the fast path is widened, not made reckless.
    """
    got = _app_spec(text)
    assert got is not None and got[1] == ""


@pytest.mark.parametrize("text", [
    "make a case for hiring more staff",
    "make a stand against this",
    "build a bridge between the teams",
    "create a plan for the weekend",
])
def test_the_wider_pattern_did_not_start_eating_idioms(text):
    assert _app_spec(text) is None


# -- circuit_help vs. the schematic generator ---------------------------------

def _circuit_skill(studio_on):
    settings = load_settings()
    settings.studio.enabled = studio_on
    return CircuitSkill(settings)


@pytest.mark.parametrize("text", [
    "design a circuit for an ESP32 blinking an LED",
    "draw a schematic for a 3.3V regulator",
    "sketch a circuit for a motor driver",
    "нацртај ми шема за напојување",
])
def test_a_request_to_DRAW_one_is_left_to_the_studio(text):
    assert _circuit_skill(True).match(text) is None


@pytest.mark.parametrize("text", [
    "how do I wire up an LED to an Arduino",
    "help me build the circuit for a temperature sensor",
    "wiring diagram for an ultrasonic distance sensor",
    "како да поврзам ардуино со диода",
])
def test_a_QUESTION_about_wiring_is_still_answered(text):
    assert _circuit_skill(True).match(text) is not None


@pytest.mark.parametrize("text", [
    "design a circuit for an ESP32 blinking an LED",
    "draw a schematic for a 3.3V regulator",
])
def test_with_the_studio_off_nothing_is_given_up(text):
    """An explanation beats a phrase that routes nowhere."""
    assert _circuit_skill(False).match(text) is not None


# -- the whole registry -------------------------------------------------------

def _registry(studio_on):
    from main import Announcer, build_registry
    settings = load_settings()
    settings.studio.enabled = studio_on
    return build_registry(settings, Announcer())


def test_the_studio_switch_is_what_makes_the_skills_exist():
    """The reason "build me a 3D model" was answered with "I have no tool for
    that": with the switch off there genuinely is no tool."""
    on, off = _registry(True).names(), _registry(False).names()
    for name in ("model_3d", "design_circuit", "build_code", "studio_projects"):
        assert name in on
        assert name not in off


@pytest.mark.parametrize("text, expected", [
    ("Can you build me a 3D model of a simple rocket", "model_3d"),
    ("3d print a phone stand", "model_3d"),
    ("model a bracket for a motor", "model_3d"),
    ("design an enclosure for an ESP32", "model_3d"),
    ("направи ми 3д модел на држач", "model_3d"),
    # The phrase docs/maker-studio.md tells you to say.
    ("design a circuit for an ESP32 blinking an LED", "design_circuit"),
    ("draw a schematic for a 3.3V regulator", "design_circuit"),
    ("how do I wire up an LED to an Arduino", "circuit_help"),
    ("So, can you build me a simple counter app?", "make_app"),
    ("make me a to-do app", "make_app"),
    ("build a tool called budgeter that logs expenses", "make_app"),
    ("направи ми мала игра", "make_app"),
    # Untouched neighbours.
    ("what time is it", "datetime"),
    ("какво е времето", "weather"),
    ("set a timer for five minutes", "timers"),
])
def test_the_making_phrases_reach_the_right_skill(text, expected):
    hit = _registry(True).find_match(text)
    assert (hit[0].name if hit else None) == expected


# -- a build must not take the assistant away with it -------------------------

class _FakeEngine:
    """An engine that takes as long as you let it."""

    def __init__(self, *, gate=None, result=None):
        self.gate = gate
        self.result = result
        self.calls = 0

    async def create(self, domain, description, answers=None):
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        from studio.engine import StudioResult
        return self.result or StudioResult(ok=True, artifacts=["model.stl"],
                                           version=1)


def _model_skill(**kw):
    from skills.maker_studio import Model3DSkill
    settings = load_settings()
    settings.studio.enabled = True
    return Model3DSkill(settings, **kw)


def _request(text):
    from skills.base import SkillRequest
    skill = _model_skill()
    return SkillRequest(text=text, match=skill.match(text), args={}, context={})


async def test_a_long_build_answers_at_once_and_speaks_when_done():
    """Measured: a real phone stand is 514 s from the sentence to the STL.

    Awaited inside the turn, that is eight minutes of an assistant that cannot
    hear you. So the turn returns immediately and the result is announced.
    """
    said = []
    gate = asyncio.Event()
    engine = _FakeEngine(gate=gate)
    skill = _model_skill(engine=engine, announcer=lambda m: said.append(m))

    reply = await skill.execute(_request("3d print a phone stand"))

    assert "ready" in reply.speech.lower()
    assert reply.data.get("studio") == "started"
    assert said == []                       # nothing announced yet
    gate.set()                              # the build finishes
    await skill._task
    assert len(said) == 1 and "model.stl" in said[0]


async def test_a_second_request_while_one_is_running_is_declined():
    gate = asyncio.Event()
    engine = _FakeEngine(gate=gate)
    skill = _model_skill(engine=engine, announcer=lambda m: None)
    await skill.execute(_request("3d print a phone stand"))
    second = await skill.execute(_request("model a bracket for a motor"))
    assert second.success is False and "finish" in second.speech.lower()
    gate.set()
    await skill._task
    assert engine.calls == 1        # the bracket never started a second build


async def test_the_builder_frees_itself_after_a_crash():
    """A failed build must not wedge the skill shut forever."""
    class Boom:
        async def create(self, *a, **k):
            raise RuntimeError("the brain fell over")

    said = []
    skill = _model_skill(engine=Boom(), announcer=lambda m: said.append(m))
    await skill.execute(_request("3d print a phone stand"))
    await skill._task
    assert skill._busy is False
    assert said and "fell over" in said[0]


async def test_without_an_announcer_it_stays_synchronous():
    """Nowhere to announce to -> answer in the turn, as the tests expect."""
    engine = _FakeEngine()
    skill = _model_skill(engine=engine)
    reply = await skill.execute(_request("3d print a phone stand"))
    assert engine.calls == 1
    assert reply.data.get("studio") != "started"


async def test_a_missing_spec_is_asked_in_the_turn_not_announced_later():
    """A question ten minutes after you spoke is no use to anyone."""
    engine = _FakeEngine()
    skill = _model_skill(engine=engine, announcer=lambda m: None)
    from skills.base import SkillRequest
    reply = await skill.execute(SkillRequest(text="model it", match=None,
                                             args={"description": "a"},
                                             context={}))
    assert reply.await_reply is True
    assert engine.calls == 0


# -- the CLI brain must not be told to speak when asked for code --------------

def test_a_cli_prompt_for_speech_asks_for_a_spoken_answer():
    from llm.client import _flatten_for_cli
    prompt = _flatten_for_cli([{"role": "system", "content": "Be brief."},
                               {"role": "user", "content": "hello"}])
    assert "spoken answer" in prompt


def test_a_cli_prompt_for_an_artifact_does_not():
    """The trailer is the LAST line the agent reads, so it beats the system
    prompt above it. With it, "Output ONLY the Python code" produced a
    paragraph of English prose describing a phone stand — which was then
    written to model.py and executed."""
    from llm.client import _flatten_for_cli
    system = "Output ONLY the Python code — no prose, no markdown fences."
    prompt = _flatten_for_cli([{"role": "system", "content": system},
                               {"role": "user", "content": "model a bracket"}],
                              for_speech=False)
    assert "spoken answer" not in prompt
    assert system in prompt and "model a bracket" in prompt


async def test_the_studio_brain_asks_for_code_not_speech():
    from studio.brain import Brain
    seen = {}

    class FakeClient:
        async def chat(self, model, messages, **kw):
            seen.update(kw)
            return {"content": "import cadquery"}

    brain = Brain(model="m", settings=load_settings())
    brain._client = FakeClient()
    await brain.write_code("sys", "user")
    assert seen.get("for_speech") is False


def test_writing_code_gets_a_longer_budget_than_a_spoken_turn():
    """195 s of real code writing was being killed by a 180 s conversation
    timeout, four times over, and reported as "I couldn't model that"."""
    from core.config import StudioConfig
    assert StudioConfig().generate_timeout_s > 180.0


# -- round two: what was still missed after the first pass --------------------

@pytest.mark.parametrize("text, spec", [
    # Reported verbatim. Four words of description before the noun, so the
    # {0,2} modifier run from the first pass still missed it.
    ("Can you build me a simple camera with filters app?",
     "simple camera with filters app"),
    ("Make a snapchat clone app", "snapchat clone app"),
    # Whisper writes the wake word into the transcript, and mishears it.
    ("Hey, Middle, make a Snapchat clone app.", "snapchat clone app"),
    ("make a really nice photo editing app", "really nice photo editing app"),
])
def test_a_longer_description_still_reaches_the_builder(text, spec):
    got = _app_spec(text)
    assert got is not None, f"{text!r} matched nothing"
    assert got[1].lower() == spec


@pytest.mark.parametrize("text", [
    # A determiner inside the run means this is a different sentence.
    "make a note about the tool",
    "make a copy of that script",
    "write me a song about my program",
    "make a list of every tool",
    "make a video about the game",
    "take a picture of the app",
    "make a joke about my website",
    "направи ми белешка за алатката",
])
def test_a_determiner_in_the_run_is_not_a_build_request(text):
    assert _app_spec(text) is None


# -- "turn off scan lines" was read as an app called "lines" ------------------

def _learn_skill():
    from skills.software_learn import LearnAppSkill

    class _Mech:
        def focus_app(self, name): return False

    return LearnAppSkill(_Mech(), None)


@pytest.mark.parametrize("text", [
    "Turn off scan lines.",
    "turn on scan lines",
    "the scan results were fine",
    "I want to study French",
    "can you map out a plan",
])
def test_a_verb_in_mid_sentence_is_not_an_app_to_learn(text):
    """Matching the verb ANYWHERE claimed any sentence containing "scan <word>".

    "Turn off scan lines" became *learn the app called "lines"*, answered with
    "I couldn't find lines open. Open it first, then say 'learn lines'." These
    are imperatives, so the verb has to lead.
    """
    assert _learn_skill().match(text) is None


@pytest.mark.parametrize("text", [
    "scan CapCut",
    "learn how to use CapCut",
    "learn the capcut app",
    "map davinci resolve",
    "explore Photoshop's UI",
    "can you scan Photoshop",
    "hey MEDO, learn how to use Blender",
    "please map Photoshop's interface",
])
def test_the_real_learn_requests_still_work(text):
    assert _learn_skill().match(text) is not None


# -- the description can come BEFORE the noun ---------------------------------

def _circuit_brief(text):
    """The brief design_circuit extracts, or None if it doesn't claim the text."""
    from skills.base import SkillRequest
    from skills.maker_studio import DesignCircuitSkill
    settings = load_settings()
    settings.studio.enabled = True
    skill = DesignCircuitSkill(settings)
    m = skill.match(text)
    if m is None:
        return None
    return skill._description(SkillRequest(text=text, match=m, context={}))


@pytest.mark.parametrize("text, brief", [
    # Reported verbatim. The noun is LAST, after three words of description,
    # so a pattern expecting it straight after the article missed entirely —
    # and the LLM answered with the sentence "Detailed design and explanations
    # for an RC alternating current circuit." and nothing else.
    ("Could you design an RC alternating current circuit?",
     "RC alternating current"),
    ("design a simple low pass filter circuit", "simple low pass filter"),
    # ...and the noun-first shape has to keep working.
    ("design a circuit for an ESP32 blinking an LED", "an ESP32 blinking an LED"),
    ("draw a schematic for a 3.3V regulator", "a 3.3V regulator"),
])
def test_a_circuit_described_before_the_noun_still_gets_designed(text, brief):
    assert _circuit_brief(text) == brief


@pytest.mark.parametrize("text", [
    "how do I wire up an LED to an Arduino",
    "help me build the circuit for a temperature sensor",
    "wiring diagram for an ultrasonic distance sensor",
])
def test_the_wider_circuit_pattern_still_leaves_questions_alone(text):
    """Widening the slot must not start swallowing wiring QUESTIONS."""
    from main import Announcer, build_registry
    settings = load_settings()
    settings.studio.enabled = True
    hit = build_registry(settings, Announcer()).find_match(text)
    assert hit is not None and hit[0].name == "circuit_help"


@pytest.mark.parametrize("text, expected", [
    # Straight from the bug list, after the fixes.
    ("build me a Flappy Bird app.", "make_app"),
    ("Build me a simple app where you have a camera and it has filters "
     "like snapchat", "make_app"),
    ("Make build me a camera with filters app.", "make_app"),
    ("Could you design an RC alternating current circuit?", "design_circuit"),
])
def test_the_reported_phrases_reach_a_builder(text, expected):
    from main import Announcer, build_registry
    settings = load_settings()
    settings.studio.enabled = True
    hit = build_registry(settings, Announcer()).find_match(text)
    assert (hit[0].name if hit else None) == expected
