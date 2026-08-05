"""Maker Studio S2/S3 domains + skills: brief/prompt building, sandboxed-execute
artifact detection, the real STL printability check, and the routing skills. The
brain, the sandbox and the heavy CAD libs are all faked/absent, so every rung is
pinned here without SchemDraw/CadQuery installed.
"""

from __future__ import annotations

import asyncio
import importlib.util as _iu
import struct
from pathlib import Path

import pytest

from skills.base import SkillRequest
from studio.domain import ExecOutcome
from studio.domains.mesh import inspect_stl
from studio.domains.model3d import Model3DDomain
from studio.domains.schematic import SchematicDomain


# -- helpers ------------------------------------------------------------------

def _write_binary_stl(path, faces, verts):
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(faces)))
        for (a, b, c) in faces:
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for idx in (a, b, c):
                f.write(struct.pack("<3f", *verts[idx]))
            f.write(struct.pack("<H", 0))


def _cube_stl(path, s=10.0):
    v = [(0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0),
         (0, 0, s), (s, 0, s), (s, s, s), (0, s, s)]
    f = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 5, 1), (0, 4, 5),
         (1, 6, 2), (1, 5, 6), (2, 7, 3), (2, 6, 7), (3, 4, 0), (3, 7, 4)]
    _write_binary_stl(path, f, v)


class FakeSandbox:
    def __init__(self, *, ok=True, writes=None, stderr=""):
        self.ok = ok
        self.writes = writes or {}
        self.stderr = stderr

    async def run_python(self, script, workdir):
        from studio.sandbox import SandboxResult
        for name, content in self.writes.items():
            (Path(workdir) / name).write_text(content, encoding="utf-8")
        return SandboxResult(ok=self.ok, returncode=0 if self.ok else 1,
                             stdout="", stderr=self.stderr)


def _pretend_installed(monkeypatch, *libs):
    real = _iu.find_spec
    monkeypatch.setattr(_iu, "find_spec",
                        lambda n: object() if n in libs else real(n))


def _run(coro):
    return asyncio.run(coro)


# -- STL inspection (the real printability check) -----------------------------

def test_inspect_stl_watertight_cube(tmp_path):
    p = tmp_path / "cube.stl"
    _cube_stl(p, s=10.0)
    info = inspect_stl(p)
    assert info["triangles"] == 12 and info["watertight"] is True
    assert info["open_edges"] == 0
    assert tuple(round(b) for b in info["bbox"]) == (10, 10, 10)


def test_inspect_stl_flags_a_non_watertight_mesh(tmp_path):
    p = tmp_path / "open.stl"
    _write_binary_stl(p, [(0, 1, 2)], [(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    info = inspect_stl(p)
    assert info["triangles"] == 1 and info["watertight"] is False
    assert info["open_edges"] == 3


# -- 3D domain ----------------------------------------------------------------

def test_model3d_prompts_and_brief():
    dom = Model3DDomain()
    assert dom.build_brief("x", {}).question         # too vague -> asks
    assert not dom.build_brief("a bracket for a motor", {}).question
    assert "cadquery" in dom.system_prompt().lower()
    assert "model.stl" in dom.system_prompt() and "model.step" in dom.system_prompt()
    from studio.domain import Brief
    p = dom.user_prompt(Brief(description="a gear"), "prior=code", "BOOM")
    assert "a gear" in p and "prior=code" in p and "BOOM" in p


def test_model3d_execute_reports_missing_library(tmp_path):
    # cadquery is genuinely absent here -> honest install message, no crash.
    out = _run(Model3DDomain().execute("code", tmp_path, FakeSandbox()))
    assert out.ok is False and "cadquery" in out.stderr and "install" in out.stderr


def test_model3d_execute_detects_the_exported_artifact(tmp_path, monkeypatch):
    _pretend_installed(monkeypatch, "cadquery")
    sb = FakeSandbox(writes={"model.stl": "solid", "model.step": "ISO"})
    out = _run(Model3DDomain().execute("code", tmp_path, sb))
    assert out.ok and set(out.artifacts) == {"model.stl", "model.step"}
    assert out.preview == "model.stl"


def test_model3d_execute_fails_when_no_artifact(tmp_path, monkeypatch):
    _pretend_installed(monkeypatch, "cadquery")
    out = _run(Model3DDomain().execute("code", tmp_path, FakeSandbox(writes={})))
    assert out.ok is False and "exported no" in out.stderr.lower()


def test_model3d_check_watertight_vs_not(tmp_path):
    _cube_stl(tmp_path / "model.stl")
    adv = Model3DDomain().check(ExecOutcome(ok=True, artifacts=["model.stl"]),
                                tmp_path)
    assert any(a.level == "info" and "watertight" in a.message for a in adv)
    _write_binary_stl(tmp_path / "model.stl", [(0, 1, 2)],
                      [(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    adv2 = Model3DDomain().check(ExecOutcome(ok=True, artifacts=["model.stl"]),
                                 tmp_path)
    assert any(a.level == "fail" for a in adv2)   # not shipped as print-ready


# -- schematic domain ---------------------------------------------------------

def test_schematic_prompts_and_missing_library(tmp_path):
    dom = SchematicDomain()
    assert "schemdraw" in dom.system_prompt().lower()
    out = _run(dom.execute("code", tmp_path, FakeSandbox()))
    assert out.ok is False and "schemdraw" in out.stderr


def test_schematic_check_flags_missing_resistor_and_ground(tmp_path):
    (tmp_path / "schematic.py").write_text(
        "d += elm.LED()\n", encoding="utf-8")     # LED, no resistor, no ground
    adv = SchematicDomain().check(ExecOutcome(ok=True), tmp_path)
    msgs = " ".join(a.message.lower() for a in adv)
    assert "resistor" in msgs and "ground" in msgs


# -- routing skills -----------------------------------------------------------

def test_model3d_skill_matches_the_screw_request():
    from core.config import load_settings
    from skills.maker_studio import Model3DSkill
    s = Model3DSkill(load_settings())
    m = s.match("Create me a 3d model of a screw")
    assert m is not None and "screw" in m.groupdict()["desc"].lower()


def test_circuit_skill_matches_and_extracts():
    from core.config import load_settings
    from skills.maker_studio import DesignCircuitSkill
    s = DesignCircuitSkill(load_settings())
    m = s.match("design a circuit for an ESP32 blinking an LED")
    assert m is not None and "esp32" in m.groupdict()["desc"].lower()


def test_studio_skill_off_by_default():
    from core.config import load_settings
    from skills.maker_studio import Model3DSkill
    s = load_settings()
    assert s.studio.enabled is False
    r = _run(Model3DSkill(s).execute(
        SkillRequest(text="", args={"description": "a bracket"})))
    assert r.success is False and r.data.get("reason") == "disabled"


def test_studio_skill_reports_a_successful_build(tmp_path, monkeypatch):
    from core.config import load_settings
    from skills.maker_studio import Model3DSkill
    from studio.domain import Advisory
    from studio.engine import StudioResult

    class FakeProject:
        root = tmp_path

    class FakeEngine:
        async def create(self, domain, desc):
            self.desc = desc
            return StudioResult(ok=True, project=FakeProject(), version=1,
                                artifacts=["model.stl", "model.step"],
                                advisories=[Advisory("info", "watertight · 120 tris")])

    monkeypatch.setattr("core.platform.open_path", lambda p: None)
    s = load_settings()
    s.studio.enabled = True
    skill = Model3DSkill(s, engine=FakeEngine())
    r = _run(skill.execute(SkillRequest(text="", args={"description": "a gear"})))
    assert r.success and "model.stl" in r.speech and "watertight" in r.speech


# -- code domain (S4) — real sandboxed run verifies it ------------------------

def test_code_domain_runs_and_self_verifies(tmp_path):
    from studio.domains.code import CodeDomain
    from studio.sandbox import Sandbox
    (tmp_path / "main.py").write_text("assert 1 + 1 == 2\nprint('ok')\n",
                                      encoding="utf-8")
    out = _run(CodeDomain().execute("", tmp_path, Sandbox()))
    assert out.ok and out.artifacts == ["main.py"] and "ok" in out.stdout


def test_code_domain_fails_when_the_script_raises(tmp_path):
    from studio.domains.code import CodeDomain
    from studio.sandbox import Sandbox
    (tmp_path / "main.py").write_text("assert False, 'boom'\n", encoding="utf-8")
    out = _run(CodeDomain().execute("", tmp_path, Sandbox()))
    assert out.ok is False and "boom" in out.stderr


def test_code_domain_check_flags_missing_self_tests(tmp_path):
    from studio.domains.code import CodeDomain
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
    assert any(a.level == "warn"
               for a in CodeDomain().check(ExecOutcome(ok=True), tmp_path))
    (tmp_path / "main.py").write_text("assert True\n", encoding="utf-8")
    assert any(a.level == "info"
               for a in CodeDomain().check(ExecOutcome(ok=True), tmp_path))


def test_code_skill_matches_and_extracts():
    from core.config import load_settings
    from skills.maker_studio import CodeBuildSkill
    m = CodeBuildSkill(load_settings()).match(
        "write me a script that renames files by date")
    assert m is not None and "renames files" in m.groupdict()["desc"].lower()


# -- studio projects (S5) -----------------------------------------------------

def test_studio_projects_skill_lists_and_handles_empty(tmp_path):
    from core.config import load_settings
    from skills.maker_studio import StudioProjectsSkill
    s = load_settings()
    s.studio.projects_dir = str(tmp_path)
    skill = StudioProjectsSkill(s)
    r = _run(skill.execute(SkillRequest(text="show my studio projects",
                                        match=skill.match("show my studio projects"))))
    assert r.success is False and r.data.get("count") == 0
    # once a project exists, it lists it
    from studio.projects import StudioProject
    StudioProject.create(tmp_path, "model3d", "a gear")
    r2 = _run(skill.execute(SkillRequest(text="show my studio projects",
                                         match=skill.match("show my studio projects"))))
    assert r2.success and r2.data.get("count") == 1


# -- fixlist regressions ------------------------------------------------------

def _studio_settings(tmp_path=None):
    from core.config import load_settings
    s = load_settings()
    s.studio.enabled = True
    if tmp_path is not None:
        s.studio.projects_dir = str(tmp_path)
    return s


class _AskThenBuildEngine:
    """Asks a clarifying question first, then builds — records the brief it got."""

    def __init__(self, question="What should I model?"):
        self._question = question
        self.briefs = []

    async def create(self, domain, desc):
        from studio.engine import StudioResult
        self.briefs.append(desc)
        if self._question:
            q, self._question = self._question, ""
            return StudioResult(ok=False, question=q)
        return StudioResult(ok=True, project=None, version=1, artifacts=["model.stl"])


def test_clarifying_question_captures_the_reply(tmp_path):
    """The answer to 'What should I model?' must reach the engine, not vanish."""
    from skills.maker_studio import Model3DSkill
    eng = _AskThenBuildEngine()
    skill = Model3DSkill(_studio_settings(), engine=eng)
    first = _run(skill.execute(SkillRequest(text="model a", args={"description": "a"})))
    assert first.await_reply is True and "model" in first.speech.lower()
    second = _run(skill.execute(
        SkillRequest(text="a gear", context={"captured_reply": True})))
    assert second.success                       # it built, instead of dead-ending
    assert any("gear" in b for b in eng.briefs)  # the answer reached the engine


def test_clarifying_reply_can_be_cancelled():
    from skills.maker_studio import Model3DSkill
    skill = Model3DSkill(_studio_settings(), engine=_AskThenBuildEngine())
    _run(skill.execute(SkillRequest(text="model a", args={"description": "a"})))
    r = _run(skill.execute(
        SkillRequest(text="never mind", context={"captured_reply": True})))
    assert "never mind" in r.speech.lower() and r.await_reply is False


def test_missing_description_asks_and_captures():
    from skills.maker_studio import Model3DSkill
    r = _run(Model3DSkill(_studio_settings()).execute(SkillRequest(text="")))
    assert r.await_reply is True                # was a dead end before


@pytest.mark.parametrize("text", [
    # idiom-prone nouns after a SOFT verb — English can't be regexed apart here
    # ("a case for my phone" vs "a case for my promotion"), so these go to the
    # LLM, which has the tool and can judge from meaning.
    "make a case for hiring more engineers",
    "make a stand against corruption",
    "let me make a case for it",
    "design a business case",
    "make a compelling case for the merger",
    "make my case for a raise",
    "make a test case",
    "make a git hook that runs tests",
    "make a text box in the form",
    "make a handle on the situation",
    "build a use case diagram",
    # part words whose head noun isn't physical here
    "make a video clip of that",
    "create a short clip for instagram",
    "design a data adapter for the API",
    "create a mock adapter",
    # fixed non-physical compounds, even after a fabrication verb
    "print a case study for the client",
    "model a stand-alone system",
    "model a stand-in for the actor",
    "create a box office report",
    "design a stand-up meeting",
    "print a report",
])
def test_3d_fast_path_ignores_idioms(text):
    from skills.maker_studio import Model3DSkill
    assert Model3DSkill(_studio_settings()).match(text) is None, text


@pytest.mark.parametrize("text", [
    # a fabrication verb IS the physical cue — any part noun, modifier allowed
    "print a case for my ESP32",
    "model a box",
    "print a knob",
    "print a wall mount",
    "model a battery holder",
    "print a strong hook for my wall",
    "model a phone stand",
    "print a camera mount",
    # soft verbs still fast-path an unambiguous part noun
    "design a bracket for a motor",
    "make me a gear",
    "create an enclosure for an esp32",
    # an explicit 3D cue always wins
    "3d print a stand",
    "3d model of a screw",
])
def test_3d_fast_path_still_matches_real_parts(text):
    from skills.maker_studio import Model3DSkill
    assert Model3DSkill(_studio_settings()).match(text) is not None, text


def test_3d_fast_path_has_no_catastrophic_backtracking():
    import time as _t
    from skills.maker_studio import Model3DSkill
    skill = Model3DSkill(_studio_settings())
    t0 = _t.time()
    skill.match("make a " + "word " * 300 + "case for something")
    assert _t.time() - t0 < 1.0


def test_projects_listing_is_not_pc_control(tmp_path):
    """Listing changes nothing, so the PC-control switch must not block it."""
    from skills.maker_studio import StudioProjectsSkill
    from studio.projects import StudioProject
    s = _studio_settings(tmp_path)
    s.safety.pc_control_enabled = False
    StudioProject.create(tmp_path, "model3d", "a gear")
    skill = StudioProjectsSkill(s)
    assert skill.controls_pc is False
    r = _run(skill.execute(SkillRequest(text="show my studio projects",
                                        match=skill.match("show my studio projects"))))
    assert r.success and r.data.get("count") == 1


def test_projects_open_respects_pc_control_and_llm_arg(tmp_path, monkeypatch):
    from skills.maker_studio import StudioProjectsSkill
    from studio.projects import StudioProject
    opened = []
    monkeypatch.setattr("core.platform.open_path", lambda p: opened.append(str(p)))
    s = _studio_settings(tmp_path)
    StudioProject.create(tmp_path, "model3d", "a gear")
    skill = StudioProjectsSkill(s)
    # PC control OFF -> says where it is, opens nothing
    s.safety.pc_control_enabled = False
    r = _run(skill.execute(SkillRequest(text="", args={"open": True})))
    assert opened == [] and r.data.get("opened") is False
    # PC control ON, LLM tool path (no regex match) -> actually opens
    s.safety.pc_control_enabled = True
    r2 = _run(skill.execute(SkillRequest(text="", args={"open": True})))
    assert opened and r2.data.get("opened") is True


def test_orphan_version_dirs_are_swept(tmp_path):
    """A crash between begin_version and finalize leaves a vN dir behind."""
    import os
    import time as _t
    from studio.projects import StudioProject
    p = StudioProject.create(tmp_path, "model3d", "a gear")
    n, vdir = p.begin_version()                 # reserved, never finalized
    (vdir / "model.py").write_text("x", encoding="utf-8")
    old = _t.time() - 7200                      # pretend it's two hours stale
    os.utime(vdir, (old, old))
    reopened = StudioProject.open(p.root)
    assert not vdir.exists()                    # orphan removed
    assert reopened.versions() == []


def test_a_live_version_dir_is_never_swept(tmp_path):
    from studio.projects import StudioProject
    p = StudioProject.create(tmp_path, "model3d", "a gear")
    n, vdir = p.begin_version()                 # fresh — a build in flight
    StudioProject.open(p.root)
    assert vdir.exists()                        # too young to sweep


def test_corrupt_metadata_never_wipes_the_version_history(tmp_path):
    """The sweep must not run against RECOVERED metadata: it lists no versions,
    so every real vN looks like an orphan and the whole project gets deleted."""
    import os
    import time as _t
    from studio.projects import StudioProject
    p = StudioProject.create(tmp_path, "model3d", "a gear")
    for _ in range(3):
        n, vdir = p.begin_version()
        (vdir / "model.py").write_text("x", encoding="utf-8")
        p.finalize_version(n, vdir, filename="model.py", artifacts=["model.stl"],
                           advisories=[], ok=True)
        old = _t.time() - 7200
        os.utime(vdir, (old, old))
    (p.root / "project.json").write_text("{ truncated", encoding="utf-8")
    StudioProject.open(p.root)                  # recovery path
    assert [d.name for d in sorted(p.root.glob("v*"))] == ["v1", "v2", "v3"]


def test_project_metadata_is_written_atomically(tmp_path):
    """A truncated project.json is how history got lost in the first place."""
    from studio.projects import StudioProject
    p = StudioProject.create(tmp_path, "model3d", "a gear")
    n, vdir = p.begin_version()
    p.finalize_version(n, vdir, filename="model.py", artifacts=[], advisories=[],
                       ok=True)
    assert not list(p.root.glob("*.tmp"))       # no temp file left behind
    assert StudioProject.open(p.root).versions()[0].n == 1


def test_open_flag_from_the_llm_is_coerced_from_a_string(tmp_path, monkeypatch):
    """Small models emit JSON booleans as strings, and bool("false") is True —
    which opened Explorer against an explicit no."""
    from skills.maker_studio import StudioProjectsSkill
    from studio.projects import StudioProject
    opened = []
    monkeypatch.setattr("core.platform.open_path", lambda p: opened.append(str(p)))
    s = _studio_settings(tmp_path)
    s.safety.pc_control_enabled = True
    StudioProject.create(tmp_path, "model3d", "a gear")
    skill = StudioProjectsSkill(s)
    for falsy in ("false", "no", "0", ""):
        _run(skill.execute(SkillRequest(text="", args={"open": falsy})))
    assert opened == []                         # listed, never opened
    _run(skill.execute(SkillRequest(text="", args={"open": "true"})))
    assert len(opened) == 1


def test_projects_open_refuses_untrusted_provenance(tmp_path, monkeypatch):
    from skills.maker_studio import StudioProjectsSkill
    from studio.projects import StudioProject
    opened = []
    monkeypatch.setattr("core.platform.open_path", lambda p: opened.append(str(p)))
    s = _studio_settings(tmp_path)
    s.safety.pc_control_enabled = True
    StudioProject.create(tmp_path, "model3d", "a gear")
    r = _run(StudioProjectsSkill(s).execute(
        SkillRequest(text="", args={"open": True},
                     context={"provenance": "untrusted"})))
    assert r.success is False and opened == []


def test_clarifying_reply_is_not_polluted_by_a_fragment_brief():
    """The pending brief is a <3-char fragment by construction, so prepending it
    produced project names like 'it — a phone stand for my desk'."""
    from skills.maker_studio import Model3DSkill
    eng = _AskThenBuildEngine()
    skill = Model3DSkill(_studio_settings(), engine=eng)
    _run(skill.execute(SkillRequest(text="3D print it", args={"description": "it"})))
    _run(skill.execute(SkillRequest(text="a phone stand for my desk",
                                    context={"captured_reply": True})))
    assert eng.briefs[-1] == "a phone stand for my desk"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
