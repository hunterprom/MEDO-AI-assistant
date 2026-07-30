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


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
