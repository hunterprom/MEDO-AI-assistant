"""Building standalone apps: the AppBuilder engine and the MakeAppSkill.

A fake agent (an async fn that writes files into the app dir) exercises the whole
flow — folder creation, file listing, collision handling, the off switch — with
no real CLI. The skill is driven with a fake builder + announcer, file-opening
stubbed, so no app windows pop up.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.app_builder import AppBuild, AppBuildError, AppBuilder
from core.config import AppBuilderConfig
from skills.app_builder_skill import MakeAppSkill
from skills.base import SkillRequest


def _writes(*names):
    async def agent(app_dir: Path, spec: str) -> str:
        for n in names:
            f = app_dir / n
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x", encoding="utf-8")
        return f"wrote {len(names)} file(s)"
    return agent


def _cfg(tmp_path, **over):
    base = dict(enabled=True, apps_dir=str(tmp_path))
    base.update(over)
    return AppBuilderConfig(**base)


# --- the engine ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_scaffolds_a_folder_with_the_files(tmp_path):
    engine = AppBuilder(_cfg(tmp_path), agent=_writes("main.py", "README.md"))
    build = await engine.build("Water Tracker", "track my water intake")
    assert build.ok
    assert build.path.parent == tmp_path and build.path.name == "water-tracker"
    assert set(build.files) == {"main.py", "README.md"}
    assert (build.path / "main.py").exists()


@pytest.mark.asyncio
async def test_disabled_engine_refuses(tmp_path):
    engine = AppBuilder(_cfg(tmp_path, enabled=False), agent=_writes("a.py"))
    with pytest.raises(AppBuildError, match="disabled"):
        await engine.build("x", "do a thing")


@pytest.mark.asyncio
async def test_empty_spec_refuses(tmp_path):
    engine = AppBuilder(_cfg(tmp_path), agent=_writes("a.py"))
    with pytest.raises(AppBuildError, match="description"):
        await engine.build("x", "   ")


@pytest.mark.asyncio
async def test_agent_that_writes_nothing_is_not_ok(tmp_path):
    async def empty_agent(app_dir, spec):
        return "looked around"
    build = await AppBuilder(_cfg(tmp_path), agent=empty_agent).build("x", "a thing")
    assert build.ok is False and "no files" in build.error


@pytest.mark.asyncio
async def test_name_collision_gets_its_own_folder(tmp_path):
    engine = AppBuilder(_cfg(tmp_path), agent=_writes("main.py"))
    a = await engine.build("notes", "take notes")
    b = await engine.build("notes", "take notes again")
    assert a.path != b.path and b.path.name.startswith("notes-")


# --- the skill ----------------------------------------------------------------

class _FakeBuilder:
    def __init__(self, build=None, raise_exc=None):
        self._build = build
        self._raise = raise_exc
        self.calls: list = []

    async def build(self, name, spec):
        self.calls.append((name, spec))
        if self._raise is not None:
            raise self._raise
        return self._build


class _Announcer:
    def __init__(self):
        self.messages: list[str] = []

    async def __call__(self, text):
        self.messages.append(text)


@pytest.fixture(autouse=True)
def _no_open(monkeypatch):
    monkeypatch.setattr("skills.app_builder_skill.open_path", lambda *a, **k: None)


def _build(tmp_path):
    return AppBuild(name="water-tracker", spec="track water",
                    path=tmp_path / "water-tracker", files=["main.py"])


async def _run(skill, text):
    return await skill.execute(SkillRequest(text=text, match=skill.match(text)))


@pytest.mark.asyncio
async def test_skill_starts_a_build_and_announces(tmp_path):
    ann = _Announcer()
    builder = _FakeBuilder(build=_build(tmp_path))
    skill = MakeAppSkill(builder, ann)
    r = await _run(skill, "make an app that tracks my water intake")
    assert r.success and r.data.get("make_app") == "started"
    await skill._task
    assert builder.calls and "water" in builder.calls[0][1]
    assert any("built" in m.lower() for m in ann.messages)


@pytest.mark.asyncio
async def test_bare_request_asks_and_captures(tmp_path):
    builder = _FakeBuilder(build=_build(tmp_path))
    skill = MakeAppSkill(builder, None)
    r = await _run(skill, "make an app")
    assert r.await_reply is True and "what should the app do" in r.speech.lower()
    # the captured follow-up becomes the spec
    r2 = await skill.execute(SkillRequest(
        text="log my expenses", context={"captured_reply": True}))
    assert r2.success and r2.data.get("make_app") == "started"
    await skill._task
    assert builder.calls[0][1] == "log my expenses"


@pytest.mark.asyncio
async def test_captured_cancel(tmp_path):
    skill = MakeAppSkill(_FakeBuilder(), None)
    await _run(skill, "make an app")
    r = await skill.execute(SkillRequest(text="never mind",
                                         context={"captured_reply": True}))
    assert r.success and "never mind" in r.speech.lower() and skill._task is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
