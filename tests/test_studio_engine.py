"""The shared Maker Studio engine (S1): brief → code → sandbox-execute →
self-correct → check → version. The brain and the domain are faked, so the whole
pipeline is pinned without a model or a real engineering library.
"""

from __future__ import annotations

import asyncio

from core.config import load_settings
from studio.domain import Advisory, Brief, ExecOutcome
from studio.engine import StudioEngine
from studio.projects import StudioProject, list_projects


class FakeBrain:
    """Returns canned code; records the (system, user) prompts it was given."""

    def __init__(self, code="print('hi')\n"):
        self._code = code
        self.calls = []

    async def write_code(self, system, user):
        self.calls.append((system, user))
        return self._code


class FakeDomain:
    name = "fake"
    filename = "design.py"

    def __init__(self, *, question="", fail_times=0, check_fail=False):
        self._question = question
        self._fail_times = fail_times
        self._check_fail = check_fail
        self.exec_calls = 0
        self.errors_seen = []
        self.prior_seen = []

    def build_brief(self, description, answers):
        return Brief(description=description, question=self._question)

    def system_prompt(self):
        return "SYSTEM"

    def user_prompt(self, brief, prior_code, error):
        self.errors_seen.append(error)
        self.prior_seen.append(bool(prior_code))
        return f"make {brief.description}"

    async def execute(self, code, workdir, sandbox):
        self.exec_calls += 1
        if self.exec_calls <= self._fail_times:
            return ExecOutcome(ok=False, stderr=f"boom {self.exec_calls}")
        (workdir / "out.txt").write_text("ARTIFACT", encoding="utf-8")
        return ExecOutcome(ok=True, artifacts=["out.txt"], stdout="done",
                           preview="out.txt")

    def check(self, outcome, workdir):
        return [Advisory("fail", "wall too thin")] if self._check_fail \
            else [Advisory("warn", "just so you know")]


def _engine(tmp_path, brain=None, **cfg):
    s = load_settings()
    s.studio.enabled = cfg.pop("enabled", True)
    for k, v in cfg.items():
        setattr(s.studio, k, v)
    # a dummy sandbox — FakeDomain.execute doesn't touch it, so no policy build.
    return s, StudioEngine(s, brain=brain or FakeBrain(), sandbox=object(),
                           base_dir=tmp_path)


def _run(coro):
    return asyncio.run(coro)


# -- gating + clarifying question ---------------------------------------------

def test_disabled_refuses(tmp_path):
    _, eng = _engine(tmp_path, enabled=False)
    r = _run(eng.create(FakeDomain(), "a thing"))
    assert r.ok is False and "studio.enabled" in r.error


def test_missing_spec_asks_instead_of_guessing(tmp_path):
    _, eng = _engine(tmp_path)
    r = _run(eng.create(FakeDomain(question="What voltage?"), "a regulator"))
    assert r.ok is False and r.question == "What voltage?"
    assert list_projects(tmp_path) == []            # nothing created yet


# -- generate → execute → version ---------------------------------------------

def test_generate_execute_success_writes_a_versioned_artifact(tmp_path):
    _, eng = _engine(tmp_path)
    dom = FakeDomain()
    r = _run(eng.create(dom, "a bracket"))
    assert r.ok and r.version == 1 and r.attempts == 1 and r.artifacts == ["out.txt"]
    art = r.project.root / "v1" / "out.txt"
    code = r.project.root / "v1" / "design.py"
    assert art.read_text() == "ARTIFACT" and code.exists()
    assert str(r.project.root).startswith(str(tmp_path))    # managed dir


# -- self-correction loop -----------------------------------------------------

def test_self_correction_retries_then_fixes(tmp_path):
    _, eng = _engine(tmp_path)
    dom = FakeDomain(fail_times=2)               # fail twice, succeed on the 3rd
    r = _run(eng.create(dom, "a circuit"))
    assert r.ok and r.attempts == 3
    # the execution error was fed back into the next generation
    assert "boom 1" in dom.errors_seen[1] and "boom 2" in dom.errors_seen[2]


def test_retries_are_bounded_and_failure_is_honest(tmp_path):
    _, eng = _engine(tmp_path, max_retries=3)
    dom = FakeDomain(fail_times=99)              # never succeeds
    r = _run(eng.create(dom, "an impossible thing"))
    assert r.ok is False and r.attempts == 4     # 1 + 3 retries
    assert "boom" in r.error
    v = r.project.latest()
    assert v.ok is False and v.error             # recorded as a failed version


def test_check_failure_is_not_shipped_as_ready(tmp_path):
    _, eng = _engine(tmp_path)
    dom = FakeDomain(check_fail=True)            # executes, but a hard check fails
    r = _run(eng.create(dom, "a thin part"))
    assert r.ok is False                         # produced, but NOT called ready
    assert any(a.level == "fail" for a in r.advisories)
    assert (r.project.root / "v1" / "out.txt").exists()   # artifact still saved


# -- iteration + version history ----------------------------------------------

def test_iterate_makes_a_new_version_from_prior_code(tmp_path):
    _, eng = _engine(tmp_path)
    dom = FakeDomain()
    first = _run(eng.create(dom, "a mount"))
    second = _run(eng.iterate(dom, first.project, "make the wall 3mm"))
    assert second.version == 2
    assert dom.prior_seen[-1] is True            # iteration saw the prior code
    assert len(StudioProject.open(first.project.root).versions()) == 2


def test_version_history_is_pruned_to_keep(tmp_path):
    _, eng = _engine(tmp_path, keep_versions=2)
    dom = FakeDomain()
    r = _run(eng.create(dom, "a box"))
    for _ in range(3):
        _run(eng.iterate(dom, r.project, "tweak"))
    proj = StudioProject.open(r.project.root)
    assert len(proj.versions()) == 2             # capped
    assert not (proj.root / "v1").exists()       # oldest dirs removed


def test_extract_code_tolerates_crlf_and_trailing_tag_space():
    from studio.brain import extract_code
    assert extract_code("```py \nx=1\n```").strip() == "x=1"        # trailing space
    assert extract_code("```python\r\nx=1\r\n```").strip() == "x=1"  # CRLF endings
    assert extract_code("no fence at all").strip() == "no fence at all"


if __name__ == "__main__":  # pragma: no cover
    import pytest
    pytest.main([__file__, "-v"])
