"""The self-programming engine: propose in isolation, gate on tests, apply/rollback.

Uses a real throwaway git repo and a FAKE coding agent (an async fn that edits the
worktree), plus deterministic gate commands, so the whole git + gate + apply flow
is exercised without the real CLI agent or the network — the same discipline as
the web-fetch tests mocking httpx. The safety invariants under test:

* propose() never touches the live working tree — the change lives on a branch.
* a proposal that fails the gate is not ``ok`` and apply() refuses it.
* apply() lands the change on the live branch; discard() removes every trace.
* self-dev disabled -> propose() refuses to run at all.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from core.config import SelfDevConfig
from core.self_dev import SelfDevEngine, SelfDevError


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init")
    _git(r, "config", "user.email", "test@example.com")
    _git(r, "config", "user.name", "MEDO Test")
    (r / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(r, "add", "-A")
    _git(r, "commit", "-m", "init")
    return r


def _config(**over) -> SelfDevConfig:
    # Deterministic gate: the "python" just exits 0/1, so the flow doesn't depend
    # on pytest/ruff behaving inside a throwaway repo. Scope checks off by default
    # here (safelist/denylist empty) — they get their own dedicated tests.
    base = dict(enabled=True, test_cmd=["-c", "print('tests ok')"],
                lint_cmd=["-c", "print('lint ok')"], safelist=[], denylist=[])
    base.update(over)
    return SelfDevConfig(**base)


def _adds_file(name: str = "feature.py", body: str = "ADDED = True\n"):
    async def agent(worktree: Path, request: str) -> str:
        target = worktree / name
        target.parent.mkdir(parents=True, exist_ok=True)   # allow nested paths
        target.write_text(body, encoding="utf-8")
        return f"added {name}"
    return agent


def _engine(repo: Path, config: SelfDevConfig, agent) -> SelfDevEngine:
    return SelfDevEngine(config, repo, agent=agent, python_exe=sys.executable)


# --- gate hardening: run agent code with no secrets and an isolated home -----

@pytest.mark.asyncio
async def test_proposal_touching_a_pytest_hook_is_refused(repo):
    # conftest.py runs at pytest collection for ANY test, so an agent must not be
    # able to smuggle host code in through one — the default denylist blocks it.
    cfg = _config(denylist=SelfDevConfig().denylist, safelist=["**"])
    engine = _engine(repo, cfg, _adds_file("conftest.py", "import os\n"))
    proposal = await engine.propose("add a conftest")
    assert not proposal.ok
    assert "denylist" in (proposal.error or "") and "conftest" in (proposal.error or "")


def test_gate_env_scrubs_secrets_and_isolates_home(repo, tmp_path, monkeypatch):
    # The gate runs the agent's code; its env must carry no credentials, and HOME
    # /TEMP must point at a throwaway so a '~'-relative write can't hit the repo.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    engine = _engine(repo, _config(), _adds_file())
    home = tmp_path / "gate-home"
    env = engine._gate_env(home)
    assert "OPENAI_API_KEY" not in env and "GITHUB_TOKEN" not in env
    assert env["HOME"] == str(home) and env["TEMP"] == str(home)
    assert "PATH" in env                      # non-secret vars kept so tools run
    assert home.is_dir()                       # throwaway home was created


# --- propose is isolated and gated -------------------------------------------

@pytest.mark.asyncio
async def test_propose_builds_an_isolated_passing_proposal(repo):
    engine = _engine(repo, _config(), _adds_file())
    proposal = await engine.propose("add a feature flag")

    assert proposal.files_changed == ["feature.py"]
    assert "ADDED = True" in proposal.diff
    assert proposal.tests_ok and proposal.lint_ok and proposal.ok
    # The live working tree was NEVER touched — the change is only on the branch.
    assert not (repo / "feature.py").exists()
    assert proposal.branch in _git(repo, "branch", "--list", proposal.branch)


@pytest.mark.asyncio
async def test_failing_gate_makes_the_proposal_not_ok_and_apply_refuses(repo):
    engine = _engine(repo, _config(test_cmd=["-c", "import sys; sys.exit(1)"]),
                     _adds_file())
    proposal = await engine.propose("add something that fails tests")

    assert proposal.tests_ok is False and proposal.ok is False
    assert "ADDED = True" in proposal.diff          # the change still exists to review
    with pytest.raises(SelfDevError):
        await engine.apply(proposal)                # ...but is never merged
    assert not (repo / "feature.py").exists()


@pytest.mark.asyncio
async def test_agent_that_changes_nothing_is_reported(repo):
    async def noop_agent(worktree, request):
        return "looked around, changed nothing"

    engine = _engine(repo, _config(), noop_agent)
    proposal = await engine.propose("do nothing")
    assert proposal.ok is False
    assert "no changes" in proposal.error
    # No stray branch left behind.
    assert proposal.branch not in _git(repo, "branch", "--list", proposal.branch)


# --- apply / discard ----------------------------------------------------------

@pytest.mark.asyncio
async def test_apply_lands_the_change_on_the_live_branch(repo):
    engine = _engine(repo, _config(), _adds_file())
    proposal = await engine.propose("add feature.py")
    assert await engine.apply(proposal) is True

    # The change is now in the real working tree + history; the temp worktree and
    # the proposal branch are gone.
    assert (repo / "feature.py").read_text(encoding="utf-8") == "ADDED = True\n"
    assert "self-dev: apply" in _git(repo, "log", "-1", "--pretty=%s")
    assert proposal.branch not in _git(repo, "branch", "--list", proposal.branch)
    assert not proposal.worktree.exists()


@pytest.mark.asyncio
async def test_discard_removes_every_trace(repo):
    engine = _engine(repo, _config(), _adds_file())
    proposal = await engine.propose("add feature.py")
    await engine.discard(proposal)

    assert not (repo / "feature.py").exists()
    assert proposal.branch not in _git(repo, "branch", "--list", proposal.branch)
    assert not proposal.worktree.exists()


# --- the gate lints only what changed ----------------------------------------

@pytest.mark.asyncio
async def test_lint_gate_ignores_preexisting_debt_in_untouched_files(repo):
    # A real lint violation (unused import) in a file the proposal never touches.
    (repo / "legacy.py").write_text("import os\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "pre-existing lint debt")

    # Real ruff this time; the agent adds a CLEAN file.
    engine = _engine(repo, _config(lint_cmd=["-m", "ruff", "check"]),
                     _adds_file("clean.py", "ADDED = True\n"))
    proposal = await engine.propose("add a clean module")

    # legacy.py's F401 must NOT fail this proposal — only clean.py is linted.
    assert proposal.lint_ok is True and proposal.ok is True


# --- the safety boundary: safelist / denylist --------------------------------

@pytest.mark.asyncio
async def test_denylist_refuses_a_protected_file(repo):
    engine = _engine(repo, _config(denylist=["core/safety.py"]),
                     _adds_file("core/safety.py", "HACKED = True\n"))
    proposal = await engine.propose("mess with the safety module")
    assert proposal.ok is False
    assert "denylist" in proposal.error and "core/safety.py" in proposal.error
    # refused proposals leave nothing behind
    assert proposal.branch not in _git(repo, "branch", "--list", proposal.branch)
    assert not (repo / "core" / "safety.py").exists()


@pytest.mark.asyncio
async def test_safelist_refuses_an_out_of_scope_file(repo):
    engine = _engine(repo, _config(safelist=["skills/**"]),
                     _adds_file("core/newthing.py", "X = 1\n"))
    proposal = await engine.propose("add a core module")
    assert proposal.ok is False
    assert "safelist" in proposal.error


@pytest.mark.asyncio
async def test_safelist_allows_an_in_scope_file(repo):
    engine = _engine(repo, _config(safelist=["skills/**"], denylist=[]),
                     _adds_file("skills/newthing.py", "ADDED = True\n"))
    proposal = await engine.propose("add a new skill")
    assert proposal.ok is True and proposal.files_changed == ["skills/newthing.py"]


# --- the off switch -----------------------------------------------------------

@pytest.mark.asyncio
async def test_disabled_engine_refuses_to_run(repo):
    engine = _engine(repo, _config(enabled=False), _adds_file())
    with pytest.raises(SelfDevError, match="disabled"):
        await engine.propose("do anything")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
