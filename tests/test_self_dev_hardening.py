"""P0-2: what the self-programming engine is not allowed to do.

The gate runs code a coding agent wrote, before any human has read it, and
``apply`` rewrites the repository MEDO is running from. Four claims are made
about that and tested here:

* the gate cannot reach the network,
* a proposal too large to review is refused outright,
* nothing merges without an approval that is written down and bound to the
  exact diff a human saw,
* and a merge that breaks the suite undoes itself.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from core.config import SelfDevConfig
from core.self_dev import SelfDevEngine, SelfDevError
from core.selfdev_approval import (
    APPROVAL_TTL_S,
    Approval,
    ApprovalStore,
    diff_digest,
)


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
    base = dict(enabled=True, test_cmd=["-c", "print('tests ok')"],
                lint_cmd=["-c", "print('lint ok')"], safelist=[], denylist=[],
                verify_after_apply=False)
    base.update(over)
    return SelfDevConfig(**base)


def _writes(files: dict[str, str]):
    async def agent(worktree: Path, request: str) -> str:
        for name, body in files.items():
            target = worktree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        return f"wrote {len(files)} file(s)"
    return agent


def _engine(repo: Path, config: SelfDevConfig, agent, **kw) -> SelfDevEngine:
    return SelfDevEngine(config, repo, agent=agent, python_exe=sys.executable, **kw)


# --- the gate has no network -------------------------------------------------


def _run_guarded(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a subprocess with the gate's network guard installed."""
    import shutil

    guard_dir = tmp_path / "pypath"
    guard_dir.mkdir(exist_ok=True)
    shutil.copyfile(Path("core/gate_netblock.py"), guard_dir / "sitecustomize.py")
    script = tmp_path / "probe.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    env = {"PATH": "", "PYTHONPATH": str(guard_dir), "SYSTEMROOT":
           __import__("os").environ.get("SYSTEMROOT", "")}
    return subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, env=env, timeout=60)


def test_the_gate_cannot_open_an_outbound_connection(tmp_path):
    result = _run_guarded(tmp_path, """
        import socket
        try:
            socket.create_connection(("example.com", 80), timeout=5)
        except OSError as exc:
            print("BLOCKED", exc)
        else:
            print("REACHED THE NETWORK")
    """)
    assert "BLOCKED" in result.stdout, result.stdout + result.stderr
    assert "blocked" in result.stdout


def test_the_gate_cannot_resolve_an_external_name(tmp_path):
    result = _run_guarded(tmp_path, """
        import socket
        try:
            socket.getaddrinfo("example.com", 80)
        except OSError as exc:
            print("BLOCKED", exc)
        else:
            print("RESOLVED")
    """)
    assert "BLOCKED" in result.stdout, result.stdout + result.stderr


def test_the_gate_cannot_connect_a_raw_socket_either(tmp_path):
    result = _run_guarded(tmp_path, """
        import socket
        s = socket.socket()
        try:
            s.connect(("93.184.216.34", 80))
        except OSError as exc:
            print("BLOCKED", exc)
        else:
            print("CONNECTED")
    """)
    assert "BLOCKED" in result.stdout, result.stdout + result.stderr


def test_loopback_still_works_so_the_suite_can_run(tmp_path):
    """MEDO's own tests bind aiohttp servers on 127.0.0.1 — those must survive."""
    result = _run_guarded(tmp_path, """
        import socket, threading
        srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
        port = srv.getsockname()[1]
        threading.Thread(target=lambda: srv.accept(), daemon=True).start()
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            print("LOOPBACK OK")
    """)
    assert "LOOPBACK OK" in result.stdout, result.stdout + result.stderr


def test_the_guard_is_installed_into_the_gate_environment(repo, tmp_path):
    engine = _engine(repo, _config(block_network_in_gate=True), _writes({}))
    env = engine._gate_env(tmp_path / "home")
    assert "PYTHONPATH" in env
    import os

    guard = Path(env["PYTHONPATH"].split(os.pathsep)[0]) / "sitecustomize.py"
    assert guard.exists()
    assert env["https_proxy"].endswith(":9")      # dead port for proxy users


def test_the_guard_can_be_switched_off(repo, tmp_path):
    engine = _engine(repo, _config(block_network_in_gate=False), _writes({}))
    env = engine._gate_env(tmp_path / "home")
    assert "https_proxy" not in env


# --- the diff has a ceiling --------------------------------------------------


@pytest.mark.asyncio
async def test_a_proposal_over_the_byte_limit_is_refused(repo):
    engine = _engine(repo, _config(max_diff_bytes=500),
                     _writes({"big.py": "X = 1\n" * 500}))
    proposal = await engine.propose("write a lot")
    assert not proposal.ok
    assert "over the 500-byte limit" in proposal.error


@pytest.mark.asyncio
async def test_a_proposal_over_the_file_limit_is_refused(repo):
    files = {f"mod{i}.py": f"N = {i}\n" for i in range(6)}
    engine = _engine(repo, _config(max_files_changed=3), _writes(files))
    proposal = await engine.propose("touch everything")
    assert not proposal.ok
    assert "over the 3-file limit" in proposal.error


@pytest.mark.asyncio
async def test_an_oversized_proposal_leaves_nothing_behind(repo):
    engine = _engine(repo, _config(max_diff_bytes=100),
                     _writes({"big.py": "X = 1\n" * 500}))
    proposal = await engine.propose("write a lot")
    assert not proposal.worktree.exists()
    assert proposal.branch not in _git(repo, "branch", "--list", proposal.branch)


@pytest.mark.asyncio
async def test_a_limit_of_zero_means_no_limit(repo):
    engine = _engine(repo, _config(max_diff_bytes=0, max_files_changed=0),
                     _writes({"big.py": "X = 1\n" * 500}))
    proposal = await engine.propose("write a lot")
    assert proposal.ok


# --- a failed proposal cleans up after itself --------------------------------


@pytest.mark.asyncio
async def test_a_failed_gate_removes_the_worktree(repo):
    engine = _engine(repo, _config(test_cmd=["-c", "raise SystemExit(1)"]),
                     _writes({"feature.py": "A = 1\n"}))
    proposal = await engine.propose("add a feature")
    assert not proposal.ok and not proposal.tests_ok
    assert not proposal.worktree.exists()
    assert proposal.branch not in _git(repo, "branch", "--list", proposal.branch)
    # The evidence a human needs to understand the failure survives the cleanup.
    assert proposal.diff and proposal.checks_output


@pytest.mark.asyncio
async def test_a_failed_worktree_can_be_kept_for_debugging(repo):
    engine = _engine(repo, _config(test_cmd=["-c", "raise SystemExit(1)"],
                                   keep_failed_worktree=True),
                     _writes({"feature.py": "A = 1\n"}))
    proposal = await engine.propose("add a feature")
    assert proposal.worktree.exists()


@pytest.mark.asyncio
async def test_stale_branches_from_a_crash_are_reaped(repo):
    engine = _engine(repo, _config(keep_failed_worktree=True,
                                   test_cmd=["-c", "raise SystemExit(1)"]),
                     _writes({"feature.py": "A = 1\n"}))
    proposal = await engine.propose("add a feature")
    # Simulate the crash: the temp worktree vanishes, its registration doesn't.
    import shutil as _shutil

    _shutil.rmtree(proposal.worktree.parent, ignore_errors=True)
    assert await engine.cleanup_stale() >= 1
    assert proposal.branch not in _git(repo, "branch", "--list", proposal.branch)


# --- approval, on disk -------------------------------------------------------


def test_an_approval_survives_a_restart(tmp_path):
    path = tmp_path / "approvals.json"
    ApprovalStore(path).approve("medo/self-dev/x", "diff text", request="fix it")
    reopened = ApprovalStore(path)          # a whole new process would see this
    assert reopened.approval_for("medo/self-dev/x", "diff text") is not None


def test_an_approval_does_not_cover_a_different_diff(tmp_path):
    store = ApprovalStore(tmp_path / "a.json")
    store.approve("branch", "the diff I read")
    assert store.approval_for("branch", "a completely different diff") is None


def test_an_approval_expires(tmp_path):
    store = ApprovalStore(tmp_path / "a.json")
    store.approve("branch", "diff")
    aged = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    aged["branch"]["approved_at"] = time.time() - APPROVAL_TTL_S - 60
    (tmp_path / "a.json").write_text(json.dumps(aged), encoding="utf-8")
    assert store.approval_for("branch", "diff") is None


def test_an_approval_can_be_revoked(tmp_path):
    store = ApprovalStore(tmp_path / "a.json")
    store.approve("branch", "diff")
    assert store.revoke("branch") is True
    assert store.approval_for("branch", "diff") is None
    assert store.revoke("branch") is False


def test_a_corrupt_approvals_file_approves_nothing(tmp_path):
    path = tmp_path / "a.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert ApprovalStore(path).approval_for("branch", "diff") is None


def test_a_partially_corrupt_file_keeps_the_readable_records(tmp_path):
    path = tmp_path / "a.json"
    good = Approval("good", diff_digest("d"), "r", "human", time.time())
    path.write_text(json.dumps({"good": good.to_json(), "bad": "nonsense"}),
                    encoding="utf-8")
    store = ApprovalStore(path)
    assert store.approval_for("good", "d") is not None
    assert store.approval_for("bad", "d") is None


def test_pending_lists_live_approvals_newest_first(tmp_path):
    store = ApprovalStore(tmp_path / "a.json")
    store.approve("one", "d1")
    time.sleep(0.01)
    store.approve("two", "d2")
    assert [a.branch for a in store.pending()] == ["two", "one"]


# --- approval, enforced by the engine ----------------------------------------


@pytest.mark.asyncio
async def test_apply_refuses_a_proposal_nobody_approved(repo):
    engine = _engine(repo, _config(), _writes({"feature.py": "A = 1\n"}))
    proposal = await engine.propose("add a feature")
    with pytest.raises(SelfDevError, match="nobody approved"):
        await engine.apply(proposal)
    assert not (repo / "feature.py").exists()


@pytest.mark.asyncio
async def test_apply_proceeds_once_approved(repo):
    engine = _engine(repo, _config(), _writes({"feature.py": "A = 1\n"}))
    proposal = await engine.propose("add a feature")
    engine.approve(proposal)
    assert await engine.apply(proposal) is True
    assert (repo / "feature.py").exists()


@pytest.mark.asyncio
async def test_an_approval_cannot_be_reused_after_the_diff_changes(repo):
    """The recorded yes is for a diff, not for a branch name."""
    engine = _engine(repo, _config(), _writes({"feature.py": "A = 1\n"}))
    proposal = await engine.propose("add a feature")
    engine.approve(proposal)
    proposal.diff += "\n+ something nobody read\n"
    with pytest.raises(SelfDevError, match="nobody approved"):
        await engine.apply(proposal)


@pytest.mark.asyncio
async def test_a_failed_proposal_cannot_be_approved(repo):
    engine = _engine(repo, _config(test_cmd=["-c", "raise SystemExit(1)"]),
                     _writes({"feature.py": "A = 1\n"}))
    proposal = await engine.propose("add a feature")
    with pytest.raises(SelfDevError, match="didn't pass its checks"):
        engine.approve(proposal)


@pytest.mark.asyncio
async def test_auto_apply_records_its_own_approval(repo):
    """The autonomy level is honoured, and it still leaves a trail."""
    engine = _engine(repo, _config(auto_apply=True),
                     _writes({"feature.py": "A = 1\n"}))
    proposal = await engine.propose("add a feature")
    assert await engine.apply(proposal) is True
    recorded = json.loads(engine.approvals.path.read_text(encoding="utf-8"))
    assert any(a["approved_by"] == "auto_apply" for a in recorded.values())


@pytest.mark.asyncio
async def test_discarding_a_proposal_revokes_its_approval(repo):
    engine = _engine(repo, _config(), _writes({"feature.py": "A = 1\n"}))
    proposal = await engine.propose("add a feature")
    engine.approve(proposal)
    await engine.discard(proposal)
    assert engine.approvals.approval_for(proposal.branch, proposal.diff) is None


@pytest.mark.asyncio
async def test_apply_refuses_when_the_tree_has_uncommitted_work(repo):
    engine = _engine(repo, _config(), _writes({"feature.py": "A = 1\n"}))
    proposal = await engine.propose("add a feature")
    engine.approve(proposal)
    (repo / "app.py").write_text("VALUE = 99  # someone's unsaved work\n",
                                 encoding="utf-8")
    with pytest.raises(SelfDevError, match="uncommitted changes"):
        await engine.apply(proposal)
    assert "99" in (repo / "app.py").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_an_untracked_scratch_file_does_not_block_apply(repo):
    """`reset --hard` never touches untracked files, so they are not at risk."""
    engine = _engine(repo, _config(), _writes({"feature.py": "A = 1\n"}))
    proposal = await engine.propose("add a feature")
    engine.approve(proposal)
    (repo / "notes.txt").write_text("scratch\n", encoding="utf-8")
    assert await engine.apply(proposal) is True
    assert (repo / "notes.txt").exists()


# --- rollback verification ---------------------------------------------------


@pytest.mark.asyncio
async def test_a_merge_that_breaks_the_suite_is_rolled_back(repo):
    """The gate passed on the branch; the merged repo fails. Undo the merge."""
    calls = {"n": 0}

    engine = _engine(repo, _config(verify_after_apply=True),
                     _writes({"feature.py": "A = 1\n"}))
    proposal = await engine.propose("add a feature")
    engine.approve(proposal)
    before = _git(repo, "rev-parse", "HEAD").strip()

    real_run = engine._run

    async def failing_verify(argv, cwd, **kw):
        # Only the post-merge verification fails; every git call is untouched.
        if argv[0] == engine._python and Path(cwd) == repo:
            calls["n"] += 1
            return 1, "", "2 failed"
        return await real_run(argv, cwd, **kw)

    engine._run = failing_verify
    with pytest.raises(SelfDevError, match="rolled it back"):
        await engine.apply(proposal)

    assert calls["n"] == 1
    assert _git(repo, "rev-parse", "HEAD").strip() == before
    assert not (repo / "feature.py").exists()


@pytest.mark.asyncio
async def test_a_rolled_back_change_loses_its_approval(repo):
    engine = _engine(repo, _config(verify_after_apply=True),
                     _writes({"feature.py": "A = 1\n"}))
    proposal = await engine.propose("add a feature")
    engine.approve(proposal)
    real_run = engine._run

    async def failing_verify(argv, cwd, **kw):
        if argv[0] == engine._python and Path(cwd) == repo:
            return 1, "", "boom"
        return await real_run(argv, cwd, **kw)

    engine._run = failing_verify
    with pytest.raises(SelfDevError):
        await engine.apply(proposal)
    assert engine.approvals.approval_for(proposal.branch, proposal.diff) is None


@pytest.mark.asyncio
async def test_verification_that_passes_keeps_the_merge(repo):
    engine = _engine(repo, _config(verify_after_apply=True),
                     _writes({"feature.py": "A = 1\n"}))
    proposal = await engine.propose("add a feature")
    engine.approve(proposal)
    assert await engine.apply(proposal) is True
    assert (repo / "feature.py").exists()
    assert "self-dev: apply" in _git(repo, "log", "-1", "--pretty=%s")
