"""The Studio sandbox: really runs generated code, but bounded, scrubbed, and
policy-gated. Uses the same interpreter with tiny real scripts (fast, no heavy
deps)."""

from __future__ import annotations

import asyncio

from studio.sandbox import Sandbox


def _run(coro):
    return asyncio.run(coro)


def test_runs_a_script_and_captures_output(tmp_path):
    (tmp_path / "s.py").write_text(
        "open('made.txt','w').write('hi'); print('ran ok')", encoding="utf-8")
    r = _run(Sandbox().run_python(tmp_path / "s.py", tmp_path))
    assert r.ok and r.returncode == 0 and "ran ok" in r.stdout
    assert (tmp_path / "made.txt").read_text() == "hi"   # wrote in its own cwd


def test_nonzero_exit_is_reported_not_raised(tmp_path):
    (tmp_path / "s.py").write_text("raise SystemExit(3)", encoding="utf-8")
    r = _run(Sandbox().run_python(tmp_path / "s.py", tmp_path))
    assert r.ok is False and r.returncode == 3


def test_timeout_kills_and_flags(tmp_path):
    (tmp_path / "s.py").write_text("import time; time.sleep(30)", encoding="utf-8")
    r = _run(Sandbox(timeout_s=0.6).run_python(tmp_path / "s.py", tmp_path))
    assert r.ok is False and r.timed_out is True


def test_secrets_are_stripped_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SECRET_TOKEN", "hunter2")
    (tmp_path / "s.py").write_text(
        "import os; print('SEES:', os.environ.get('MY_SECRET_TOKEN', 'none'))",
        encoding="utf-8")
    r = _run(Sandbox().run_python(tmp_path / "s.py", tmp_path))
    assert r.ok and "SEES: none" in r.stdout and "hunter2" not in r.stdout


def test_policy_denial_blocks_execution(tmp_path):
    class _Deny:
        reason = "PC control is off"
        def denied(self):
            return True

    class DenyPolicy:
        def check(self, _request):
            return _Deny()

    (tmp_path / "s.py").write_text("open('made.txt','w').write('x')",
                                   encoding="utf-8")
    r = _run(Sandbox(policy=DenyPolicy()).run_python(tmp_path / "s.py", tmp_path))
    assert r.ok is False and r.denied == "PC control is off"
    assert not (tmp_path / "made.txt").exists()          # never ran
