"""The launcher's supervisor (S1): pure logic, driven by fake procs + clock.

No OS processes are spawned. A FakeProc lets a test 'kill' a child; a list-based
clock lets it step time past the backoff. These pin the S1 contract: the right
children start, a dead child restarts with backoff, a crash-loop trips ONE
friendly error and then stops, and quit terminates only what we launched (a
shared Ollama is left running).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.childspec import ChildSpec
from app.launcher import (
    build_children,
    engine_command,
    ollama_command,
    vision_command,
)
from app.supervisor import Supervisor


class FakeProc:
    def __init__(self, wedged: bool = False):
        self._rc = None
        self._wedged = wedged
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._rc

    def die(self, rc: int = 1):
        self._rc = rc

    def terminate(self):
        self.terminated = True
        if not self._wedged:
            self._rc = -15

    def kill(self):
        self.killed = True
        self._rc = -9

    def wait(self, timeout=None):
        if self._rc is None:
            if self._wedged:
                raise TimeoutError("child ignored terminate")
            self._rc = 0
        return self._rc


def _down():
    return False


def _spec(name="engine", health=_down, **kw):
    return ChildSpec(name, ["run", name], friendly_name=name, health=health, **kw)


def _sup(specs):
    clock = [0.0]
    spawned: list[tuple[list, FakeProc]] = []
    errors: list[tuple[str, str]] = []
    statuses: list[tuple[str, str]] = []

    def spawn(cmd):
        p = FakeProc()
        spawned.append((list(cmd), p))
        return p

    sup = Supervisor(specs, spawn=spawn, now=lambda: clock[0],
                     on_error=lambda s, m: errors.append((s.name, m)),
                     on_status=lambda s, st: statuses.append((s.name, st)))
    return sup, clock, spawned, errors, statuses


# -- start --------------------------------------------------------------------

def test_start_all_launches_each_unhealthy_child():
    sup, _clock, spawned, _e, _st = _sup(
        [_spec("ollama"), _spec("engine"), _spec("vision")])
    sup.start_all()
    assert [cmd[1] for cmd, _ in spawned] == ["ollama", "engine", "vision"]


def test_already_running_child_is_not_launched_or_owned():
    up = ChildSpec("ollama", ["ollama", "serve"], health=lambda: True)
    sup, _clock, spawned, _e, _st = _sup([up, _spec("engine")])
    sup.start_all()
    assert [cmd[1] for cmd, _ in spawned] == ["engine"]     # ollama skipped
    assert sup.managed("ollama") is None
    assert sup.managed("engine") is not None


# -- restart + backoff --------------------------------------------------------

def test_dead_child_is_restarted():
    sup, _clock, spawned, _e, statuses = _sup([_spec("engine")])
    sup.start_all()
    spawned[-1][1].die()
    sup.supervise_once()
    assert len(spawned) == 2                                 # relaunched
    assert ("engine", "restarted") in statuses


def test_live_child_is_left_alone():
    sup, _clock, spawned, _e, _st = _sup([_spec("engine")])
    sup.start_all()
    sup.supervise_once()                                     # still alive
    assert len(spawned) == 1


def test_crash_loop_trips_one_friendly_error_then_stops():
    sup, clock, spawned, errors, _st = _sup(
        [_spec("engine", max_restarts=3, restart_window_s=60.0, backoff_base_s=1.0)])
    sup.start_all()
    for _ in range(6):
        spawned[-1][1].die()          # kill whatever is currently running
        sup.supervise_once()
        clock[0] += 10                # < window (stamps accrue), > backoff (retry ok)
    assert sup.is_failed() and "engine" in sup.failures()
    assert len(errors) == 1 and "keeps stopping" in errors[0][1]
    spawned_before = len(spawned)
    sup.supervise_once()              # failed -> no more launches
    assert len(spawned) == spawned_before


# -- shutdown -----------------------------------------------------------------

def test_stop_all_terminates_managed_and_leaves_shared_ollama():
    up = ChildSpec("ollama", ["ollama", "serve"], health=lambda: True)
    sup, _clock, spawned, _e, _st = _sup([up, _spec("engine")])
    sup.start_all()
    engine_proc = spawned[0][1]
    sup.stop_all()
    assert engine_proc.terminated is True
    assert len(spawned) == 1                                 # ollama never touched


def test_wedged_child_is_killed_after_terminate_times_out():
    clock = [0.0]
    procs: list[FakeProc] = []

    def spawn(cmd):
        p = FakeProc(wedged=True)
        procs.append(p)
        return p

    sup = Supervisor([_spec("engine")], spawn=spawn, now=lambda: clock[0])
    sup.start_all()
    sup.stop_all(grace_s=0.01)
    assert procs[0].terminated and procs[0].killed


def test_supervise_is_a_no_op_after_stop():
    sup, _clock, spawned, _e, _st = _sup([_spec("engine")])
    sup.start_all()
    sup.stop_all()
    spawned[-1][1].die()
    sup.supervise_once()
    assert len(spawned) == 1                                 # stopped -> no restart


def test_all_ready_reflects_health():
    up = {"v": False}
    spec = ChildSpec("engine", ["x"], health=lambda: up["v"])
    sup = Supervisor([spec], spawn=lambda c: FakeProc(), now=lambda: 0.0)
    assert sup.all_ready() is False
    up["v"] = True
    assert sup.all_ready() is True


# -- command building (dev checkout form) -------------------------------------

def test_engine_command_is_the_dev_engine_invocation():
    cmd = engine_command()
    assert cmd[-3:] == ["--voice", "--hud", "--serve"]
    assert any(str(p).endswith("main.py") for p in cmd)


def test_vision_command_targets_the_sidecar_module():
    assert vision_command()[-2:] == ["-m", "vision.run"]


def test_ollama_command_is_serve():
    assert ollama_command() == ["ollama", "serve"]


def test_build_children_yields_the_three_processes():
    s = SimpleNamespace(
        llm=SimpleNamespace(host="http://127.0.0.1:11434"),
        hud=SimpleNamespace(port=8730),
        vision=SimpleNamespace(stream_port=8731))
    kids = build_children(s)
    assert [k.name for k in kids] == ["ollama", "engine", "vision"]
    assert all(callable(k.health) for k in kids)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
