"""The process supervisor — pure, injectable logic (no real subprocesses here).

`spawn` (launch a command → a process handle) and `now` (a monotonic clock) are
injected, so every branch — start, dead-child restart with exponential backoff,
the give-up-and-warn cap, clean teardown — is unit-tested with fakes and no OS
processes. `app.launcher` injects the real `subprocess.Popen` + `time.monotonic`.

Ownership rule (no orphans, no yanking shared services): the supervisor only
manages a child it actually LAUNCHED. A service already running when we started
(a shared Ollama) is probed, never spawned, and never stopped on quit.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Protocol

from app.childspec import ChildSpec

logger = logging.getLogger(__name__)


class Proc(Protocol):
    """The slice of subprocess.Popen the supervisor uses."""

    def poll(self) -> Optional[int]: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: Optional[float] = None) -> int: ...


Spawn = Callable[[List[str]], Proc]
Clock = Callable[[], float]
OnEvent = Callable[[ChildSpec, str], None]


class Supervisor:
    def __init__(self, specs: List[ChildSpec], *, spawn: Spawn, now: Clock,
                 on_error: Optional[OnEvent] = None,
                 on_status: Optional[OnEvent] = None) -> None:
        self._specs = list(specs)
        self._spawn = spawn
        self._now = now
        self._on_error = on_error or (lambda spec, msg: None)
        self._on_status = on_status or (lambda spec, state: None)
        self._procs: dict[str, Optional[Proc]] = {}
        self._restarts: dict[str, list[float]] = {}
        self._next_retry: dict[str, float] = {}
        self._failed: set[str] = set()
        self._stopped = False

    # -- lifecycle ------------------------------------------------------------

    def start_all(self) -> None:
        for spec in self._specs:
            self._start(spec, initial=True)

    def _start(self, spec: ChildSpec, *, initial: bool) -> None:
        # Already serving (a shared Ollama)? Probe it, don't spawn, don't own it.
        if self._safe(spec.health, default=False):
            self._procs.setdefault(spec.name, None)
            self._on_status(spec, "already-running")
            return
        if not self._safe(spec.start_if, default=True):
            self._procs.setdefault(spec.name, None)
            return
        self._procs[spec.name] = self._spawn(spec.command)
        self._on_status(spec, "started" if initial else "restarted")

    def supervise_once(self) -> None:
        """One tick of the watch loop: restart any child WE launched that died,
        with backoff, until it trips the cap (then warn, once, and stop)."""
        if self._stopped:
            return
        for spec in self._specs:
            if spec.name in self._failed:
                continue
            proc = self._procs.get(spec.name)
            if proc is None:
                continue                       # not ours to manage
            if proc.poll() is None:
                continue                       # still alive
            self._on_dead(spec)

    def _on_dead(self, spec: ChildSpec) -> None:
        now = self._now()
        if now < self._next_retry.get(spec.name, 0.0):
            return                             # cooling down between restarts
        stamps = self._restarts.setdefault(spec.name, [])
        stamps[:] = [t for t in stamps if now - t <= spec.restart_window_s]
        stamps.append(now)
        if len(stamps) > spec.max_restarts:
            self._failed.add(spec.name)
            self._procs[spec.name] = None
            self._on_error(spec, self._fail_message(spec))
            self._on_status(spec, "failed")
            logger.error("child %s exceeded %d restarts; giving up",
                         spec.name, spec.max_restarts)
            return
        backoff = spec.backoff_base_s * (2 ** (len(stamps) - 1))
        self._next_retry[spec.name] = now + backoff
        self._procs[spec.name] = self._spawn(spec.command)
        self._on_status(spec, "restarted")
        logger.warning("restarting child %s (attempt %d, backoff %.1fs)",
                       spec.name, len(stamps), backoff)

    def stop_all(self, *, grace_s: float = 5.0) -> None:
        """Terminate every child WE launched, in reverse order — graceful first,
        then kill. Children we never started (shared Ollama) are left alone."""
        self._stopped = True
        for spec in reversed(self._specs):
            proc = self._procs.get(spec.name)
            if proc is None:
                continue
            try:
                proc.terminate()
            except Exception:
                logger.warning("terminate %s failed", spec.name, exc_info=True)
            try:
                proc.wait(timeout=grace_s)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    logger.warning("kill %s failed", spec.name, exc_info=True)
            self._procs[spec.name] = None

    # -- introspection (used by the launcher's tray + browser-open) -----------

    def all_ready(self) -> bool:
        """True when every child reports healthy — the cue to open the HUD."""
        return all(self._safe(s.health, default=False) for s in self._specs)

    def is_failed(self) -> bool:
        return bool(self._failed)

    def failures(self) -> List[str]:
        return sorted(self._failed)

    def managed(self, name: str) -> Optional[Proc]:
        return self._procs.get(name)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _safe(fn: Callable[[], bool], *, default: bool) -> bool:
        try:
            return bool(fn())
        except Exception:            # a flaky probe must never crash the loop
            return default

    @staticmethod
    def _fail_message(spec: ChildSpec) -> str:
        return (f"{spec.label()} keeps stopping, so MEDO couldn't keep it "
                f"running. The technical details are saved to the log file. "
                f"Try restarting MEDO — if it keeps happening, the log will help "
                f"pin down why.")
