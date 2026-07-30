"""Restricted-subprocess sandbox for Maker Studio's generated code.

Generated code is UNTRUSTED — the model wrote it. It runs in its own project
working directory, with a scrubbed environment (secrets stripped; HOME/TEMP
redirected into the project so a '~'-relative or temp write can't escape), a hard
timeout, and — before it runs — a deny-by-default policy check
(security/policy.py). An optional OS-isolation wrapper (studio.sandbox_cmd) adds
true isolation (e.g. no network) where the OS supports it.

Honest limit: without ``sandbox_cmd`` this is process + env + cwd isolation —
strong for the filesystem and secrets, but it does NOT hard-block network. It
reuses core/self_dev.py's proven gate-environment approach.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

#: Start generated code in its own process group so a timeout kills the tree.
_NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

#: Env vars that look like credentials — never handed to generated code.
#: Best-effort by NAME: also catches connection strings that carry inline creds
#: (DATABASE_URL, DSN, CONNECTION_STRING) and session/cookie material.
_SECRET_ENV_RE = re.compile(
    r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|PASS|CRED|AUTH|DSN|DATABASE_URL|"
    r"CONNECTION|POSTGRES|MYSQL|MONGO|REDIS|SESSION|COOKIE|PRIVATE|API",
    re.I)


@dataclass
class SandboxResult:
    ok: bool
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    denied: str = ""          # non-empty when the policy engine refused to run it


def _scrubbed_env(home: Path) -> dict:
    """Real env minus secrets, with HOME/TEMP redirected into a throwaway dir."""
    with contextlib.suppress(OSError):
        home.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV_RE.search(k)}
    hp = str(home)
    env.update({
        "HOME": hp, "USERPROFILE": hp, "TEMP": hp, "TMP": hp, "TMPDIR": hp,
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
    })
    return env


class Sandbox:
    """Runs a command list in a scrubbed, cwd-isolated, time-bounded subprocess,
    gated by the policy engine. ``policy`` is optional (None = no gate, for
    tests)."""

    def __init__(self, *, policy=None, actor=None, sandbox_cmd: Sequence[str] = (),
                 timeout_s: float = 60.0) -> None:
        self._policy = policy
        self._actor = actor
        self._wrap = list(sandbox_cmd)
        self._timeout = timeout_s

    def _gate(self, subject: str) -> str:
        """'' if allowed to run generated code, else the deny reason. Fail-closed:
        any error in the check denies."""
        if self._policy is None:
            return ""
        try:
            from security.capabilities import Capability
            from security.policy import ActionRequest, Actor, Provenance

            decision = self._policy.check(ActionRequest(
                actor=self._actor or Actor.MODEL,
                capability=Capability.RUN_COMMAND,
                provenance=Provenance.UNTRUSTED, subject=subject,
                declared=frozenset({Capability.RUN_COMMAND,
                                    Capability.WRITE_FILES})))
            return "" if not decision.denied() else decision.reason
        except Exception:
            logger.warning("studio policy check failed — denying", exc_info=True)
            return "the policy engine couldn't authorise running generated code"

    async def run(self, argv: Sequence[object], cwd: Path, *,
                  timeout_s: Optional[float] = None) -> SandboxResult:
        cwd = Path(cwd)
        cwd.mkdir(parents=True, exist_ok=True)
        argv = [str(a) for a in argv]
        denied = self._gate(f"studio-exec: {' '.join(argv)[:100]}")
        if denied:
            return SandboxResult(False, stderr=denied, denied=denied)
        env = _scrubbed_env(cwd / ".sandbox-home")
        full = [*self._wrap, *argv]
        # Start in its OWN process group so a timeout can kill the whole TREE
        # (generated code may spawn children) — mirrors app/launcher.py's leak fix.
        group_kwargs = ({"creationflags": _NEW_GROUP} if os.name == "nt"
                        else {"start_new_session": True})
        try:
            proc = await asyncio.create_subprocess_exec(
                *full, cwd=str(cwd), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                **group_kwargs)
        except Exception as exc:
            return SandboxResult(False, stderr=f"couldn't start: {exc}")
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s or self._timeout)
        except asyncio.TimeoutError:
            await self._kill_tree(proc)
            return SandboxResult(False, stderr="the code timed out",
                                 timed_out=True)
        return SandboxResult(
            proc.returncode == 0, returncode=proc.returncode,
            stdout=out.decode(errors="replace"),
            stderr=err.decode(errors="replace"))

    async def _kill_tree(self, proc) -> None:
        """Kill the process AND anything it spawned — untrusted code may fork."""
        with contextlib.suppress(Exception):
            if os.name == "nt":
                killer = await asyncio.create_subprocess_exec(
                    "taskkill", "/PID", str(proc.pid), "/T", "/F",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL)
                await asyncio.wait_for(killer.wait(), timeout=5)
            else:
                import signal

                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(Exception):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()

    async def run_python(self, script: Path, cwd: Path, *,
                         timeout_s: Optional[float] = None) -> SandboxResult:
        """Run a generated Python script with the SAME interpreter MEDO uses (so
        installed engineering libs are importable), sandboxed."""
        return await self.run([sys.executable, str(script)], cwd,
                              timeout_s=timeout_s)
