"""The self-programming engine — MEDO changing its own code, safely.

MEDO can already *be* different brains (``/provider``: Ollama, an OpenAI-compatible
API, Anthropic, or the ``claude``/``codex`` CLI agents). Two of those brains are
full agentic coding agents, so "MEDO fixes its own bugs / adds its own features"
is: drive that coding agent against MEDO's OWN repository, gated by the test suite.

The safety model is **propose & wait**, and it is the whole point of this module:

* Every change happens in an **isolated git worktree** on a fresh branch, off the
  current HEAD — the running MEDO's files are never touched while a proposal is
  built.
* The coding agent only ever **edits files**; the engine runs the tests + lint
  itself, so the agent needs no shell access and a proposal that fails the gate is
  never offered.
* A proposal is **never merged, and MEDO never restarts itself** — :meth:`apply`
  runs only when a human calls it. :meth:`discard` is a one-call rollback.

Disabled by default (:class:`~core.config.SelfDevConfig` ``enabled=False``). The
worktree still runs a coding agent that executes on this machine, so turning it on
is an explicit "yes, MEDO may write code as me" — the isolation bounds the blast
radius, it doesn't remove the trust decision.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import logging
import re
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from core.config import SelfDevConfig

logger = logging.getLogger(__name__)

#: Environment variables scrubbed from the test/lint gate's env — anything that
#: looks like a credential, so agent code running at collection time can't read
#: real secrets out of the environment it inherits.
_SECRET_ENV_RE = re.compile(
    r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|_API|API_|AUTH|SESSION", re.I)

#: What the coding agent is told. It EDITS only — the engine owns commit + tests,
#: so the agent never needs git or a shell (smaller blast radius, cleaner diff).
_AGENT_PROMPT = (
    "You are working inside the MEDO repository (a local voice assistant). Make "
    "the change described below by editing files directly. Match the surrounding "
    "code style, keep the change focused, and add or update tests when it makes "
    "sense. Do NOT run git, do NOT commit, and do NOT push — just leave the edited "
    "files in the working tree; the caller handles version control and testing.\n\n"
    "CHANGE REQUESTED:\n{request}\n"
)


class SelfDevError(Exception):
    """A self-dev step that failed in a way worth reporting verbatim."""


@dataclass
class Proposal:
    """One proposed change: an isolated branch, its diff, and the gate's verdict."""

    request: str
    branch: str
    worktree: Path
    base: str                       # the HEAD sha the branch was cut from
    diff: str = ""                  # git diff base..branch (empty if no change)
    files_changed: list[str] = field(default_factory=list)
    tests_ok: bool = False
    lint_ok: bool = False
    checks_output: str = ""         # tail of the test + lint runs, for review
    agent_output: str = ""          # what the coding agent reported doing
    error: str = ""                 # set when the proposal never reached the gate

    @property
    def ok(self) -> bool:
        """True only when there is a real change that passed every check."""
        return bool(self.diff) and self.tests_ok and self.lint_ok and not self.error

    def summary(self) -> str:
        """A short human-readable verdict for a spoken/printed report."""
        if self.error:
            return f"couldn't propose a change: {self.error}"
        if not self.diff:
            return "the coding agent made no changes"
        n = len(self.files_changed)
        gate = ("passed" if self.ok
                else f"FAILED ({'tests' if not self.tests_ok else 'lint'})")
        return (f"proposed a change to {n} file(s) on {self.branch}; "
                f"the test+lint gate {gate}")


def _slugify(text: str, limit: int = 32) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:limit].rstrip("-")) or "change"


def _path_matches(path: str, patterns: list[str]) -> bool:
    """Repo-relative ``path`` against a glob list; ``dir/**`` matches the subtree."""
    p = path.replace("\\", "/")
    for pat in patterns:
        pat = pat.replace("\\", "/")
        if pat.endswith("/**"):
            prefix = pat[:-3]
            if p == prefix or p.startswith(prefix + "/"):
                return True
        elif fnmatch.fnmatch(p, pat):
            return True
    return False


def _tail(text: str, limit: int = 2000) -> str:
    text = text.strip()
    return text if len(text) <= limit else "…\n" + text[-limit:]


class SelfDevEngine:
    """Propose / apply / discard self-changes on isolated git worktrees.

    ``agent`` is injectable: it's an ``async (worktree: Path, request: str) -> str``
    that edits files in the worktree and returns what it did. The default shells
    out to the configured coding CLI; tests pass a fake so the whole git + test +
    apply flow is exercised without the real agent or the network.
    """

    def __init__(self, config: SelfDevConfig, repo_root: Path, *,
                 agent=None, python_exe: str | None = None,
                 git_exe: str = "git") -> None:
        self._config = config
        self._repo = Path(repo_root)
        self._agent = agent or self._run_cli_agent
        # The interpreter the gate runs under — the running app's is the venv's,
        # which has pytest/ruff and MEDO's deps; the worktree provides the code.
        self._python = python_exe or sys.executable
        self._git = git_exe

    # -- public API ---------------------------------------------------------

    async def propose(self, request: str) -> Proposal:
        """Build a change for ``request`` in an isolated worktree and gate it.

        Never touches the live branch or working tree. On any failure the
        worktree + branch are cleaned up and a :class:`Proposal` with ``error``
        (or a raised :class:`SelfDevError`) is returned.
        """
        request = (request or "").strip()
        if not request:
            raise SelfDevError("a self-dev request can't be empty")
        if not self._config.enabled:
            raise SelfDevError(
                "self-dev is disabled — set self_dev.enabled to allow MEDO to "
                "edit its own code")

        base = (await self._run_git(["rev-parse", "HEAD"])).strip()
        branch = f"{self._config.branch_prefix}/{_slugify(request)}-{uuid.uuid4().hex[:8]}"
        # mkdtemp makes the parent; git worktree add wants to create the leaf.
        parent = Path(tempfile.mkdtemp(prefix="medo-selfdev-"))
        worktree = parent / "tree"
        await self._run_git(["worktree", "add", "-b", branch, str(worktree), base])

        try:
            agent_output = await self._agent(worktree, request)
            await self._run_git(["add", "-A"], cwd=worktree)
            staged = await self._run_git(
                ["diff", "--cached", "--name-only"], cwd=worktree)
            files = [f for f in staged.splitlines() if f.strip()]
            if not files:
                await self._cleanup(branch, parent)
                return Proposal(request, branch, worktree, base,
                                agent_output=agent_output,
                                error="the coding agent made no changes")
            # Hard safety boundary: refuse a proposal that touched anything the
            # denylist protects or the safelist doesn't allow — checked on the
            # ACTUAL changed files, so it holds even if the agent ignored the note.
            violation = self._scope_violation(files)
            if violation:
                logger.warning("self-dev: %s", violation)
                await self._cleanup(branch, parent)
                return Proposal(request, branch, worktree, base, files_changed=files,
                                agent_output=agent_output, error=violation)
            await self._run_git(
                ["commit", "-m", f"self-dev: {request[:60]}", "--no-verify"],
                cwd=worktree)
            diff = await self._run_git(["diff", f"{base}..{branch}"])
            tests_ok, lint_ok, checks = await self._run_checks(worktree, files)
            proposal = Proposal(request, branch, worktree, base, diff, files,
                                tests_ok, lint_ok, checks, agent_output)
            logger.info("self-dev: %s", proposal.summary())
            return proposal
        except Exception as exc:
            await self._cleanup(branch, parent)
            if isinstance(exc, SelfDevError):
                raise
            raise SelfDevError(f"self-dev failed: {exc}") from exc

    async def apply(self, proposal: Proposal) -> bool:
        """Merge an approved proposal into the current branch (no restart).

        Refuses a proposal that didn't pass its gate. Aborts and reports on a
        merge conflict rather than leaving the tree half-merged. The change lands
        on disk; MEDO keeps running its old code until a manual restart.
        """
        if not proposal.ok:
            raise SelfDevError("refusing to apply a proposal that didn't pass its checks")
        code, out, err = await self._run(
            [self._git, "merge", "--no-ff", proposal.branch,
             "-m", f"self-dev: apply {proposal.request[:60]}"], cwd=self._repo)
        if code != 0:
            with contextlib.suppress(Exception):
                await self._run_git(["merge", "--abort"])
            raise SelfDevError(f"could not merge cleanly: {_tail(err or out, 200)}")
        await self._cleanup(proposal.branch, proposal.worktree.parent)
        logger.info("self-dev: applied %s", proposal.branch)
        return True

    async def discard(self, proposal: Proposal) -> None:
        """Roll a proposal back completely: drop its worktree and branch."""
        await self._cleanup(proposal.branch, proposal.worktree.parent)
        logger.info("self-dev: discarded %s", proposal.branch)

    # -- internals ----------------------------------------------------------

    async def _run_checks(self, worktree: Path,
                          files: list[str]) -> tuple[bool, bool, str]:
        parts: list[str] = []
        # The gate IMPORTS and runs the agent's code (pytest collection executes
        # module-level code), so it runs in a scrubbed, HOME/TEMP-isolated
        # environment — no secrets in the env, and a '~'-relative or temp write
        # lands in a throwaway home, not the live repo. sandbox_cmd (if set) wraps
        # it for true OS isolation. Belt-and-braces on top of the denylist, which
        # can only bound the DIFF, not what the code does when executed.
        env = self._gate_env(worktree.parent / "gate-home")
        wrap = list(self._config.sandbox_cmd)
        # Tests run whole: a change must not break anything, anywhere.
        tcode, tout, terr = await self._run(
            [*wrap, self._python, *self._config.test_cmd], cwd=worktree,
            timeout=self._config.check_timeout_s, env=env)
        parts.append("$ tests\n" + _tail(tout + terr))
        # Lint ONLY the files this proposal changed — a good change must not be
        # blocked by pre-existing lint debt elsewhere in the repo.
        py_files = [f for f in files if f.endswith(".py")]
        if py_files:
            lcode, lout, lerr = await self._run(
                [*wrap, self._python, *self._config.lint_cmd, *py_files],
                cwd=worktree, timeout=180.0, env=env)
            parts.append(f"$ lint {' '.join(py_files)}\n" + _tail(lout + lerr))
            lint_ok = lcode == 0
        else:
            parts.append("$ lint (no python files changed — skipped)")
            lint_ok = True
        return tcode == 0, lint_ok, "\n\n".join(parts)

    def _gate_env(self, sandbox_home: Path) -> dict[str, str]:
        """Environment for the test/lint gate: real env minus secrets, with HOME
        and TEMP redirected into a throwaway dir. A payload that runs at
        collection time then can't read credentials from the environment, can't
        persist into the user's real home, and a '~'-relative write (the
        demonstrated self-dev exploit) lands in the throwaway — not the live repo.
        """
        import os

        with contextlib.suppress(OSError):
            sandbox_home.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not _SECRET_ENV_RE.search(k)}
        hp = str(sandbox_home)
        env.update({
            "HOME": hp, "USERPROFILE": hp, "TEMP": hp, "TMP": hp, "TMPDIR": hp,
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        })
        return env

    def _scope_violation(self, files: list[str]) -> str:
        """Empty if every changed file is in scope, else why it's refused."""
        denied = sorted(f for f in files
                        if _path_matches(f, self._config.denylist))
        if denied:
            return "would change protected files (denylist): " + ", ".join(denied)
        if self._config.safelist:
            outside = sorted(f for f in files
                             if not _path_matches(f, self._config.safelist))
            if outside:
                return "would change files outside the safelist: " + ", ".join(outside)
        return ""

    def _scope_note(self) -> str:
        allowed = ", ".join(self._config.safelist) or "(anything)"
        denied = ", ".join(self._config.denylist) or "(nothing)"
        return (f"\nSCOPE — edit ONLY files matching: {allowed}\n"
                f"NEVER edit: {denied}\n")

    async def _run_cli_agent(self, worktree: Path, request: str) -> str:
        """Drive the configured coding CLI to edit files in ``worktree``."""
        prompt = _AGENT_PROMPT.format(request=request) + self._scope_note()
        engine = self._config.engine
        if engine == "claude-code":
            argv = ["claude", "-p", prompt,
                    "--permission-mode", self._config.permission_mode,
                    "--output-format", "text"]
        elif engine == "codex":
            argv = ["codex", "exec", "--skip-git-repo-check", prompt]
        else:
            raise SelfDevError(f"unknown self-dev engine {engine!r}")
        code, out, err = await self._run(
            argv, cwd=worktree, timeout=self._config.agent_timeout_s)
        if code != 0:
            raise SelfDevError(
                f"the coding agent exited {code}: {_tail(err or out, 300)}")
        return out.strip()

    async def _cleanup(self, branch: str, tmp_parent: Path,
                       keep_branch: bool = False) -> None:
        # Worktree first (it holds the branch checked out), then the branch, then
        # the temp dir. Best-effort: cleanup must never mask the real result.
        with contextlib.suppress(Exception):
            await self._run_git(["worktree", "remove", "--force", str(tmp_parent / "tree")])
        if not keep_branch:
            with contextlib.suppress(Exception):
                await self._run_git(["branch", "-D", branch])
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_parent, ignore_errors=True)

    async def _run_git(self, args: list[str], cwd: Path | None = None) -> str:
        code, out, err = await self._run(
            [self._git, *args], cwd=cwd or self._repo, timeout=120.0)
        if code != 0:
            raise SelfDevError(f"git {args[0]} failed: {_tail(err or out, 200)}")
        return out

    async def _run(self, argv: list[str], cwd: Path,
                   timeout: float = 120.0,
                   env: dict[str, str] | None = None) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(cwd), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            raise SelfDevError(
                f"'{argv[0]} {argv[1] if len(argv) > 1 else ''}' timed out "
                f"after {timeout:.0f}s") from None
        return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


def _amain() -> None:  # pragma: no cover - manual, real-agent trial
    """`python -m core.self_dev "<request>"` — build a proposal for real.

    Running this IS the opt-in, so it enables self-dev for this invocation. It
    leaves the branch + worktree in place for you to inspect; apply or discard
    from a REPL with the printed branch name.
    """
    import sys as _sys

    from core.config import PROJECT_ROOT, load_settings

    request = " ".join(_sys.argv[1:]).strip()
    if not request:
        print('usage: python -m core.self_dev "fix the bug where ..."')
        raise SystemExit(2)
    settings = load_settings()
    settings.self_dev.enabled = True
    engine = SelfDevEngine(settings.self_dev, PROJECT_ROOT)

    async def _go() -> None:
        proposal = await engine.propose(request)
        print("\n=== PROPOSAL ===")
        print(proposal.summary())
        print("branch :", proposal.branch)
        print("files  :", ", ".join(proposal.files_changed) or "(none)")
        print("\n--- checks (tail) ---\n" + proposal.checks_output)
        print("\n--- diff (first 4000 chars) ---\n" + proposal.diff[:4000])
        print("\nReady to apply." if proposal.ok else "\nNOT ready — see above.")

    asyncio.run(_go())


if __name__ == "__main__":  # pragma: no cover
    _amain()
