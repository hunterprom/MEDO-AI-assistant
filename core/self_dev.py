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
from core.selfdev_approval import ApprovalStore

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
                 git_exe: str = "git", approvals: ApprovalStore | None = None) -> None:
        self._config = config
        self._repo = Path(repo_root)
        self._agent = agent or self._run_cli_agent
        # The interpreter the gate runs under — the running app's is the venv's,
        # which has pytest/ruff and MEDO's deps; the worktree provides the code.
        self._python = python_exe or sys.executable
        self._git = git_exe
        approvals_path = Path(config.approvals_path)
        if not approvals_path.is_absolute():
            approvals_path = self._repo / approvals_path
        self.approvals = approvals or ApprovalStore(approvals_path)

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
            # Size ceiling, checked BEFORE the gate: an oversized proposal is
            # refused whether or not its tests pass, and there is no reason to
            # spend ten minutes running them to find that out. See
            # SelfDevConfig.max_diff_bytes for why a cap exists at all.
            oversized = self._size_violation(files, diff)
            if oversized:
                logger.warning("self-dev: %s", oversized)
                await self._cleanup(branch, parent)
                return Proposal(request, branch, worktree, base, diff, files,
                                agent_output=agent_output, error=oversized)
            tests_ok, lint_ok, checks = await self._run_checks(worktree, files)
            proposal = Proposal(request, branch, worktree, base, diff, files,
                                tests_ok, lint_ok, checks, agent_output)
            logger.info("self-dev: %s", proposal.summary())
            if not proposal.ok and not self._config.keep_failed_worktree:
                # A proposal that failed its gate can never be applied, so its
                # worktree is litter — and litter that holds a branch checked
                # out, which makes the next `git worktree` operation awkward.
                # Everything a human needs to review the failure (the diff, the
                # tail of the test run) is already in the Proposal.
                await self._cleanup(branch, parent)
            return proposal
        except Exception as exc:
            await self._cleanup(branch, parent)
            if isinstance(exc, SelfDevError):
                raise
            raise SelfDevError(f"self-dev failed: {exc}") from exc

    def approve(self, proposal: Proposal, *, approved_by: str = "human"):
        """Record a human's yes for this exact diff, on disk.

        Separate from :meth:`apply` on purpose. Approval is a decision about a
        diff someone has read; applying is a mechanical step that may happen
        later, from somewhere else, and must be able to check that the decision
        was really made — which an in-memory flag on the object being applied
        could never do.
        """
        if not proposal.ok:
            raise SelfDevError(
                "refusing to approve a proposal that didn't pass its checks")
        return self.approvals.approve(proposal.branch, proposal.diff,
                                      request=proposal.request,
                                      approved_by=approved_by)

    async def apply(self, proposal: Proposal) -> bool:
        """Merge an APPROVED proposal into the current branch (no restart).

        Four gates, in order, and all of them can say no:

        1. The proposal passed its own test + lint run.
        2. A human approved *this diff*, and it is written down
           (``core/selfdev_approval.py``). ``auto_apply`` satisfies this by
           recording the approval itself — a decision made in advance, and
           logged as such, rather than a hole in the check.
        3. The working tree is clean. Merging into a dirty tree mixes MEDO's
           change with a person's unsaved work, and makes the rollback below
           unsafe — it could not tell them apart.
        4. The merged result still passes the suite. The gate proved the branch
           was good on its own; whether it is good *here* is a separate question
           whenever the branch was cut from an older HEAD. A failure rolls the
           merge back to the exact commit we started from.

        The change lands on disk; MEDO keeps running its old code until a
        manual restart.
        """
        if not proposal.ok:
            raise SelfDevError("refusing to apply a proposal that didn't pass its checks")
        approval = self.approvals.approval_for(proposal.branch, proposal.diff)
        if approval is None:
            if not self._config.auto_apply:
                raise SelfDevError(
                    "refusing to apply a change nobody approved — review the "
                    "diff and approve it first")
            approval = self.approve(proposal, approved_by="auto_apply")
        # TRACKED changes only. `reset --hard` — the rollback below — restores
        # tracked files and leaves untracked ones alone, so untracked files are
        # never at risk and demanding a pristine tree would just mean self-dev
        # stopped working for anyone with a scratch file in the repo.
        dirty = (await self._run_git(
            ["status", "--porcelain", "--untracked-files=no"])).strip()
        if dirty:
            raise SelfDevError(
                "the working tree has uncommitted changes; commit or stash them "
                "before applying a self-change (so a rollback can't lose them)")
        before = (await self._run_git(["rev-parse", "HEAD"])).strip()
        code, out, err = await self._run(
            [self._git, "merge", "--no-ff", proposal.branch,
             "-m", f"self-dev: apply {proposal.request[:60]}"], cwd=self._repo)
        if code != 0:
            with contextlib.suppress(Exception):
                await self._run_git(["merge", "--abort"])
            raise SelfDevError(f"could not merge cleanly: {_tail(err or out, 200)}")
        if self._config.verify_after_apply:
            ok, output = await self._verify_merge()
            if not ok:
                await self._rollback(before)
                self.approvals.revoke(proposal.branch)
                raise SelfDevError(
                    "the change broke the suite once merged, so I rolled it "
                    "back:\n" + _tail(output, 800))
        await self._cleanup(proposal.branch, proposal.worktree.parent)
        logger.info("self-dev: applied %s (approved by %s)",
                    proposal.branch, approval.approved_by)
        return True

    async def _verify_merge(self) -> tuple[bool, str]:
        """Run the test suite against the MERGED repo. (ok, output tail)."""
        env = self._gate_env(Path(tempfile.mkdtemp(prefix="medo-selfdev-verify-")))
        wrap = list(self._config.sandbox_cmd)
        code, out, err = await self._run(
            [*wrap, self._python, *self._config.test_cmd], cwd=self._repo,
            timeout=self._config.check_timeout_s, env=env)
        return code == 0, out + err

    async def _rollback(self, sha: str) -> None:
        """Return the repo to ``sha``, discarding the merge.

        ``reset --hard`` is safe here and nowhere else: :meth:`apply` refused to
        start unless the tree was clean, so the only thing being discarded is
        the merge this method is undoing.
        """
        try:
            await self._run_git(["reset", "--hard", sha])
            logger.warning("self-dev: rolled back to %s", sha[:8])
        except Exception:
            logger.exception(
                "self-dev: COULD NOT ROLL BACK to %s — the repository is left "
                "with a merge that failed its tests. Run: git reset --hard %s",
                sha, sha)
            raise

    async def discard(self, proposal: Proposal) -> None:
        """Roll a proposal back completely: drop its worktree, branch, approval."""
        await self._cleanup(proposal.branch, proposal.worktree.parent)
        # Revoke too, so a discarded-then-rebuilt branch of the same name can
        # never inherit the yes that was given to the old one.
        self.approvals.revoke(proposal.branch)
        logger.info("self-dev: discarded %s", proposal.branch)

    async def cleanup_stale(self) -> int:
        """Reap worktrees and branches left behind by a crash. Returns the count.

        A process killed mid-proposal leaves a registered worktree pointing at a
        temp directory that no longer exists, and every later ``git worktree``
        call trips over it. Git's own ``prune`` handles the registration; the
        branches need naming explicitly because they are not reachable from
        anything and ``prune`` leaves them alone.
        """
        reaped = 0
        with contextlib.suppress(Exception):
            await self._run_git(["worktree", "prune"])
        try:
            listing = await self._run_git(
                ["for-each-ref", "--format=%(refname:short)",
                 f"refs/heads/{self._config.branch_prefix}/**"])
        except SelfDevError:
            return reaped
        checked_out = ""
        with contextlib.suppress(Exception):
            checked_out = await self._run_git(["worktree", "list", "--porcelain"])
        for branch in (b.strip() for b in listing.splitlines() if b.strip()):
            if branch in checked_out:      # still in use by a live proposal
                continue
            with contextlib.suppress(Exception):
                await self._run_git(["branch", "-D", branch])
                self.approvals.revoke(branch)
                reaped += 1
        self.approvals.prune()
        if reaped:
            logger.info("self-dev: reaped %d orphaned proposal branch(es)", reaped)
        return reaped

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

    def _install_netblock(self, gate_dir: Path) -> str | None:
        """Copy the egress guard in as ``sitecustomize``; return its directory.

        CPython imports ``sitecustomize`` automatically at startup, before
        pytest and before any conftest, which is the only hook early enough to
        be in place when collection executes agent-authored module-level code.
        Returns None (and logs) if the copy fails — a gate that runs with the
        network up is worse than one that doesn't run at all only if you don't
        notice, so this is loud rather than fatal.
        """
        source = Path(__file__).with_name("gate_netblock.py")
        try:
            gate_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, gate_dir / "sitecustomize.py")
        except OSError as exc:
            logger.warning("self-dev: could not install the gate network guard "
                           "(%s); the gate will run WITH network access", exc)
            return None
        return str(gate_dir)

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
        if self._config.block_network_in_gate:
            pypath = self._install_netblock(sandbox_home / "gate-pypath")
            if pypath:
                existing = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = (f"{pypath}{os.pathsep}{existing}"
                                     if existing else pypath)
            # Point every proxy-honouring library at a dead port as well. Some
            # HTTP stacks resolve and connect through a proxy without ever
            # calling the functions the guard patches.
            env.update({"http_proxy": "http://127.0.0.1:9",
                        "https_proxy": "http://127.0.0.1:9",
                        "HTTP_PROXY": "http://127.0.0.1:9",
                        "HTTPS_PROXY": "http://127.0.0.1:9",
                        "no_proxy": "127.0.0.1,localhost",
                        "NO_PROXY": "127.0.0.1,localhost"})
        return env

    def _size_violation(self, files: list[str], diff: str) -> str:
        """Empty if the change is small enough to be reviewed, else why not."""
        max_files = int(self._config.max_files_changed or 0)
        if max_files > 0 and len(files) > max_files:
            return (f"the change touches {len(files)} files, over the "
                    f"{max_files}-file limit — ask for something narrower")
        max_bytes = int(self._config.max_diff_bytes or 0)
        size = len(diff.encode("utf-8", "replace"))
        if max_bytes > 0 and size > max_bytes:
            return (f"the diff is {size:,} bytes, over the {max_bytes:,}-byte "
                    f"limit — a change that big can't be meaningfully reviewed")
        return ""

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
