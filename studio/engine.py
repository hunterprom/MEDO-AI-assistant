"""The shared Maker Studio engine (S1): the pipeline all three domains sit on.

    intent → BRIEF  (ask a clarifying question if a critical spec is missing)
           → CODE   (the brain writes real library code — the source of truth)
           → EXECUTE in the sandbox → artifact
                └─ error? feed the traceback back and retry (bounded)
           → CHECK  (sanity/printability/tests → advisories; a 'fail' means NOT
                     shipped as "ready")
           → VERSION + present  (save code + artifact as vN; iterate = vN+1)

Domain-agnostic: a :class:`studio.domain.Domain` supplies the library-specific
prompt / execute / check; everything here is shared and unit-tested with fakes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from studio.brain import Brain
from studio.domain import Advisory, Brief, Domain, ExecOutcome
from studio.projects import StudioProject
from studio.sandbox import Sandbox

logger = logging.getLogger(__name__)


def _tail(text: str, n: int = 1600) -> str:
    return (text or "").strip()[-n:]


@dataclass
class StudioResult:
    ok: bool
    project: Optional[StudioProject] = None
    version: int = 0
    artifacts: List[str] = field(default_factory=list)
    advisories: List[Advisory] = field(default_factory=list)
    preview: Optional[str] = None
    error: str = ""
    attempts: int = 0
    #: Non-empty => a critical spec is missing; ask this before generating.
    question: str = ""


class StudioEngine:
    """Runs the pipeline. ``brain`` / ``sandbox`` are injectable (tests pass
    fakes); by default the sandbox is policy-gated from settings and the brain is
    wired by the skill layer (S2–S4) to the active LLM."""

    def __init__(self, settings, *, brain: Optional[Brain] = None,
                 sandbox: Optional[Sandbox] = None, base_dir=None) -> None:
        self._settings = settings
        self._cfg = settings.studio
        self._base_dir = (Path(base_dir).expanduser() if base_dir
                          else Path(self._cfg.projects_dir).expanduser())
        self._brain = brain or Brain(model=self._cfg.model, settings=settings)
        self._sandbox = sandbox or self._default_sandbox()

    def _default_sandbox(self) -> Sandbox:
        policy = actor = None
        try:
            from core.safety import PathWhitelist
            from security.policy import Actor, PolicyEngine

            policy = PolicyEngine(self._settings.security, self._settings.safety,
                                  PathWhitelist(self._settings.safety.whitelist_dirs))
            actor = Actor.MODEL
        except Exception:
            logger.warning("studio: policy engine unavailable; the sandbox still "
                           "env/cwd/timeout-isolates but runs without the policy "
                           "authorisation gate", exc_info=True)
        return Sandbox(policy=policy, actor=actor,
                       sandbox_cmd=self._cfg.sandbox_cmd,
                       timeout_s=self._cfg.exec_timeout_s)

    def _disabled(self) -> Optional[StudioResult]:
        if not self._cfg.enabled:
            return StudioResult(
                False, error="Maker Studio is off — set studio.enabled to allow it.")
        return None

    async def create(self, domain: Domain, description: str,
                     answers: Optional[dict] = None) -> StudioResult:
        gate = self._disabled()
        if gate is not None:
            return gate
        brief = domain.build_brief(description, answers or {})
        if brief.question:                     # don't guess a critical spec
            return StudioResult(False, question=brief.question)
        project = StudioProject.create(self._base_dir, domain.name, description)
        return await self._run(domain, project, brief, prior_code="")

    async def iterate(self, domain: Domain, project: StudioProject,
                      edit: str) -> StudioResult:
        """Apply an edit ("make the wall 3mm", "add a pull-up on GPIO4") to the
        project's latest code and re-run — a new version, history kept."""
        gate = self._disabled()
        if gate is not None:
            return gate
        return await self._run(domain, project, Brief(description=edit),
                               prior_code=project.latest_code())

    # -- the generate → execute → self-correct → check → version loop ---------

    async def _run(self, domain: Domain, project: StudioProject, brief: Brief,
                   prior_code: str) -> StudioResult:
        n, vdir = project.begin_version()
        code, error, outcome = prior_code, "", None
        attempts = 0
        for attempt in range(self._cfg.max_retries + 1):
            attempts = attempt + 1
            try:
                code = await self._brain.write_code(
                    domain.system_prompt(),
                    domain.user_prompt(brief, code, error))
            except Exception as exc:
                error = f"code generation failed: {exc}"
                break
            if not code.strip():
                if not error:            # keep a real execution error if we have one
                    error = "the brain returned no code"
                continue
            (vdir / domain.filename).write_text(code, encoding="utf-8")
            try:
                outcome = await domain.execute(code, vdir, self._sandbox)
            except Exception as exc:            # a domain bug must not crash the turn
                outcome = ExecOutcome(ok=False, stderr=f"execution crashed: {exc}")
            if outcome.ok:
                break
            error = _tail(outcome.stderr) or "execution produced no artifact"

        # Checks only run on a real artifact; a 'fail' advisory blocks "ready".
        advisories = domain.check(outcome, vdir) if (outcome and outcome.ok) else []
        hard_fail = any(a.level == "fail" for a in advisories)
        ok = bool(outcome and outcome.ok) and not hard_fail
        version = project.finalize_version(
            n, vdir, filename=domain.filename,
            artifacts=list(outcome.artifacts) if outcome else [],
            advisories=[{"level": a.level, "message": a.message} for a in advisories],
            ok=ok, error=("" if ok else error))
        project.prune(self._cfg.keep_versions)
        return StudioResult(
            ok=ok, project=project, version=version.n,
            artifacts=list(outcome.artifacts) if outcome else [],
            advisories=advisories,
            preview=(outcome.preview if outcome else None),
            error=("" if ok else error), attempts=attempts)
