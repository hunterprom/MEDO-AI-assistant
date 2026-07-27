"""Talk to MEDO's self-programming engine (core/self_dev.py) by voice or text.

"Fix your code so X" / "add a feature to yourself: Y" → MEDO builds the change in
an isolated worktree IN THE BACKGROUND (a coding agent + the full test suite takes
minutes, far longer than a turn), then ANNOUNCES the verdict when it's ready. The
change is applied only on an explicit "apply your change", which — because it
edits the live code — goes through the normal yes/no confirmation gate.

Only registered when ``self_dev.enabled`` is set, so its very presence is the
opt-in; the engine's safelist/denylist + the isolated worktree bound what a
proposal can touch.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re

from core.self_dev import Proposal, SelfDevEngine, SelfDevError
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

#: Start a change. Deliberately demands a "your (own) code/self" marker so a
#: normal "fix the weather skill" request doesn't get hijacked into self-editing.
_START = re.compile(
    r"\b(?:fix|change|improve|update|refactor|rewrite|patch)\s+your\s+(?:own\s+)?"
    r"(?:code|self|bug|bugs|source)\b"
    r"|\bprogram\s+yourself\b"
    r"|\badd\s+(?:a\s+)?feature\s+to\s+yourself\b"
    r"|\bwork\s+on\s+your(?:\s+own)?\s+(?:code|self)\b"
    r"|\bself[-\s]?(?:dev|develop|program|improve)\b",
    re.IGNORECASE)
_APPLY = re.compile(
    r"\bapply\s+(?:your\s+|the\s+|that\s+|this\s+)?(?:change|proposal|fix|patch|code)\b",
    re.IGNORECASE)
_DISCARD = re.compile(
    r"\b(?:discard|throw\s+away|drop|cancel|reject)\s+(?:your\s+|the\s+|that\s+|this\s+)?"
    r"(?:change|proposal|fix|patch)\b",
    re.IGNORECASE)


def _strip_trigger(text: str) -> str:
    """Reduce "fix your code so <X>" to the underlying ask "<X>"."""
    t = _START.sub("", text, count=1).strip(" ,:;.-—")
    t = re.sub(r"^(?:so\s+that|so|to|that|and|:)\s+", "", t, flags=re.IGNORECASE)
    return t.strip()


class SelfDevSkill(Skill):
    name = "self_dev"
    description = (
        "Propose a change to MEDO's own code — fix a bug or add a feature — built "
        "in isolation, gated by the tests, and applied only on your approval.")
    patterns = [_START, _APPLY, _DISCARD]

    def __init__(self, engine: SelfDevEngine, announcer=None) -> None:
        self._engine = engine
        self._announce = announcer          # async callable(text) or None
        self._pending: Proposal | None = None
        self._task: asyncio.Task | None = None
        self._busy = False

    async def execute(self, request: SkillRequest) -> SkillResult:
        # The "yes" confirming an apply comes back through here with confirmed set.
        if request.context.get("confirmed"):
            return await self._apply_now()
        text = request.text
        if _APPLY.search(text):
            return self._apply_prompt()
        if _DISCARD.search(text):
            return await self._discard()
        return self._start(request)

    # -- start a proposal (background) --------------------------------------

    def _start(self, request: SkillRequest) -> SkillResult:
        if self._busy:
            return SkillResult(
                "I'm already working on a change — let me finish that one first.",
                success=False)
        dev_request = _strip_trigger(request.text)
        if len(dev_request) < 4:
            return SkillResult(
                "Tell me what to fix or add — for example, 'fix your code so the "
                "weather skill handles an empty city'.", success=False)
        self._busy = True
        self._task = asyncio.ensure_future(self._work(dev_request))
        return SkillResult(
            "Okay — I'll work on that in the background and let you know when "
            "there's a proposal to review.",
            data={"self_dev": "started", "request": dev_request})

    async def _work(self, dev_request: str) -> None:
        try:
            proposal = await self._engine.propose(dev_request)
            if proposal.ok:
                self._pending = proposal
                msg = (f'I have a proposal for "{dev_request}": it changes '
                       f"{len(proposal.files_changed)} file(s) and passed all my "
                       "tests. Say 'apply your change' to merge it, or 'discard it'.")
            else:
                # Not mergeable (failed the gate, empty, or out of scope) — set it
                # aside so nothing dangling is left, and say why.
                with contextlib.suppress(Exception):
                    await self._engine.discard(proposal)
                self._pending = None
                msg = (f'I tried "{dev_request}" but it didn\'t pass, so I set it '
                       f"aside — {proposal.summary()}.")
        except SelfDevError as exc:
            self._pending = None
            msg = f"I couldn't build that change: {exc}"
        finally:
            self._busy = False
        logger.info("self-dev: %s", msg)
        if self._announce is not None:
            with contextlib.suppress(Exception):
                await self._announce(msg)

    # -- apply / discard ----------------------------------------------------

    def _apply_prompt(self) -> SkillResult:
        if self._pending is None:
            return SkillResult("I don't have a proposal ready to apply.", success=False)
        p = self._pending
        return SkillResult(
            f"Apply my proposed change to {len(p.files_changed)} file(s)? It passed "
            "all the tests. Say yes to merge it.",
            needs_confirmation=True)

    async def _apply_now(self) -> SkillResult:
        if self._pending is None:
            return SkillResult("There's nothing to apply.", success=False)
        p = self._pending
        try:
            await self._engine.apply(p)
        except SelfDevError as exc:
            return SkillResult(f"I couldn't merge it: {exc}", success=False)
        self._pending = None
        return SkillResult(
            f"Done — I merged the change to {len(p.files_changed)} file(s). "
            "Restart me to run the new code.")

    async def _discard(self) -> SkillResult:
        if self._pending is None:
            return SkillResult("There's no pending proposal to discard.", success=False)
        p, self._pending = self._pending, None
        with contextlib.suppress(Exception):
            await self._engine.discard(p)
        return SkillResult("Discarded the proposal.")
