"""Human approval for a self-change, written down where a crash can't lose it.

The self-dev safety model is "propose & wait": MEDO builds a change in an
isolated worktree, gates it on the test suite, and then stops. What restarts it
has to be a person.

Holding that decision in memory was the weak part. A ``Proposal`` object in the
router's state says nothing about *who* let it through, survives nothing, and —
because ``apply()`` merely checked that the gate had passed — any code path
that reached the engine could merge a proposal no one had read. This module
makes approval a fact on disk with three properties:

* **Explicit.** It is written by one function, called from one place: the
  human-facing approve action. Passing the gate is not approval.
* **Bound to the diff.** The record stores a hash of the exact diff that was
  shown. Approve a change, let the branch move, and the approval no longer
  matches — you are asked again rather than merging something else.
* **Durable and readable.** A JSON file you can open, audit, and revoke, so
  "what has MEDO been allowed to do to itself" is answerable after a restart.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Approvals older than this are ignored. A change approved a fortnight ago and
#: never applied has been overtaken by the repository around it; re-reading the
#: diff is cheap and re-approving is one word.
APPROVAL_TTL_S = 7 * 24 * 3600.0


def diff_digest(diff: str) -> str:
    """The identity of a change: SHA-256 over the exact diff text."""
    return hashlib.sha256((diff or "").encode("utf-8", "replace")).hexdigest()


@dataclass(frozen=True)
class Approval:
    """One recorded human decision."""

    branch: str
    diff_sha256: str
    request: str
    approved_by: str
    approved_at: float

    def expired(self, now: float | None = None,
                ttl: float = APPROVAL_TTL_S) -> bool:
        return (time.time() if now is None else now) - self.approved_at > ttl

    def to_json(self) -> dict:
        return {"branch": self.branch, "diff_sha256": self.diff_sha256,
                "request": self.request, "approved_by": self.approved_by,
                "approved_at": self.approved_at}

    @classmethod
    def from_json(cls, raw: dict) -> "Approval | None":
        try:
            return cls(str(raw["branch"]), str(raw["diff_sha256"]),
                       str(raw.get("request", "")), str(raw.get("approved_by", "?")),
                       float(raw["approved_at"]))
        except (KeyError, TypeError, ValueError):
            return None


class ApprovalStore:
    """The on-disk record of which proposals a human has cleared."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> dict[str, Approval]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt file must fail CLOSED: an unreadable approvals file
            # means nothing is approved, never that everything is.
            logger.warning("self-dev approvals unreadable (%s); treating as empty", exc)
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, Approval] = {}
        for branch, entry in raw.items():
            approval = Approval.from_json(entry) if isinstance(entry, dict) else None
            if approval is not None:
                out[str(branch)] = approval
        return out

    def _save(self, approvals: dict[str, Approval]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {b: a.to_json() for b, a in approvals.items()}
        # Write-then-replace: a crash mid-write must not leave a truncated file
        # that reads as "nothing is approved" *and* loses the other records.
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def approve(self, branch: str, diff: str, *, request: str = "",
                approved_by: str = "human") -> Approval:
        """Record a decision. Returns the stored :class:`Approval`."""
        approval = Approval(branch=branch, diff_sha256=diff_digest(diff),
                            request=request, approved_by=approved_by,
                            approved_at=time.time())
        approvals = self._load()
        approvals[branch] = approval
        self._save(approvals)
        logger.info("self-dev: %s approved by %s", branch, approved_by)
        return approval

    def approval_for(self, branch: str, diff: str) -> Approval | None:
        """The live approval for this exact diff, or None.

        None covers all three ways an approval can fail to apply: there isn't
        one, it was for a different diff, or it has aged out.
        """
        approval = self._load().get(branch)
        if approval is None:
            return None
        if approval.diff_sha256 != diff_digest(diff):
            logger.warning("self-dev: %s changed since it was approved", branch)
            return None
        if approval.expired():
            logger.warning("self-dev: the approval for %s has expired", branch)
            return None
        return approval

    def revoke(self, branch: str) -> bool:
        """Withdraw an approval. False when there was nothing to withdraw."""
        approvals = self._load()
        if approvals.pop(branch, None) is None:
            return False
        self._save(approvals)
        logger.info("self-dev: approval for %s revoked", branch)
        return True

    def pending(self) -> list[Approval]:
        """Every stored approval that hasn't expired, newest first."""
        live = [a for a in self._load().values() if not a.expired()]
        return sorted(live, key=lambda a: a.approved_at, reverse=True)

    def prune(self) -> int:
        """Drop expired records. Returns how many were removed."""
        approvals = self._load()
        stale = [b for b, a in approvals.items() if a.expired()]
        for branch in stale:
            del approvals[branch]
        if stale:
            self._save(approvals)
        return len(stale)
