"""A Studio project on disk: one folder per creation, with the code, the
artifacts, NUMBERED versions (so you can go back), and metadata.

Layout::

    <projects_dir>/<slug>-<id>/
        project.json          # domain, description, versions[], next_n
        v1/ design.py + artifacts…
        v2/ design.py + artifacts…   (an iteration = a new version)

Pure filesystem + JSON — no model, no execution — so it's unit-tested directly.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _slug(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:limit].rstrip("-") or "project"


@dataclass
class Version:
    n: int
    filename: str
    artifacts: List[str] = field(default_factory=list)
    advisories: List[dict] = field(default_factory=list)
    ok: bool = False
    error: str = ""
    created_at: float = 0.0


class StudioProject:
    """One project folder + its version history."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._recovered = False        # set by _load_meta when metadata was lost
        self._meta = self._load_meta()
        if not self._recovered:
            # NEVER sweep against recovered metadata: it lists no versions, so
            # every real vN would look like an orphan and the whole history
            # would be deleted — the exact clobber _load_meta exists to prevent.
            self._sweep_orphans()

    #: Don't sweep a reserved-but-unfinalized dir younger than this — it may be a
    #: build running RIGHT NOW in another StudioProject handle on the same root.
    ORPHAN_MIN_AGE_S = 3600.0

    def _sweep_orphans(self, min_age_s: float | None = None) -> int:
        """Remove ``vN`` dirs that were reserved but never finalized — a run that
        crashed mid-build leaves one behind, invisible to ``versions()`` and
        never pruned. Only touches dirs older than ``ORPHAN_MIN_AGE_S`` so a
        live build is never deleted. Returns how many were removed."""
        import shutil

        age = self.ORPHAN_MIN_AGE_S if min_age_s is None else min_age_s
        known = {int(v.get("n", 0)) for v in self._meta.get("versions", [])}
        next_n = int(self._meta.get("next_n", 1))
        cutoff = time.time() - age
        removed = 0
        try:
            for d in self.root.glob("v*"):
                if not (d.is_dir() and d.name[1:].isdigit()):
                    continue
                if int(d.name[1:]) >= next_n or int(d.name[1:]) in known:
                    continue
                if d.stat().st_mtime > cutoff:      # too fresh — may be in flight
                    continue
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except Exception:
            logger.debug("orphan sweep failed in %s", self.root, exc_info=True)
        return removed

    # -- lifecycle ------------------------------------------------------------

    @classmethod
    def create(cls, base_dir, domain: str, description: str) -> "StudioProject":
        base = Path(base_dir).expanduser()
        folder = _slug(description)
        root = base / f"{folder}-{uuid.uuid4().hex[:6]}"
        root.mkdir(parents=True, exist_ok=True)
        meta = {"domain": domain, "description": description,
                "created_at": time.time(), "next_n": 1, "versions": []}
        (root / "project.json").write_text(json.dumps(meta, indent=2),
                                           encoding="utf-8")
        return cls(root)

    @classmethod
    def open(cls, path) -> "StudioProject":
        return cls(Path(path))

    def _load_meta(self) -> dict:
        p = self.root / "project.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                # Corrupt metadata: don't silently reset to v1 and clobber the
                # existing version dirs — recover next_n from what's on disk so
                # history isn't overwritten.
                logger.warning("studio: project.json unreadable at %s — "
                               "recovering version number from disk", self.root)
        self._recovered = True
        existing = [int(d.name[1:]) for d in self.root.glob("v*")
                    if d.name[1:].isdigit()]
        return {"domain": "", "description": "", "created_at": time.time(),
                "next_n": (max(existing) + 1 if existing else 1), "versions": []}

    def _save_meta(self) -> None:
        # Atomic: a crash mid-write used to truncate project.json, losing the
        # whole version list. Write beside it, then replace in one step.
        target = self.root / "project.json"
        tmp = target.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(self._meta, indent=2), encoding="utf-8")
            tmp.replace(target)
        except Exception:
            logger.warning("studio: could not save project.json in %s", self.root,
                           exc_info=True)
            with contextlib.suppress(Exception):
                tmp.unlink()

    # -- versions -------------------------------------------------------------

    def begin_version(self) -> Tuple[int, Path]:
        """RESERVE the next version number + its dir (returns ``(n, dir)``). The
        number is committed immediately so two concurrent runs can't collide on
        the same ``vN``; metadata for the entry is written by
        ``finalize_version``. Retries write + re-run inside this same dir."""
        n = self._meta["next_n"]
        vdir = self.root / f"v{n}"
        vdir.mkdir(parents=True, exist_ok=True)
        self._meta["next_n"] = n + 1        # reserve now, not at finalize
        self._save_meta()
        return n, vdir

    def finalize_version(self, n: int, vdir: Path, *, filename: str,
                         artifacts: List[str], advisories: List[dict],
                         ok: bool, error: str = "") -> Version:
        version = Version(n=n, filename=filename, artifacts=list(artifacts),
                          advisories=list(advisories), ok=bool(ok), error=error,
                          created_at=time.time())
        self._meta["versions"].append(asdict(version))
        self._save_meta()
        return version

    def prune(self, keep: int) -> int:
        """Keep only the newest ``keep`` versions (0 = unlimited). Returns the
        number removed."""
        import shutil

        versions = self._meta.get("versions", [])
        if keep <= 0 or len(versions) <= keep:
            return 0
        drop = versions[:-keep]
        for v in drop:
            with_dir = self.root / f"v{v['n']}"
            if with_dir.exists():
                shutil.rmtree(with_dir, ignore_errors=True)
        self._meta["versions"] = versions[-keep:]
        self._save_meta()
        return len(drop)

    # -- reads ----------------------------------------------------------------

    def versions(self) -> List[Version]:
        return [Version(**v) for v in self._meta.get("versions", [])]

    def latest(self) -> Optional[Version]:
        vs = self._meta.get("versions", [])
        return Version(**vs[-1]) if vs else None

    def latest_code(self) -> str:
        v = self.latest()
        if v is None:
            return ""
        p = self.root / f"v{v.n}" / v.filename
        return p.read_text(encoding="utf-8") if p.exists() else ""

    @property
    def domain(self) -> str:
        return self._meta.get("domain", "")

    @property
    def description(self) -> str:
        return self._meta.get("description", "")


def list_projects(base_dir) -> List[dict]:
    """Light summary of every project (for the HUD), newest first. Never raises."""
    base = Path(base_dir).expanduser()
    out: List[dict] = []
    try:
        for p in base.glob("*/project.json"):
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
                vs = m.get("versions", [])
                out.append({
                    "path": str(p.parent), "domain": m.get("domain", ""),
                    "description": m.get("description", ""),
                    "versions": len(vs),
                    "ok": bool(vs[-1].get("ok")) if vs else False,
                    "created_at": float(m.get("created_at", 0.0)),
                })
            except Exception:
                continue
    except Exception:
        return []
    out.sort(key=lambda m: m["created_at"], reverse=True)
    return out
