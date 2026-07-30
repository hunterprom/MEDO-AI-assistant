"""Tamper-evident security audit log + the HUD security-status provider (S6).

Append-only, HASH-CHAINED JSONL: each entry carries the hash of the previous one,
so editing, deleting, or reordering any entry is detectable (``verify``). It
records security-relevant EVENTS as metadata only — event type, capability, skill,
actor, decision, reason — never secret values (string fields are run through the
redactor) and never message/document PAYLOADS.

``python -m security.audit --report`` summarizes recent events + verifies the
chain. ``security_status`` feeds the HUD's security panel.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Event types (metadata only — never a payload).
AUTH_OK = "auth_ok"
AUTH_FAIL = "auth_fail"
CONFIRM_GRANTED = "confirm_granted"
CONFIRM_DENIED = "confirm_denied"
ACTION = "action"
CLOUD_CALL = "cloud_call"
DEVICE_CMD = "device_cmd"
PLUGIN_INSTALL = "plugin_install"
POLICY_DENY = "policy_deny"

_GENESIS = "0" * 64


def _default_path() -> Path:
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"),
                                                     ".config")
    return Path(base) / "MEDO" / "audit.log"


def _entry_hash(prev: str, entry: dict) -> str:
    body = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256((prev + body).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: Optional[Path] = None, *,
                 now: Callable[[], float] = time.time,
                 redactor: Optional[Callable[[str], str]] = None) -> None:
        self._path = Path(path) if path is not None else _default_path()
        self._now = now
        self._redact = redactor or (lambda t: t)

    # -- write ----------------------------------------------------------------

    def record(self, event: str, **fields: Any) -> None:
        """Append one event. Never raises into the caller — a broken audit log
        must not break the action it was recording."""
        try:
            entry = {"ts": round(self._now(), 3), "event": event}
            for k, v in fields.items():
                entry[k] = self._redact(v) if isinstance(v, str) else v
            prev = self._last_hash()
            entry["prev"] = prev
            entry["hash"] = _entry_hash(prev, entry)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("audit record failed for %s", event, exc_info=True)

    def _last_hash(self) -> str:
        entries = self.entries()
        return entries[-1]["hash"] if entries else _GENESIS

    # -- read + verify --------------------------------------------------------

    def entries(self, limit: Optional[int] = None) -> List[dict]:
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out[-limit:] if limit else out

    def verify(self) -> Tuple[bool, int]:
        """Re-walk the chain. Returns (ok, first_bad_index); (True, -1) when the
        log is intact. Any edited field (hash mismatch) or broken link (prev
        mismatch) is caught."""
        prev = _GENESIS
        for i, entry in enumerate(self.entries()):
            stored = entry.get("hash")
            if entry.get("prev") != prev:
                return False, i
            body = {k: v for k, v in entry.items() if k != "hash"}
            if _entry_hash(entry.get("prev", ""), body) != stored:
                return False, i
            prev = stored
        return True, -1

    def report(self) -> dict:
        entries = self.entries()
        counts: dict[str, int] = {}
        for e in entries:
            counts[e.get("event", "?")] = counts.get(e.get("event", "?"), 0) + 1
        ok, bad = self.verify()
        return {"total": len(entries), "counts": counts,
                "intact": ok, "first_bad_index": bad,
                "recent": entries[-10:]}


def security_status(settings: Any, audit: Optional[AuditLog] = None) -> dict:
    """Plain-language security state for the HUD panel: which brain (with a big
    LOCAL/CLOUD badge), the guards' on/off, and recent security events."""
    from security.egress import is_cloud

    provider = getattr(settings.llm, "provider", "ollama")
    sec = settings.security
    return {
        "brain": provider,
        "cloud_active": is_cloud(provider),           # the obvious cloud indicator
        "owner_voice": bool(getattr(sec, "owner_voice", False)),
        "plugin_approval": bool(getattr(sec, "plugin_approval", False)),
        "untrusted_action_policy": getattr(sec, "untrusted_action_policy", "confirm"),
        "cloud_egress_optin": dict(getattr(sec, "cloud_egress_optin", {}) or {}),
        "recent_events": audit.entries(limit=20) if audit else [],
        "audit_intact": audit.verify()[0] if audit else None,
    }


def _main(argv: List[str]) -> int:  # pragma: no cover - thin CLI
    log = AuditLog()
    rep = log.report()
    print(f"MEDO security audit — {rep['total']} events, "
          f"chain {'INTACT' if rep['intact'] else 'TAMPERED @%d' % rep['first_bad_index']}")
    for ev, n in sorted(rep["counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:>5}  {ev}")
    if rep["recent"]:
        print("recent:")
        for e in rep["recent"]:
            print(f"  {e.get('ts')}  {e.get('event')}  {e.get('detail', '')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(_main(sys.argv[1:]))
