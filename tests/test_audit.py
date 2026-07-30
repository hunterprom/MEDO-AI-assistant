"""Tamper-evident audit log (S6): the hash chain catches any edit.

Also: entries carry metadata only (secrets redacted), the report aggregates, the
HUD status reflects the active brain, and the router writes an entry per event
class when auditing is on.
"""

from __future__ import annotations

import json

import pytest

from core.config import load_settings
from core.events import EventBus
from core.router import Router
from llm.client import OllamaClient
from security.audit import (
    ACTION,
    CLOUD_CALL,
    CONFIRM_GRANTED,
    POLICY_DENY,
    AuditLog,
    security_status,
)
from skills.base import SkillRequest
from skills.system import PowerSkill


def _clock():
    t = [1000.0]

    def now():
        t[0] += 1.0
        return t[0]
    return now


def _log(tmp_path, redactor=None):
    return AuditLog(tmp_path / "audit.log", now=_clock(), redactor=redactor)


# -- write + read -------------------------------------------------------------

def test_records_and_reads_metadata(tmp_path):
    log = _log(tmp_path)
    log.record(POLICY_DENY, skill="power", code="pc_control_off")
    log.record(CONFIRM_GRANTED, skill="quit")
    entries = log.entries()
    assert [e["event"] for e in entries] == [POLICY_DENY, CONFIRM_GRANTED]
    assert entries[0]["skill"] == "power" and entries[0]["code"] == "pc_control_off"


# -- the hash chain -----------------------------------------------------------

def test_intact_chain_verifies():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        log = AuditLog(Path(d) / "a.log", now=_clock())
        for i in range(5):
            log.record(ACTION, skill=f"s{i}")
        assert log.verify() == (True, -1)


def test_editing_a_field_is_detected(tmp_path):
    p = tmp_path / "audit.log"
    log = AuditLog(p, now=_clock())
    log.record(ACTION, skill="a")
    log.record(ACTION, skill="b")
    log.record(ACTION, skill="c")
    lines = p.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1])
    tampered["skill"] = "HACKED"            # edit a field, keep the old hash
    lines[1] = json.dumps(tampered)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, bad = AuditLog(p).verify()
    assert ok is False and bad == 1


def test_deleting_an_entry_breaks_the_chain(tmp_path):
    p = tmp_path / "audit.log"
    log = AuditLog(p, now=_clock())
    for s in ("a", "b", "c"):
        log.record(ACTION, skill=s)
    lines = p.read_text(encoding="utf-8").splitlines()
    del lines[1]                            # remove the middle entry
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, bad = AuditLog(p).verify()
    assert ok is False and bad == 1


# -- no secrets ---------------------------------------------------------------

def test_secret_values_are_redacted_out(tmp_path):
    log = _log(tmp_path, redactor=lambda t: t.replace("supersecret", "***"))
    log.record("auth_ok", detail="authorized with token supersecret")
    dumped = json.dumps(log.entries())
    assert "supersecret" not in dumped and "***" in dumped


# -- report + status ----------------------------------------------------------

def test_report_aggregates_and_verifies(tmp_path):
    log = _log(tmp_path)
    log.record(POLICY_DENY, skill="a")
    log.record(POLICY_DENY, skill="b")
    log.record(CLOUD_CALL, brain="anthropic")
    rep = log.report()
    assert rep["total"] == 3 and rep["counts"][POLICY_DENY] == 2
    assert rep["intact"] is True


def test_security_status_shows_the_brain_badge():
    s = load_settings()
    s.llm.provider = "anthropic"
    st = security_status(s)
    assert st["brain"] == "anthropic" and st["cloud_active"] is True
    s.llm.provider = "ollama"
    assert security_status(s)["cloud_active"] is False


# -- router integration -------------------------------------------------------

@pytest.mark.asyncio
async def test_router_audits_a_policy_denial(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))     # keep audit under tmp
    from skills.base import SkillRegistry
    s = load_settings()
    s.security.audit_enabled = True
    s.safety.pc_control_enabled = False              # so power is denied
    reg = SkillRegistry()
    power = PowerSkill()
    reg.register(power)
    r = Router(s, reg, OllamaClient(s.llm), EventBus())
    assert r._audit_log is not None
    await r._run_skill(power, SkillRequest(text="shut down", context={}))
    events = [e["event"] for e in r._audit_log.entries()]
    assert POLICY_DENY in events


def test_router_without_audit_flag_has_no_log():
    s = load_settings()                              # audit_enabled defaults False
    from skills.base import SkillRegistry
    r = Router(s, SkillRegistry(), OllamaClient(s.llm), EventBus())
    assert r._audit_log is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
