"""The policy engine is the crown jewel — heavily unit-tested, in isolation.

These use duck-typed fakes for the config + whitelist (the engine reads via
getattr), so they pin the DECISION CONTRACT without any Settings/router wiring:
deny-by-default, the actuation master-switch baseline, the whitelist fold-in,
the trust boundary, and fail-closed-on-error.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from security.capabilities import Capability, effective_capabilities, actuates
from security.policy import (
    ActionRequest,
    Actor,
    Decision,
    Effect,
    PolicyEngine,
    Provenance,
)


# -- fakes --------------------------------------------------------------------

def _sec(enabled=True, confirm=None, untrusted="confirm"):
    return SimpleNamespace(enabled=enabled,
                           requires_confirmation=dict(confirm or {}),
                           untrusted_action_policy=untrusted)


def _safety(pc=True):
    return SimpleNamespace(pc_control_enabled=pc)


class _WL:
    def __init__(self, allow=True, boom=False):
        self._allow, self._boom = allow, boom

    def is_allowed(self, path):
        if self._boom:
            raise OSError("whitelist exploded")
        return self._allow


class _Skill:
    def __init__(self, name="s", caps=frozenset(), controls_pc=False):
        self.name = name
        self.capabilities = caps
        self.controls_pc = controls_pc


def _engine(sec=None, safety=None, wl=None):
    return PolicyEngine(sec or _sec(), safety or _safety(), wl)


def _req(cap, **kw):
    kw.setdefault("actor", Actor.LOCAL_USER)
    return ActionRequest(capability=cap, **kw)


# -- the four S1 contract tests ----------------------------------------------

def test_allowed_action_passes():
    e = _engine()
    d = e.check(_req(Capability.NETWORK))
    assert d.effect is Effect.ALLOW and d.allowed()


def test_undeclared_capability_is_denied():
    e = _engine()
    # subject declared only read_files, but asks to write.
    d = e.check(_req(Capability.WRITE_FILES,
                     declared=frozenset({Capability.READ_FILES}),
                     subject="notes"))
    assert d.effect is Effect.DENY and d.code == "undeclared"
    assert "declare" in d.reason


def test_high_impact_capability_returns_confirm():
    e = _engine(sec=_sec(confirm={"power_control": True}))
    d = e.check(_req(Capability.POWER_CONTROL))
    assert d.effect is Effect.CONFIRM and d.needs_confirmation()


def test_policy_errors_fail_closed_to_deny():
    # A whitelist that raises must NOT fail open — the check degrades to DENY.
    e = _engine(wl=_WL(boom=True))
    d = e.check(_req(Capability.WRITE_FILES, target="C:/whatever.txt"))
    assert d.effect is Effect.DENY and d.code == "check_error"


# -- baseline actuation gate (mirrors today's controls_pc) --------------------

def test_actuation_denied_when_pc_control_off():
    e = _engine(safety=_safety(pc=False))
    d = e.check(_req(Capability.CONTROL_INPUT))
    assert d.effect is Effect.DENY and d.code == "pc_control_off"


def test_non_actuation_is_not_gated_by_pc_control():
    # Closing MEDO, networking, and cloud-brain use are NOT machine actuation, so
    # the PC-control switch must never hide them (matches controls_pc=False today).
    e = _engine(safety=_safety(pc=False))
    for cap in (Capability.SESSION_CONTROL, Capability.NETWORK,
                Capability.USE_CLOUD_BRAIN):
        assert e.check(_req(cap)).effect is not Effect.DENY, cap


def test_baseline_gate_holds_even_when_security_disabled():
    # security.enabled=False turns off the NEW layer but keeps the safety floor.
    e = _engine(sec=_sec(enabled=False), safety=_safety(pc=False))
    assert e.check(_req(Capability.POWER_CONTROL)).code == "pc_control_off"


def test_escape_hatch_skips_the_new_layer_but_still_allows_reads():
    e = _engine(sec=_sec(enabled=False, confirm={"power_control": True}))
    # confirm-map is part of the NEW layer -> skipped -> allowed with pc on
    d = e.check(_req(Capability.POWER_CONTROL))
    assert d.effect is Effect.ALLOW and d.code == "legacy"


# -- whitelist fold-in --------------------------------------------------------

def test_file_write_outside_whitelist_is_denied():
    e = _engine(wl=_WL(allow=False))
    d = e.check(_req(Capability.WRITE_FILES, target="C:/etc/passwd"))
    assert d.effect is Effect.DENY and d.code == "outside_whitelist"


def test_file_read_inside_whitelist_is_allowed():
    e = _engine(wl=_WL(allow=True))
    assert e.check(_req(Capability.READ_FILES, target="C:/ok/f.txt")).allowed()


def test_network_rejects_non_http_scheme():
    e = _engine()
    assert e.check(_req(Capability.NETWORK, target="file:///etc/passwd")).code \
        == "bad_scheme"
    assert e.check(_req(Capability.NETWORK, target="https://ok.test")).allowed()


# -- trust boundary (S3 hook) -------------------------------------------------

def test_untrusted_proposed_side_effect_confirms_by_default():
    e = _engine(sec=_sec(untrusted="confirm"))
    d = e.check(_req(Capability.WRITE_FILES, provenance=Provenance.UNTRUSTED))
    assert d.effect is Effect.CONFIRM and d.code == "untrusted_confirm"


def test_untrusted_proposed_side_effect_can_be_denied():
    e = _engine(sec=_sec(untrusted="deny"))
    d = e.check(_req(Capability.RUN_COMMAND, provenance=Provenance.UNTRUSTED))
    assert d.effect is Effect.DENY and d.code == "untrusted_deny"


def test_untrusted_pure_read_is_not_side_effect_gated():
    # Reading a local file is not in SIDE_EFFECT_CAPS, so provenance alone
    # doesn't gate it (S3 may refine); it still passes the whitelist check.
    e = _engine(wl=_WL(allow=True))
    d = e.check(_req(Capability.READ_FILES, target="C:/ok/f.txt",
                     provenance=Provenance.UNTRUSTED))
    assert d.effect is Effect.ALLOW


# -- gate_skill folds to the strongest decision -------------------------------

def test_gate_skill_allows_sensing():
    e = _engine()
    assert e.gate_skill(_Skill("weather")).allowed()


def test_gate_skill_bridges_legacy_controls_pc():
    on = _engine(safety=_safety(pc=True))
    off = _engine(safety=_safety(pc=False))
    legacy = _Skill("files", controls_pc=True)          # no declared caps
    assert on.gate_skill(legacy).allowed()
    d = off.gate_skill(legacy)
    assert d.denied() and d.code == "pc_control_off"


def test_gate_skill_returns_confirm_for_a_high_impact_skill():
    e = _engine(sec=_sec(confirm={"power_control": True}))
    power = _Skill("power", caps=frozenset({Capability.POWER_CONTROL}),
                   controls_pc=True)
    assert e.gate_skill(power).needs_confirmation()


# -- capability helpers -------------------------------------------------------

def test_effective_capabilities_bridge():
    assert effective_capabilities(_Skill("w")) == frozenset()           # sensing
    assert effective_capabilities(_Skill("f", controls_pc=True)) == \
        frozenset({Capability.LEGACY_ACTUATION})
    caps = frozenset({Capability.POWER_CONTROL})
    assert effective_capabilities(_Skill("p", caps=caps)) == caps       # explicit wins


def test_actuates_matches_controls_pc_intent():
    assert actuates(_Skill("power", caps=frozenset({Capability.POWER_CONTROL})))
    assert not actuates(_Skill("quit", caps=frozenset({Capability.SESSION_CONTROL})))
    assert not actuates(_Skill("weather"))


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
