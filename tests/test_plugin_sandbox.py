"""Plugin install-review + approval gate (S4).

The load-bearing test: a plugin's capabilities are read WITHOUT executing it, and
an un-approved plugin's code is never imported. Plus runtime confinement (a
plugin skill can't use an undeclared capability) and the generated-code safety
defaults.
"""

from __future__ import annotations

import pytest

from core.plugins import load_plugins
from security.capabilities import Capability
from security.plugins import (
    PluginApprovalStore,
    file_digest,
    static_review,
)
from security.policy import ActionRequest, Actor, PolicyEngine
from skills.base import SkillRegistry


def _write(dirpath, name, body):
    p = dirpath / name
    p.write_text(body, encoding="utf-8")
    return p


# -- static review does NOT execute the plugin --------------------------------

def test_review_reads_capabilities_without_executing(tmp_path):
    p = _write(tmp_path, "danger.py",
               'CAPABILITIES = ["write_files", "network"]\n'
               'PLUGIN_NAME = "Danger Plugin"\n'
               'raise RuntimeError("this must never run during review")\n')
    review = static_review(p)                     # would blow up if it executed
    assert review.declared is True
    assert review.capabilities == frozenset({Capability.WRITE_FILES,
                                             Capability.NETWORK})
    assert review.display_name == "Danger Plugin"
    assert "write_files" in review.summary() and "network" in review.summary()


def test_review_flags_unknown_and_undeclared(tmp_path):
    p = _write(tmp_path, "weird.py", 'CAPABILITIES = ["network", "mind_control"]\n')
    r = static_review(p)
    assert Capability.NETWORK in r.capabilities
    assert r.unknown_caps == ("mind_control",)
    bare = static_review(_write(tmp_path, "bare.py", "x = 1\n"))
    assert bare.declared is False and bare.capabilities == frozenset()


def test_digest_changes_with_content(tmp_path):
    p = _write(tmp_path, "p.py", "CAPABILITIES = []\n")
    d1 = file_digest(p)
    p.write_text("CAPABILITIES = ['network']\n", encoding="utf-8")
    assert file_digest(p) != d1                   # an edit re-triggers review


# -- approval store -----------------------------------------------------------

def test_approval_is_bound_to_the_digest(tmp_path):
    store = PluginApprovalStore(tmp_path / "approvals.json")
    p = _write(tmp_path, "p.py", "CAPABILITIES = ['network']\n")
    r = static_review(p)
    assert store.is_approved(r.stem, r.digest) is False
    store.approve(r)
    assert store.is_approved(r.stem, r.digest) is True
    # editing the plugin invalidates approval (approve-then-swap can't sneak in)
    p.write_text("CAPABILITIES = ['write_files']\n", encoding="utf-8")
    assert store.is_approved(r.stem, file_digest(p)) is False


# -- the loader gate: unapproved code is not imported -------------------------

_PLUGIN = ('from skills.base import Skill, SkillResult\n'
           'CAPABILITIES = ["network"]\n'
           'class PingSkill(Skill):\n'
           '    name = "plugin_ping"\n'
           '    patterns = []\n'
           '    async def execute(self, request):\n'
           '        return SkillResult("pong")\n')


def test_unapproved_plugin_is_held_not_loaded(tmp_path):
    _write(tmp_path, "pinger.py", _PLUGIN)
    store = PluginApprovalStore(tmp_path / "approvals.json")
    reg = SkillRegistry()
    loaded = load_plugins(reg, {}, plugins_dir=tmp_path,
                          require_approval=True, approval_store=store)
    assert loaded == [] and reg.get("plugin_ping") is None    # held, not imported


def test_approved_plugin_loads(tmp_path):
    p = _write(tmp_path, "pinger.py", _PLUGIN)
    store = PluginApprovalStore(tmp_path / "approvals.json")
    store.approve(static_review(p))
    reg = SkillRegistry()
    loaded = load_plugins(reg, {}, plugins_dir=tmp_path,
                          require_approval=True, approval_store=store)
    assert reg.get("plugin_ping") is not None and "pinger:plugin_ping" in loaded


def test_gate_off_loads_as_before(tmp_path):
    _write(tmp_path, "pinger.py", _PLUGIN)
    reg = SkillRegistry()
    load_plugins(reg, {}, plugins_dir=tmp_path)   # require_approval defaults off
    assert reg.get("plugin_ping") is not None


# -- runtime confinement: a plugin skill can't use an undeclared capability ----

def test_plugin_skill_undeclared_capability_is_denied():
    from core.config import load_settings
    s = load_settings()
    engine = PolicyEngine(s.security, s.safety, None)
    # A plugin that declared only control_input tries to reach the network.
    d = engine.check(ActionRequest(actor=Actor.PLUGIN, capability=Capability.NETWORK,
                                   subject="pinger",
                                   declared=frozenset({Capability.CONTROL_INPUT})))
    assert d.denied() and d.code == "undeclared"


# -- generated code is never auto-installed -----------------------------------

def test_generated_code_is_not_auto_applied_by_default():
    from core.config import load_settings
    s = load_settings()
    # self-dev proposes in a worktree + runs tests; applying is an explicit human
    # step. The safety-critical flag is auto_apply — it must stay OFF so generated
    # code is never installed without a human saying yes (core/self_dev.py).
    assert s.self_dev.auto_apply is False
    assert s.app_builder.enabled in (True, False)   # exists; also not auto-run


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
