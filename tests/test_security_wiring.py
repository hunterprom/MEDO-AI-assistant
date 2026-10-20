"""S1 integration: the policy engine is wired into the router + registry.

The load-bearing check is the CONSISTENCY INVARIANT: for every registered skill,
the coarse ``controls_pc`` bit must equal "holds an actuation capability". That
guarantees decomposing ``controls_pc`` into fine-grained ``capabilities`` never
silently changes what the PC-control switch gates — a future migration that
declares an actuation capability without ``controls_pc`` (or vice versa) fails
here, not in production.
"""

from __future__ import annotations

from core.config import load_settings
from core.events import EventBus
from core.router import Router
from llm.client import OllamaClient
from security.capabilities import Capability, actuates, effective_capabilities
from security.policy import Actor, PolicyEngine


def _registry():
    from core.docindex import DocumentIndex
    from main import Announcer, build_registry

    settings = load_settings()
    return settings, build_registry(
        settings, Announcer(), doc_index=DocumentIndex(":memory:", None, []))


def test_controls_pc_matches_actuation_capability_for_every_skill():
    _settings, reg = _registry()
    mismatched = [s.name for s in reg.all()
                  if bool(s.controls_pc) != actuates(s)]
    assert mismatched == [], (
        "controls_pc must equal 'has an actuation capability' — these drifted: "
        f"{mismatched}")


def test_migrated_skills_declare_their_capabilities():
    _settings, reg = _registry()
    caps = {s.name: effective_capabilities(s) for s in reg.all()}
    assert Capability.POWER_CONTROL in caps["power"]
    assert Capability.SESSION_CONTROL in caps["quit"]
    assert Capability.WRITE_FILES in caps["edit_file"]
    # closing MEDO is not machine actuation -> not gated by the PC-control switch
    assert not actuates(reg.get("quit"))


def test_router_builds_a_policy_engine():
    settings = load_settings()
    r = Router(settings, __import__("skills.base", fromlist=["SkillRegistry"])
               .SkillRegistry(), OllamaClient(settings.llm), EventBus())
    assert isinstance(r._policy, PolicyEngine)


def test_actuation_skill_is_denied_when_pc_control_off():
    settings = load_settings()
    settings.safety.pc_control_enabled = False
    engine = PolicyEngine(settings.security, settings.safety, whitelist=None)
    _settings, reg = _registry()
    power, weather = reg.get("power"), reg.get("weather")
    assert engine.gate_skill(power).denied()               # actuation -> blocked
    assert engine.gate_skill(power).code == "pc_control_off"
    if weather is not None:
        assert engine.gate_skill(weather).allowed()        # sensing -> unaffected


def test_router_resolves_the_acting_principal():
    # owner_verified -> OWNER; a remote client -> LAN_CLIENT; else LOCAL_USER.
    assert Router._actor_for({"owner_verified": True}) is Actor.OWNER
    assert Router._actor_for({"source": "remote"}) is Actor.LAN_CLIENT
    assert Router._actor_for({"source": "watch"}) is Actor.LAN_CLIENT
    assert Router._actor_for({"language": "en"}) is Actor.LOCAL_USER
    assert Router._actor_for(None) is Actor.LOCAL_USER


def test_owner_voice_gates_high_impact_end_to_end():
    # With owner_voice on, a local (unverified) user can't power off; the owner can.
    settings = load_settings()
    settings.security.owner_voice = True
    engine = PolicyEngine(settings.security, settings.safety, whitelist=None)
    _settings, reg = _registry()
    power = reg.get("power")
    assert engine.gate_skill(power, actor=Actor.LOCAL_USER).denied()
    assert engine.gate_skill(power, actor=Actor.LOCAL_USER).code == "owner_required"
    assert not engine.gate_skill(power, actor=Actor.OWNER).denied()
