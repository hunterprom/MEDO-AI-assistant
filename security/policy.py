"""The central policy engine — MEDO's single deny-by-default authority.

It answers ONE question: *is THIS actor allowed to do THIS action (capability)
on THIS target, and does it need confirmation?* → ``Decision(allow|confirm|deny)``
plus a plain-language reason and a machine ``code`` (so the router can keep the
exact user-facing wording, e.g. the PC-control-off reply).

Design contract (docs/Decisions.md, S1):
- **Pure.** ``check`` is a deterministic function of its injected inputs (the
  security + safety config and the path whitelist) and the request. It performs
  NO prompting, execution, or logging — those live at the call sites.
- **Fail safe.** ANY exception inside a check degrades to DENY, never allow.
- **Baseline gate always on.** The actuation master switch
  (``safety.pc_control_enabled``) is enforced even when ``security.enabled`` is
  False — that flag only turns OFF the *new* policy layer, never the existing
  safety floor.
- **S1 mirrors today.** The router enforces only DENY from :meth:`gate_skill`
  (identical to today's ``controls_pc`` check); CONFIRM decisions are defined and
  unit-tested here but stay skill-driven until S3 wires them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from security.capabilities import (
    ACTUATION_CAPS,
    HIGH_IMPACT_CAPS,
    SIDE_EFFECT_CAPS,
    Capability,
    effective_capabilities,
)

logger = logging.getLogger(__name__)

#: Generic reason for a PC-control-off denial. The router maps
#: ``code == "pc_control_off"`` to its own exact spoken reply, so this string is
#: only ever seen in logs/tests.
PC_CONTROL_OFF_REASON = "PC control is switched off"


class Effect(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


class Actor(str, Enum):
    """Who is requesting the action. The MODEL is never an authority — an action
    it proposes is re-checked here exactly like any other (the post-LLM gate)."""

    OWNER = "owner"            # the verified primary user (S2)
    LOCAL_USER = "local_user"  # voice/text at the machine (today's default trust)
    LAN_CLIENT = "lan_client"  # a token-authed watch/phone
    DEVICE = "device"          # a paired MEDO Link device
    PLUGIN = "plugin"          # loaded plugin code
    MODEL = "model"            # the LLM proposing an action


class Provenance(str, Enum):
    """Where the REQUEST originated. USER = the authenticated user's own words
    (trusted). UNTRUSTED = derived from external content (a document, web page,
    tool output, device payload, memory) — the trust boundary (S3)."""

    USER = "user"
    UNTRUSTED = "untrusted"


_RANK = {Effect.ALLOW: 0, Effect.CONFIRM: 1, Effect.DENY: 2}


@dataclass(frozen=True)
class Decision:
    """The engine's verdict. ``code`` is a stable machine tag; ``reason`` is
    human/audit text."""

    effect: Effect
    capability: Capability | None
    reason: str
    code: str = ""

    def allowed(self) -> bool:
        return self.effect is Effect.ALLOW

    def denied(self) -> bool:
        return self.effect is Effect.DENY

    def needs_confirmation(self) -> bool:
        return self.effect is Effect.CONFIRM


@dataclass(frozen=True)
class ActionRequest:
    """One request for the engine to judge."""

    actor: Actor
    capability: Capability
    target: str | None = None                 # path / url / device-id / plugin
    provenance: Provenance = Provenance.USER
    subject: str | None = None                # the skill/plugin/device name
    #: The capabilities the subject DECLARED. None = skip the declaration check
    #: (used by :meth:`gate_skill`, which passes the skill's own effective set,
    #: so the least-privilege deny can never fire against the skill's own caps).
    declared: frozenset[Capability] | None = None


class PolicyEngine:
    """The single authority. Construct once with the live config + whitelist;
    it reads them fresh on every check, so a runtime toggle of the PC-control
    switch is honored immediately."""

    def __init__(self, security: Any, safety: Any, whitelist: Any = None) -> None:
        self._security = security      # SecurityConfig
        self._safety = safety          # SafetyConfig (for pc_control_enabled)
        self._whitelist = whitelist    # PathWhitelist | None

    # -- the one question -----------------------------------------------------

    def check(self, req: ActionRequest) -> Decision:
        """Judge one action. Never raises — any error fails CLOSED (deny)."""
        try:
            return self._check(req)
        except Exception:  # fail safe: a broken check must deny, never allow
            logger.exception("policy check errored; denying to stay safe: %r", req)
            return Decision(Effect.DENY, getattr(req, "capability", None),
                            "security check failed — denied to stay safe",
                            code="check_error")

    def _check(self, req: ActionRequest) -> Decision:
        cap = req.capability

        # BASELINE (always on, even when security.enabled is False): the
        # actuation master switch. Equivalent to today's
        # `controls_pc and not pc_control_enabled`.
        if cap in ACTUATION_CAPS and not self._pc_control_enabled():
            return Decision(Effect.DENY, cap, PC_CONTROL_OFF_REASON,
                            code="pc_control_off")

        # Escape hatch: with the new layer off, keep ONLY the baseline gate.
        if not self._enabled():
            return Decision(Effect.ALLOW, cap,
                            "security policy disabled (baseline gating only)",
                            code="legacy")

        # 1. Least privilege / deny by default: the subject must have DECLARED
        #    this capability.
        if req.declared is not None and cap not in req.declared:
            who = req.subject or "this action"
            return Decision(Effect.DENY, cap,
                            f"{who} did not declare {cap.value}", code="undeclared")

        # 1b. Owner-voice gate (S2): with owner_voice on, a HIGH-IMPACT capability
        #     needs the VERIFIED owner — not just any voice in the room. Off by
        #     default, so everyday use isn't gated on enrolment.
        if (self._owner_voice_required() and cap in HIGH_IMPACT_CAPS
                and req.actor is not Actor.OWNER):
            return Decision(Effect.DENY, cap,
                            "that needs the owner's voice", code="owner_required")

        # 2. Target check: files must be inside the whitelist; URLs must be
        #    http(s). (Folds the existing PathWhitelist policy into the engine.)
        target_deny = self._target_denied(req)
        if target_deny is not None:
            return target_deny

        # 3. Trust boundary (S3): an action PROPOSED from untrusted content is
        #    never silent — confirm or deny per policy.
        if req.provenance is Provenance.UNTRUSTED and cap in SIDE_EFFECT_CAPS:
            if self._untrusted_policy() == "deny":
                return Decision(Effect.DENY, cap,
                                "action proposed by external content — refused",
                                code="untrusted_deny")
            return Decision(Effect.CONFIRM, cap,
                            "that came from content, not from you — confirm first",
                            code="untrusted_confirm")

        # 4. Confirmation map for high-impact capabilities.
        if self._requires_confirmation(cap):
            return Decision(Effect.CONFIRM, cap,
                            f"{cap.value.replace('_', ' ')} needs your confirmation",
                            code="confirm")

        # 5. Allowed.
        return Decision(Effect.ALLOW, cap, "permitted", code="ok")

    # -- convenience: gate a whole skill --------------------------------------

    def gate_skill(self, skill: object, *, actor: Actor = Actor.LOCAL_USER,
                   provenance: Provenance = Provenance.USER) -> Decision:
        """Fold the decisions over a skill's effective capabilities into the
        STRONGEST one (deny > confirm > allow). A sensing skill (no caps) is
        always allowed. The skill's own effective set is passed as ``declared``,
        so this never denies a skill for a capability it legitimately holds — the
        only DENY it can return in S1 is the PC-control-off baseline gate, exactly
        like today's router check."""
        caps = effective_capabilities(skill)
        if not caps:
            return Decision(Effect.ALLOW, None, "read-only", code="sensing")
        worst = Decision(Effect.ALLOW, None, "permitted", code="ok")
        subject = getattr(skill, "name", None)
        for cap in caps:
            d = self.check(ActionRequest(
                actor=actor, capability=cap, provenance=provenance,
                declared=caps, subject=subject))
            if _RANK[d.effect] > _RANK[worst.effect]:
                worst = d
        return worst

    # -- internals (read config live) -----------------------------------------

    def _enabled(self) -> bool:
        return bool(getattr(self._security, "enabled", True))

    def _owner_voice_required(self) -> bool:
        return bool(getattr(self._security, "owner_voice", False))

    def _pc_control_enabled(self) -> bool:
        # Default True mirrors SafetyConfig's default; a missing safety config
        # must not itself DENY everything (that would be its own outage).
        return bool(getattr(self._safety, "pc_control_enabled", True))

    def _untrusted_policy(self) -> str:
        return str(getattr(self._security, "untrusted_action_policy", "confirm"))

    def _requires_confirmation(self, cap: Capability) -> bool:
        table = getattr(self._security, "requires_confirmation", None) or {}
        return bool(table.get(cap.value, False))

    def _target_denied(self, req: ActionRequest) -> Decision | None:
        cap = req.capability
        if cap in (Capability.READ_FILES, Capability.WRITE_FILES):
            if self._whitelist is not None and req.target is not None:
                if not self._whitelist.is_allowed(req.target):
                    return Decision(Effect.DENY, cap,
                                    "that path is outside the allowed folders",
                                    code="outside_whitelist")
        elif cap is Capability.NETWORK and req.target is not None:
            low = req.target.strip().lower()
            if not (low.startswith("http://") or low.startswith("https://")):
                return Decision(Effect.DENY, cap,
                                "only http(s) addresses can be fetched",
                                code="bad_scheme")
        return None
