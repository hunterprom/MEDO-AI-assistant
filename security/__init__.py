"""MEDO's central security layer.

One deny-by-default policy engine (``security.policy``) is the single authority
every side-effectful action routes through, over a small capability vocabulary
(``security.capabilities``). Built defensively: it protects MEDO and the user,
never attacks, and fails CLOSED (deny) on any error.

S1 ships the pure engine + capability model + config, and folds the EXISTING
actuation master-switch gate (``safety.pc_control_enabled``) through it. Later
sub-steps add owner identity (S2), the prompt-injection trust boundary (S3),
plugin sandboxing (S4), secrets/egress (S5), and the audit log + HUD (S6).
"""

from security.capabilities import (
    ACTUATION_CAPS,
    HIGH_IMPACT_CAPS,
    SIDE_EFFECT_CAPS,
    Capability,
    effective_capabilities,
)
from security.policy import (
    ActionRequest,
    Actor,
    Decision,
    Effect,
    PolicyEngine,
    Provenance,
)

__all__ = [
    "ACTUATION_CAPS",
    "HIGH_IMPACT_CAPS",
    "SIDE_EFFECT_CAPS",
    "Capability",
    "effective_capabilities",
    "ActionRequest",
    "Actor",
    "Decision",
    "Effect",
    "PolicyEngine",
    "Provenance",
]
