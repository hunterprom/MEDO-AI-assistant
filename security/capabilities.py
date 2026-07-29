"""The capability vocabulary + the ``controls_pc`` bridge.

A **capability** names a KIND of side effect — write a file, control input, shut
the machine down. Every skill, plugin, and device DECLARES the capabilities it
needs; the policy engine (``security.policy``) grants only declared + permitted
ones. Read-only sensing (weather, the time, seeing the camera/screen) declares
NOTHING and is always allowed — least privilege applies to *acting*, not
answering.

This module is pure data + two tiny, duck-typed helpers. It imports nothing from
the rest of MEDO (so ``skills.base`` can import it with no cycle), and reads a
skill only via ``getattr`` — never by type.

Migration note (see docs/Decisions.md, S1): the coarse ``Skill.controls_pc``
bit stays authored per skill and keeps working. ``capabilities`` is the richer,
additive declaration; a skill that hasn't declared any yet is bridged to
``LEGACY_ACTUATION`` so the engine gates it exactly as ``controls_pc`` does
today. The invariant ``controls_pc == bool(effective & ACTUATION_CAPS)`` is
asserted by a test over the live registry, so a future migration can't silently
change what the PC-control switch gates.
"""

from __future__ import annotations

from enum import Enum


class Capability(str, Enum):
    """What a side-effectful action DOES. String-valued so it round-trips cleanly
    through config (``requires_confirmation`` maps by ``.value``) and the audit
    log without a custom encoder."""

    READ_FILES = "read_files"          # read inside the path whitelist
    WRITE_FILES = "write_files"        # create/modify inside the whitelist
    RUN_COMMAND = "run_command"        # launch an app / install a package
    CONTROL_INPUT = "control_input"    # synthetic keystrokes & mouse
    POWER_CONTROL = "power_control"    # lock / sleep / shut down the MACHINE
    SESSION_CONTROL = "session_control"  # end/restart MEDO's OWN session
    NETWORK = "network"                # outbound HTTP
    CONTROL_BROWSER = "control_browser"  # drive an already-open browser
    CONTROL_DEVICE = "control_device"  # command a paired LAN device
    USE_CLOUD_BRAIN = "use_cloud_brain"  # send a turn to a cloud LLM (+ egress)
    INSTALL_PLUGIN = "install_plugin"  # load/register a plugin
    MODIFY_SELF = "modify_self"        # apply self-generated code
    #: Bridge bucket for a not-yet-migrated skill that only sets
    #: ``controls_pc=True``. Treated as generic actuation (gated by the
    #: PC-control master switch) until it declares finer capabilities. NEVER
    #: declared by hand.
    LEGACY_ACTUATION = "legacy_actuation"


#: Capabilities that ACT on the computer/OS — gated by the PC-control master
#: switch (``settings.safety.pc_control_enabled``), mirroring today's
#: ``controls_pc`` choke point. ``SESSION_CONTROL`` (closing MEDO itself),
#: ``NETWORK``, and ``USE_CLOUD_BRAIN`` are deliberately NOT here: they don't
#: actuate the machine, so the PC-control switch must never hide closing MEDO,
#: fetching a URL, or talking to a cloud brain (matching today's controls_pc=False
#: on QuitSkill / WebFetchSkill).
ACTUATION_CAPS: frozenset[Capability] = frozenset({
    Capability.WRITE_FILES,
    Capability.RUN_COMMAND,
    Capability.CONTROL_INPUT,
    Capability.POWER_CONTROL,
    Capability.CONTROL_BROWSER,
    Capability.CONTROL_DEVICE,
    Capability.INSTALL_PLUGIN,
    Capability.MODIFY_SELF,
    Capability.LEGACY_ACTUATION,
})


#: Capabilities that DO something to the world (vs. purely reading local files).
#: The trust boundary (S3) treats an untrusted-content-proposed action in this
#: set as confirm-or-deny, never silent.
SIDE_EFFECT_CAPS: frozenset[Capability] = ACTUATION_CAPS | frozenset({
    Capability.NETWORK,
    Capability.USE_CLOUD_BRAIN,
    Capability.SESSION_CONTROL,
})


def effective_capabilities(skill: object) -> frozenset[Capability]:
    """The capabilities a skill effectively holds, bridging un-migrated skills.

    Explicit ``skill.capabilities`` wins; otherwise a ``controls_pc=True`` skill
    is treated as generic ``LEGACY_ACTUATION`` (so it stays gated by the
    PC-control switch exactly as today); a sensing skill holds nothing.
    """
    declared = getattr(skill, "capabilities", None)
    if declared:
        return frozenset(declared)
    if getattr(skill, "controls_pc", False):
        return frozenset({Capability.LEGACY_ACTUATION})
    return frozenset()


def actuates(skill: object) -> bool:
    """True if the skill holds any actuation capability — the derived truth
    behind ``controls_pc``. Used by the consistency invariant test."""
    return bool(effective_capabilities(skill) & ACTUATION_CAPS)
