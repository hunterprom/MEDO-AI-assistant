"""Control OTHER local software by command — the connector abstraction (S1).

This is "MEDO Link, but for local apps instead of hardware devices", and it is
DIRECT LOCAL CONTROL — **not** MCP, no servers, no protocol. Each connector
declares its app + a list of :class:`Action`s; the registry turns every action
into a routable skill (fast-path patterns + an LLM tool), exactly like MEDO Link
turns device capabilities into skills. Every action is gated by the central
policy engine and only ever triggered by the user's own voice/text.

The control ladder each action prefers (robust → brittle), degrading gracefully:
  1. the app's native local API / CLI
  2. app-specific keyboard shortcuts sent to its focused window
  3. UI automation over the OS accessibility tree (reuses the M7 pointer-snap work)
If none is available for a request, MEDO says so plainly — it never fakes success.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Tuple

from security.capabilities import Capability

#: Mechanism identifiers, in the ladder's robustness order.
API = "api"
CLI = "cli"
HOTKEY = "hotkey"
ACCESSIBILITY = "accessibility"


@dataclass(frozen=True)
class DetectResult:
    installed: bool
    running: bool
    detail: str = ""


@dataclass(frozen=True)
class ActionResult:
    """What an action reports back. ``verified`` is the honesty flag: True only
    when the mechanism could CONFIRM the effect (e.g. the window really focused);
    False means "attempted, but I can't be sure" — surfaced truthfully, never
    dressed up as success."""

    success: bool
    speech: str
    verified: bool = True
    data: Dict = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """One thing a connector can do. Its ``description`` becomes the LLM-tool
    description; ``phrases`` seed the fast path (may contain ``{app}``)."""

    name: str
    description: str
    phrases: Tuple[str, ...]
    mechanisms: Tuple[str, ...]
    capabilities: FrozenSet[Capability]
    run: Callable[[Dict], ActionResult]
    requires_confirmation: bool = False
    confirm_prompt: str = ""          # bilingual gate text; falls back to a default
    #: Optional LLM-tool parameters, {name: {"type":..., "description":...}}.
    params: Dict[str, dict] = field(default_factory=dict)


class SoftwareConnector(ABC):
    """Base for every app connector. Subclasses set ``app_id`` / ``display_name``
    and implement :meth:`detect` + :meth:`actions`. Shared lifecycle primitives
    (is_installed / is_running / launch / focus / close) come from the injected
    mechanisms, so a connector never touches the OS directly."""

    app_id: str = ""
    display_name: str = ""
    #: Executable names used to detect "is it running" + to launch it. Overridable
    #: from config for non-standard install paths.
    executables: Tuple[str, ...] = ()
    #: Window-title substring used to find/focus the app's window.
    window_hint: str = ""

    def __init__(self, mechanisms) -> None:
        self._mech = mechanisms       # software.mechanisms.Mechanisms

    @abstractmethod
    def detect(self) -> DetectResult: ...

    @abstractmethod
    def actions(self) -> List[Action]: ...

    # -- shared lifecycle (all via injected mechanisms) -----------------------

    def is_installed(self) -> bool:
        return self.detect().installed

    def is_running(self) -> bool:
        return self._mech.is_process_running(self.executables)

    def launch(self) -> bool:
        return self._mech.launch(self.app_id)

    def focus(self) -> bool:
        return self._mech.focus_app(self.window_hint or self.display_name)
