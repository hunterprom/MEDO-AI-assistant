"""Turn connector actions into routable skills (S1 + S3).

Each :class:`Action` becomes a :class:`ConnectorActionSkill` — the SAME pattern
MEDO Link uses to turn a device capability into a skill — so an action is
automatically (a) a fast-path regex route and (b) an LLM tool, and it flows
through the router's policy gate like every other skill (its declared
capabilities are what ``effective_capabilities`` reads). No forked routing.

Graceful states + the trust boundary live in ``execute``: not installed → say so;
not running → open it; an instruction from untrusted content (a document, a web
page, another app's output) → refused, never acted on.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List

from security.capabilities import ACTUATION_CAPS
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult
from software.connector_base import Action, SoftwareConnector

logger = logging.getLogger(__name__)


class ConnectorActionSkill(Skill):
    """One connector action, exposed as a fast-path + LLM-tool skill."""

    def __init__(self, connector: SoftwareConnector, action: Action) -> None:
        self._connector = connector
        self._action = action
        self.name = f"{connector.app_id}_{action.name}"
        self.description = action.description
        self.capabilities = frozenset(action.capabilities)
        self.requires_confirmation = bool(action.requires_confirmation)
        # controls_pc is the derived truth (keeps the security invariant): a
        # connector action that actuates input/process is PC control.
        self.controls_pc = bool(self.capabilities & ACTUATION_CAPS)
        self.patterns = [re.compile(self._expand(p), re.IGNORECASE)
                         for p in action.phrases]

    def _expand(self, phrase: str) -> str:
        # {app} in a phrase means "the app's display name", matched literally.
        if "{app}" in phrase:
            return phrase.replace("{app}", re.escape(self._connector.display_name))
        return phrase

    async def execute(self, request: SkillRequest) -> SkillResult:
        ctx = request.context or {}
        # --- TRUST BOUNDARY: only the user's own voice/text drives software ---
        if ctx.get("untrusted") or ctx.get("provenance") == "untrusted":
            return SkillResult(
                "I only control apps when you ask me directly — not when it "
                "comes from a document, a web page, or another app.",
                success=False, data={"refused": "untrusted"})

        # detect() can walk Program Files / query psutil — blocking OS I/O. Keep
        # it (and every mechanism call below) OFF the event loop so the voice loop
        # and companion API stay responsive on the fast path.
        det = await asyncio.to_thread(self._connector.detect)
        if not det.installed:
            return SkillResult(f"{self._connector.display_name} isn't installed "
                               f"on this computer.", success=False,
                               data={"reason": "not_installed"})

        # --- confirmation for high-impact / any send-post (bilingual) ---------
        if self.requires_confirmation and not ctx.get("confirmed"):
            prompt = self._action.confirm_prompt or (
                f"Do you want me to {self._action.name.replace('_', ' ')} in "
                f"{self._connector.display_name}? Say yes to confirm.")
            return SkillResult(prompt, needs_confirmation=True)

        # --- ensure the app is available (open it if needed) ------------------
        if not await asyncio.to_thread(self._connector.is_running):
            if not await asyncio.to_thread(self._connector.launch):
                return SkillResult(
                    f"{self._connector.display_name} isn't open and I couldn't "
                    f"open it.", success=False, data={"reason": "launch_failed"})

        # --- run through the chosen mechanism; report honestly ----------------
        # Params come from the fast-path regex groups (e.g. "close <app>") AND
        # the LLM tool args, so an action works on both routes.
        params: Dict[str, Any] = {}
        if request.match is not None:
            params.update({k: v for k, v in request.match.groupdict().items()
                           if v is not None})
        params.update(request.args or {})
        result = await asyncio.to_thread(self._action.run, params)
        return SkillResult(result.speech, success=result.success,
                           data={**result.data, "verified": result.verified,
                                 "app": self._connector.app_id})

    def tool_schema(self) -> Dict[str, Any]:
        props = {n: {"type": p.get("type", "string"),
                     "description": p.get("description", "")}
                 for n, p in (self._action.params or {}).items()}
        required = list(props.keys())
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": props,
                               "required": required},
            },
        }


def build_software_skills(settings, mechanisms) -> List[Skill]:
    """Instantiate the enabled connectors and expose their actions as skills.
    Empty when ``software.enabled`` is off — restoring prior behaviour."""
    sw = getattr(settings, "software", None)
    if sw is None or not getattr(sw, "enabled", False):
        return []
    from software.connectors import CONNECTOR_CLASSES

    enabled = getattr(sw, "connectors", None) or {}
    skills: List[Skill] = []
    for cls in CONNECTOR_CLASSES:
        app_id = getattr(cls, "app_id", "")
        if enabled and enabled.get(app_id) is False:      # explicit opt-out
            continue
        try:
            connector = cls(mechanisms)
            for action in connector.actions():
                skills.append(ConnectorActionSkill(connector, action))
        except Exception:
            logger.warning("connector %s failed to load", app_id, exc_info=True)
    return skills


def register_software(registry: SkillRegistry, settings, mechanisms) -> int:
    """Register connector skills into the live registry. Returns the count."""
    skills = build_software_skills(settings, mechanisms)
    for skill in skills:
        try:
            registry.register(skill)
        except ValueError:            # duplicate name — skip, don't crash startup
            logger.warning("duplicate connector skill %s", skill.name)
    return len(skills)
