"""MEDO Lion Mode — the defensive-security expert profile.

Lion mode is a PROFILE, not a bypass. Turning it on:

* reskins the HUD deep red and shows a LION indicator;
* SURFACES a group of read-only, local-machine, advisory security skills
  (see skills/security.py): a listening-port audit, a firewall audit, a
  process explainer, a file-permission explainer, and an update/hygiene check.

What it deliberately does NOT do — and this is the whole point, stated so no
one wires it back the other way:

* it does not touch the confirmation gate. MEDO still asks before every
  destructive action, in lion mode exactly as out of it;
* it does not touch the path whitelist or the PC-control switch;
* it grants no new power to act. Every skill it surfaces is read-only and
  advisory — "here is what I see, you decide."

Turning it on therefore needs no scary confirmation: it lowers no guardrail.
It still resets to off on restart, because it is a mode you enter for a task.

(This replaced an earlier "stop asking me" bypass. An unrestricted mode is a
liability; a defensive-security profile is a real, hireable specialty. See
docs/Decisions.md.)
"""

from __future__ import annotations

import re
from typing import Any

from core import mk
from core.config import Settings
from skills.base import Skill, SkillRequest, SkillResult

ON_REPLY = (
    "Lion mode on. Defensive-security tools are live: security check, firewall "
    "audit, process and permission explainers, and an update check — all "
    "read-only, this machine only. Your safety gate is unchanged; I still ask "
    "before anything destructive. Say 'lion mode off' when you're done."
)
ON_REPLY_MK = (
    "Лав мод е вклучен. Одбранбените алатки се активни: безбедносна проверка, "
    "проверка на заштитниот ѕид, објаснувања за процеси и дозволи, и проверка "
    "за надградби — само за читање, само за овој компјутер. Безбедносната "
    "заштита останува иста. Кажи „лав мод исклучи“ кога ќе завршиш."
)
OFF_REPLY = "Lion mode off. The defensive-security tools are put away."
OFF_REPLY_MK = "Лав мод е исклучен. Одбранбените алатки се склонети."


class LionModeSkill(Skill):
    """Toggle MEDO Lion Mode (the defensive-security profile)."""

    name = "lion_mode"
    controls_pc = False          # the switch itself isn't an action on the PC
    description = (
        "Turn MEDO Lion Mode on or off. Lion mode is a defensive-security "
        "profile: it reskins the interface and surfaces read-only, "
        "local-machine security-audit skills. It does NOT change any safety "
        "rule or confirmation. Use when the user asks for it by name."
    )

    _ON = r"on|enable|engage|activate|start"
    _OFF = r"off|disable|stop|end|deactivate"

    patterns = [
        re.compile(rf"\b(?:medo\s+)?lion\s+mode\s+(?P<on>{_ON})\b", re.IGNORECASE),
        re.compile(rf"\b(?:medo\s+)?lion\s+mode\s+(?P<off>{_OFF})\b", re.IGNORECASE),
        re.compile(rf"\b(?:{_ON})\s+(?:medo\s+)?lion\s+mode\b", re.IGNORECASE),
        re.compile(rf"\b(?:{_OFF})\s+(?:medo\s+)?lion\s+mode\b", re.IGNORECASE),
        re.compile(r"\b(?:medo\s+)?lion\s+mode\b\s*[.!?]*$", re.IGNORECASE),
        # MK: "лав мод вклучи" / "исклучи лав мод"
        re.compile(r"\bлав\s+мод\s+(?P<on_mk>вклучи|активирај)\b", re.IGNORECASE),
        re.compile(r"\bлав\s+мод\s+(?P<off_mk>исклучи|стоп)\b", re.IGNORECASE),
        re.compile(r"\b(?P<on_mk2>вклучи|активирај)\s+лав\s+мод\b", re.IGNORECASE),
        re.compile(r"\b(?P<off_mk2>исклучи)\s+лав\s+мод\b", re.IGNORECASE),
    ]

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        arg = request.args.get("state")
        if arg is not None:
            want_on = str(arg).lower() in ("on", "true", "enable", "1")
        elif gd.get("off") or gd.get("off_mk") or gd.get("off_mk2"):
            want_on = False
        elif gd.get("on") or gd.get("on_mk") or gd.get("on_mk2"):
            want_on = True
        else:
            # Bare "lion mode" toggles, which is what people mean when they
            # bark it mid-task.
            want_on = not self._settings.mode.lion

        if not want_on:
            self._settings.mode.lion = False
            return SkillResult(OFF_REPLY_MK if speak_mk else OFF_REPLY,
                               data={"lion_mode": False})

        if self._settings.mode.lion:
            return SkillResult("Лав мод е веќе вклучен." if speak_mk
                               else "Lion mode is already on.",
                               data={"lion_mode": True})
        # No confirmation: lion mode lowers no guardrail, so gating it behind a
        # prompt would only teach the habit of clicking through security
        # prompts. It just enters the profile.
        self._settings.mode.lion = True
        return SkillResult(ON_REPLY_MK if speak_mk else ON_REPLY,
                           data={"lion_mode": True})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "state": {"type": "string", "enum": ["on", "off"]},
                    },
                    "required": ["state"],
                },
            },
        }
