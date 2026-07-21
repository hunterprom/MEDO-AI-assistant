"""MEDO Lion Mode — stop asking, just do it.

While Lion Mode is on the router skips the spoken confirmation before
destructive actions and ignores the PC-control switch. It is the "I know what
I'm doing, get out of the way" mode.

What it deliberately does NOT do, because these bound *where* MEDO can act
rather than *whether it asks first*:

* the file whitelist still applies — a mis-heard filename must not reach your
  system drive just because you stopped wanting prompts;
* ``browser.blocked_domains`` still applies;
* the text-only rule on file edits still applies, and edits still leave a .bak.

Turning it ON asks for confirmation, which is not a joke: it is the last
prompt you will get, so it is the one worth answering deliberately. Turning it
OFF never asks. It also resets to off on restart — a mode for a task, not a
setting you forget you left on.
"""

from __future__ import annotations

import re
from typing import Any

from core import mk
from core.config import Settings
from skills.base import Skill, SkillRequest, SkillResult

ON_REPLY = (
    "Lion mode on. I won't ask before acting, and the PC control switch no "
    "longer stops me. Say 'lion mode off' when you're done."
)
ON_REPLY_MK = (
    "Лав мод е вклучен. Нема да прашувам пред да дејствувам. Кажи „лав мод "
    "исклучи“ кога ќе завршиш."
)
OFF_REPLY = "Lion mode off. I'll ask before destructive actions again."
OFF_REPLY_MK = "Лав мод е исклучен. Пак ќе прашувам пред опасни дејства."


class LionModeSkill(Skill):
    """Toggle MEDO Lion Mode."""

    name = "lion_mode"
    controls_pc = False          # the switch itself isn't an action on the PC
    description = (
        "Turn MEDO Lion Mode on or off. In lion mode MEDO stops asking for "
        "confirmation before destructive actions and ignores the PC control "
        "switch. Only use when the user explicitly asks for it by name."
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
            want_on = not self._settings.safety.lion_mode

        if not want_on:
            self._settings.safety.lion_mode = False
            return SkillResult(OFF_REPLY_MK if speak_mk else OFF_REPLY,
                               data={"lion_mode": False})

        if self._settings.safety.lion_mode:
            return SkillResult("Лав мод е веќе вклучен." if speak_mk
                               else "Lion mode is already on.",
                               data={"lion_mode": True})
        # The last prompt you'll get, so make it count.
        if not request.context.get("confirmed"):
            return SkillResult(
                "Лав мод значи дека нема да прашувам пред опасни дејства. "
                "Сигурен си?" if speak_mk else
                "Lion mode means I stop asking before destructive actions. "
                "Are you sure?", needs_confirmation=True)
        self._settings.safety.lion_mode = True
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
