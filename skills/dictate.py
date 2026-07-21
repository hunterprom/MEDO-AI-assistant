"""Dictation mode — MEDO writes down what you say instead of answering it.

"take dictation" flips the voice loop into a mode where every utterance is
transcribed and appended to a file, with no wake word between lines and no
routing, until you say "stop dictation". That's the difference from the
``edit_file`` skill: this is a *mode* for composing at length, not a one-shot
"add this line to that file".

The file is resolved once, when the mode starts, so a mis-heard filename can't
scatter half a paragraph across the disk mid-flow. It's created if missing, but
only ever inside ``conversation.dictation_dir``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from core import mk
from core.config import Settings, expand_path
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

#: Ends the mode. Matched by the voice loop on the raw transcript, so it works
#: without the wake word and never reaches the router.
STOP_DICTATION = re.compile(
    r"\b(?:stop|end|finish|quit)\s+(?:the\s+)?dictation\b"
    r"|\bstop\s+(?:writing|dictating)\b"
    r"|\bdictation\s+off\b"
    r"|\b(?:стоп|прекини|заврши)\s+(?:со\s+)?(?:диктат|диктирање|пишување)\b"
    r"|\bдоста\s+пишување\b",
    re.IGNORECASE,
)


def target_path(settings: Settings, name: str = "") -> Path:
    """Where dictation goes: a named file, else the configured default.

    Always inside ``conversation.dictation_dir`` — a spoken filename is never
    trusted as a path, so directory separators are stripped rather than
    followed.
    """
    directory = expand_path(settings.conversation.dictation_dir)
    raw = (name or settings.conversation.dictation_file).strip().strip("\"' ")
    # Spoken names arrive as "my notes" or "notes dot md"; keep the basename
    # only, so nothing can climb out of the dictation directory.
    raw = re.sub(r"\s+(?:dot|точка)\s+", ".", raw, flags=re.IGNORECASE)
    raw = raw.replace("\\", "/").split("/")[-1].strip() or \
        settings.conversation.dictation_file
    if "." not in raw:
        raw = f"{raw}.md"
    return directory / raw


class DictateSkill(Skill):
    """Start dictation: "take dictation", "write this down for me"."""

    name = "dictation"
    controls_pc = True
    description = (
        "Start dictation mode: MEDO writes down everything the user says, into "
        "a file, until they say stop. Use when the user wants text WRITTEN, "
        "not answered."
    )

    patterns = [
        re.compile(r"\b(?:take|start|begin)\s+(?:a\s+|the\s+)?dictation\b"
                   r"(?:\s+(?:in|into|to)\s+(?:the\s+)?(?:file\s+)?(?P<file>.+))?$",
                   re.IGNORECASE),
        re.compile(r"\b(?:start|begin)\s+dictating\b"
                   r"(?:\s+(?:in|into|to)\s+(?:the\s+)?(?P<file2>.+))?$",
                   re.IGNORECASE),
        re.compile(r"\bwrite\s+(?:this|it)\s+down(?:\s+for\s+me)?\b"
                   r"(?:\s+(?:in|into|to)\s+(?:the\s+)?(?P<file3>.+))?$",
                   re.IGNORECASE),
        re.compile(r"\btake\s+(?:this\s+)?down\s+for\s+me\b", re.IGNORECASE),
        # MK: "земи диктат", "запишувај го ова", "пиши што ќе кажам"
        re.compile(r"\b(?:земи|почни|започни)\s+(?:со\s+)?дикт(?:ат|ирање)\b"
                   r"(?:\s+во\s+(?P<filem>.+))?$", re.IGNORECASE),
        re.compile(r"\bзапишувај\b|\bпиши\s+што\s+(?:ќе\s+)?кажам\b",
                   re.IGNORECASE),
    ]

    def __init__(self, settings: Settings, modes) -> None:
        self._settings = settings
        self._modes = modes

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        name = (request.args.get("file") or gd.get("file") or gd.get("file2")
                or gd.get("file3") or gd.get("filem") or "")
        path = target_path(self._settings, name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("", encoding="utf-8")
        except OSError as exc:
            logger.warning("dictation target unusable: %s", exc)
            return SkillResult(
                f"Не можам да пишувам во {path.name}." if speak_mk
                else f"I can't write to {path.name}.", success=False)

        self._modes.dictating = True
        self._modes.dictation_path = str(path)
        return SkillResult(
            f"Слушам и запишувам во {path.name}. Кажи „стоп диктат“ кога ќе завршиш."
            if speak_mk else
            f"Go ahead — I'm writing to {path.name}. Say 'stop dictation' when "
            f"you're done.",
            data={"path": str(path), "dictating": True})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string",
                                 "description": "File to dictate into. Optional."},
                    },
                    "required": [],
                },
            },
        }
