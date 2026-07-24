"""Timers, reminders, and alarms.

Timers are scheduled as asyncio tasks on the main loop. When one fires it calls
the ``announce`` callback the app provides — which prints in text mode and speaks
in voice mode — so reminders reach you without blocking anything.

    "set a timer for 5 minutes"
    "remind me in 10 seconds to stretch"
    "list timers" / "cancel timers"
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from core.memory import ReminderStore
from skills.base import Skill, SkillRequest, SkillResult

Announce = Callable[[str], Awaitable[None] | None]

_UNIT_SECONDS = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}
_DURATION_RE = re.compile(
    r"(\d+)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)", re.IGNORECASE
)

#: A TRAILING delay phrase inside a reminder label — "call mom IN 5 minutes" ->
#: "call mom". Requires a number (digit or word) followed by a time unit, so a
#: task's own "in"/"for" ("fill IN the form", "wait FOR the tone") is untouched.
_TRAILING_DELAY = re.compile(
    r"\s+(?:in|after|within|for)\s+"
    r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"fifteen|twenty|thirty|forty|fifty|sixty|half|quarter|couple|few)\b"
    r"[\w\s-]*?\b(?:second|sec|minute|min|hour|hr|day|week)s?\b.*$"
    r"|\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?\b.*$",
    re.IGNORECASE)

# Whisper (especially large-v3-turbo) transcribes numbers as WORDS — "set a
# timer for one minute" — which the digit-only regex above never matched, so
# every spoken timer got "How long should the timer be?". Normalize word
# numbers to digits before matching.
_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "twenty": 20,
    "twenty five": 25, "thirty": 30, "forty": 40, "forty five": 45,
    "fifty": 50, "sixty": 60, "ninety": 90,
}
_WORD_NUMBER_RE = re.compile(
    r"\b(" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")\s+"
    r"(?=(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\b)",
    re.IGNORECASE,
)
_HALF_HOUR_RE = re.compile(r"\bhalf\s+(?:an\s+)?hour\b", re.IGNORECASE)


def parse_duration(text: str) -> int:
    """Sum every '<n> <unit>' in ``text`` into total seconds (0 if none).

    Accepts digits and spoken word numbers: "5 minutes", "one minute",
    "an hour", "forty five seconds", "half an hour".
    """
    text = _HALF_HOUR_RE.sub("30 minutes", text)
    text = _WORD_NUMBER_RE.sub(
        lambda m: f"{_WORD_NUMBERS[m.group(1).lower()]} ", text)
    total = 0
    for value, unit in _DURATION_RE.findall(text):
        base = unit.lower().rstrip("s")
        base = {"hr": "hour", "hrs": "hour", "min": "minute", "sec": "second"}.get(base, base)
        total += int(value) * _UNIT_SECONDS.get(base, 0)
    return total


@dataclass
class _Timer:
    id: int
    seconds: int
    label: str
    task: asyncio.Task = field(repr=False)
    reminder_id: int | None = None   # row id when persisted (labelled reminders)


class TimerSkill(Skill):
    name = "timers"
    description = "Set countdown timers and reminders."

    patterns = [
        re.compile(r"\bcancel\s+(?:all\s+)?(?:timers?|reminders?|alarms?)\b", re.IGNORECASE),
        re.compile(r"\b(?:list|show)\s+(?:my\s+)?(?:timers?|reminders?|alarms?)\b", re.IGNORECASE),
        re.compile(r"\b(?:set|start)\s+(?:a\s+)?(?:timer|alarm|countdown)\b.*", re.IGNORECASE),
        re.compile(r"\bremind\s+me\b.*", re.IGNORECASE),
        re.compile(r"\btimer\s+for\b.*", re.IGNORECASE),
        # Everyday alarm/timer phrasings (duration is parsed from the whole text).
        re.compile(r"\bwake\s+me\s+(?:up\s+)?in\b.*", re.IGNORECASE),
        re.compile(r"\b(?:ping|buzz|alert|nudge)\s+me\s+in\b.*", re.IGNORECASE),
        re.compile(r"\b(?:let\s+me\s+know|tell\s+me)\s+in\s+\d.*", re.IGNORECASE),
    ]

    def __init__(self, announce: Announce, store: ReminderStore | None = None) -> None:
        self._announce = announce
        self._store = store
        self._timers: dict[int, _Timer] = {}
        self._next_id = 1
        if store is not None:
            self._reschedule_persisted()

    def _reschedule_persisted(self) -> None:
        """Re-arm reminders saved from a previous run (skipped without a loop)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # constructed outside an event loop (e.g. some tests)
        now = datetime.now(timezone.utc).astimezone()
        for rem in self._store.all():  # type: ignore[union-attr]
            try:
                remaining = max(0, int((datetime.fromisoformat(rem.due_at) - now).total_seconds()))
            except ValueError:
                continue
            self._arm(remaining, rem.label, reminder_id=rem.id)

    def _arm(self, seconds: int, label: str, reminder_id: int | None) -> int:
        """Schedule one timer/reminder and track it. Returns its local id."""
        timer_id = self._next_id
        self._next_id += 1
        task = asyncio.create_task(self._fire(timer_id, seconds, label, reminder_id))
        self._timers[timer_id] = _Timer(timer_id, seconds, label, task, reminder_id)
        return timer_id

    async def _fire(
        self, timer_id: int, seconds: int, label: str, reminder_id: int | None = None
    ) -> None:
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        self._timers.pop(timer_id, None)
        if reminder_id is not None and self._store is not None:
            self._store.delete(reminder_id)
        message = f"Reminder: {label}." if label else "Timer's up."
        result = self._announce(message)
        if isinstance(result, Awaitable):
            await result

    def _extract_label(self, text: str) -> str:
        m = re.search(r"\bto\s+(.*)", text, re.IGNORECASE)
        if not m:
            return ""
        label = m.group(1).strip().strip(".!?")
        # Strip a TRAILING delay phrase so the label survives the most common
        # wording "remind me to X in N minutes" — which used to lose X entirely.
        trimmed = _TRAILING_DELAY.sub("", label).strip().strip(",")
        if trimmed and parse_duration(trimmed) == 0:
            return trimmed
        # Nothing meaningful left, or the whole thing was a duration
        # ("set a timer TO 5 minutes") — no label.
        return "" if parse_duration(label) > 0 else label

    async def execute(self, request: SkillRequest) -> SkillResult:
        text = request.text
        # LLM tool path: the model passes structured args the regex can't see.
        args = request.args or {}
        action = str(args.get("action") or "").strip().lower()

        if action == "cancel" or re.search(r"\bcancel\b", text, re.IGNORECASE):
            count = len(self._timers)
            for timer in list(self._timers.values()):
                timer.task.cancel()
                if timer.reminder_id is not None and self._store is not None:
                    self._store.delete(timer.reminder_id)
            self._timers.clear()
            return SkillResult(f"Cancelled {count} timer{'s' if count != 1 else ''}.")

        if action == "list" or re.search(r"\b(list|show)\b", text, re.IGNORECASE):
            if not self._timers:
                return SkillResult("You have no active timers.")
            desc = "; ".join(
                f"{t.id}: {t.label or 'timer'} ({t.seconds}s)" for t in self._timers.values()
            )
            return SkillResult(f"{len(self._timers)} active: {desc}.")

        # Duration + label from the model's args first, then the spoken text.
        seconds = 0
        if args.get("seconds") is not None:
            try:
                seconds = int(args["seconds"])
            except (ValueError, TypeError):
                seconds = 0
        if seconds <= 0:
            seconds = parse_duration(text)
        if seconds <= 0:
            return SkillResult("How long should the timer be?", success=False)

        label = str(args.get("label") or "").strip() or self._extract_label(text)
        # Labelled reminders persist across restarts; bare timers are ephemeral.
        reminder_id: int | None = None
        if label and self._store is not None:
            due = (datetime.now(timezone.utc).astimezone() + timedelta(seconds=seconds))
            reminder_id = self._store.add(due.isoformat(timespec="seconds"), label).id

        timer_id = self._arm(seconds, label, reminder_id)
        pretty = self._pretty(seconds)
        suffix = f" to {label}" if label else ""
        return SkillResult(f"Timer set for {pretty}{suffix}.", data={"id": timer_id, "seconds": seconds})

    @staticmethod
    def _pretty(seconds: int) -> str:
        parts = []
        for label, size in (("hour", 3600), ("minute", 60), ("second", 1)):
            n, seconds = divmod(seconds, size)
            if n:
                parts.append(f"{n} {label}{'s' if n != 1 else ''}")
        return " and ".join(parts) or "0 seconds"

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["set", "list", "cancel"]},
                        "seconds": {"type": "integer", "description": "duration for set"},
                        "label": {"type": "string", "description": "what to remind about"},
                    },
                    "required": ["action"],
                },
            },
        }
