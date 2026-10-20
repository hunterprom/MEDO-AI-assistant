"""«That was a bug» — the complaint that writes itself up.

The developer feature this exists for: while using MEDO normally, the moment
something goes wrong you say so, and a markdown file appears with the whole
turn in it — what you asked, what MEDO answered, which routing tier fired,
which skill it picked, how confident the match was and how long it took. You
hand that file to whoever is fixing MEDO. Nothing is typed and nothing is
reconstructed from memory an hour later.

    You:  search up the Mazda RX-7
    MEDO: The Mazda RX-7 is a Japanese sports car built around...
    You:  that was a bug, it was supposed to open the browser
    MEDO: Saved that as a bug report. That's 3 in the bugs folder.

Design notes worth keeping:

* **It reports the PREVIOUS turn, never itself.** "That was a bug" is always
  about what just happened. The router hands every skill the prior question and
  reply in ``context``, and the metrics history holds the same turn with its
  routing detail — verified: while a skill is executing, the newest recorded
  route event is the turn before it. Saying it twice in a row must also not
  file the first report as a bug, so this skill's own turns are skipped.
* **The trigger word is mandatory.** The tempting patterns are "that was
  wrong", "that's broken", "no, not that". All three are already meaningful
  elsewhere — ``core/corrections.py`` reads them as routing corrections, which
  is how the fast-path misfire statistic is produced — and stealing them here
  would corrupt that measurement to file a duplicate of what it already
  records. English needs the word *bug*; Macedonian accepts *баг* or *грешка*,
  but only anchored to тоа/ова + беше/е, or to an explicit запиши/пријави.
* **No ``routing_phrases``, so the semantic tier can't reach it.** The tier
  only indexes skills that opt in with curated phrases, and this one must not:
  a complaint-shaped question ("what's wrong with the weather skill?") deserves
  an answer, not a silently written file.
* **No capabilities, so PC control being OFF doesn't disable it.** The file
  goes to one fixed directory MEDO owns, named in config, never anywhere the
  utterance points — the same shape as taking a note. Declaring WRITE_FILES
  would put bug reporting behind the PC-control master switch, and "PC control
  is off and something is misbehaving" is exactly when you want to report it.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core import mk
from core.bugreports import BugReportStore, Turn
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

#: What may sit between the trigger and the user's note, and the note itself.
#: Optional as a whole, so a bare "that was a bug." matches with no note and a
#: trailing full stop is not mistaken for one.
_NOTE = r"(?:[\s,:;—–-]+(?P<{}>\S.*?))?\s*[.!?]*$"

DISABLED = "Bug reports are switched off in config."
DISABLED_MK = "Извештаите за багови се исклучени во конфигурацијата."
NOTHING = ("I don't have a previous turn recorded, so there's nothing to "
           "report yet.")
NOTHING_MK = "Немам запишана претходна размена, па нема што да пријавам."
FAILED = ("I couldn't write the bug report — check that the bugs folder is "
          "writable.")
FAILED_MK = ("Не можев да го запишам извештајот — провери дали папката bugs "
             "е достапна за запишување.")
REFUSED = "I only file bug reports you ask for yourself."
REFUSED_MK = "Пријавувам багови само кога ти самиот ќе побараш."


class BugReportSkill(Skill):
    name = "bug_report"
    description = ("Save the turn that just went wrong as a markdown bug "
                   "report file, with the routing detail attached.")
    # See the module docstring: deliberately not actuation, deliberately not
    # semantic-tier eligible (no routing_phrases).
    controls_pc = False

    patterns = [
        # "that was a bug" / "this is a bug" / "it was a bug"
        re.compile(r"\b(?:that|this|it)\s+(?:was|is)\s+(?:a\s+)?bug\b"
                   + _NOTE.format("note"), re.IGNORECASE),
        # "that's a bug" / "this's a bug" (Whisper does write the latter)
        re.compile(r"\b(?:that|this|it)['’]?s\s+(?:a\s+)?bug\b"
                   + _NOTE.format("note2"), re.IGNORECASE),
        # "report that as a bug" / "log a bug" / "save this as a bug"
        re.compile(r"\b(?:report|log|save|file|record)\s+"
                   r"(?:(?:that|this|it)\s+)?(?:as\s+)?(?:a\s+)?bug\b"
                   + _NOTE.format("note3"), re.IGNORECASE),
        # "write that down as a bug"
        re.compile(r"\bwrite\s+(?:that|this|it)\s+down\s+as\s+(?:a\s+)?bug\b"
                   + _NOTE.format("note4"), re.IGNORECASE),
        # "bug report: the browser never opened"
        re.compile(r"\bbug\s+report\b" + _NOTE.format("note5"), re.IGNORECASE),
        # MK: "тоа беше баг" / "ова е грешка". Anchored on тоа/ова + беше/е so
        # a bare "грешка" (an extremely common word) can never trigger it.
        re.compile(r"\b(?:тоа|ова)\s+(?:беше|е)\s+(?:баг|грешка)\b"
                   + _NOTE.format("note6"), re.IGNORECASE),
        # MK: "запиши го ова како баг" / "пријави баг" / "зачувај как баг"
        re.compile(r"\b(?:запиши|пријави|зачувај)\s+(?:го\s+)?"
                   r"(?:(?:ова|тоа)\s+)?(?:како\s+)?(?:баг|грешка)\b"
                   + _NOTE.format("note7"), re.IGNORECASE),
    ]

    def __init__(self, store: BugReportStore, config=None) -> None:
        self._store = store
        self._config = config

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _note_of(request: SkillRequest) -> str:
        """The user's own words after the trigger, from whichever group hit.

        The LLM tool path passes ``note`` as an argument instead, and wins when
        both exist: the model has usually already summarised the complaint.
        """
        arg = str((request.args or {}).get("note") or "").strip()
        if arg:
            return arg
        groups = request.match.groupdict() if request.match else {}
        for value in groups.values():
            if value and value.strip():
                return value.strip(" ,:;-—–")
        return ""

    @staticmethod
    def _fallback_turn(request: SkillRequest) -> Turn | None:
        """The previous turn as the ROUTER knows it.

        Used when routing telemetry is off or its database is unreachable. It
        has no tier, skill or latency — but a report saying what was asked and
        what came back is still worth having, and switching telemetry off
        should not silently switch bug reporting off with it.
        """
        ctx = request.context or {}
        question = str(ctx.get("last_question") or "").strip()
        reply = str(ctx.get("last_reply") or "").strip()
        if not (question or reply):
            return None
        return Turn(text=question, reply=reply)

    def _saved_speech(self, count: int, *, speak_mk: bool) -> str:
        """What MEDO says back.

        The filename is deliberately NOT spoken. It is a 60-character
        hyphenated stamp; read aloud it is a wall of noise, and the two things
        a person actually needs — that it saved, and where to look — fit in a
        sentence. The full path rides in ``data`` for the HUD, and
        ``python -m core.bugreports`` lists them properly.
        """
        if speak_mk:
            if count <= 1:
                return "Го зачував како извештај за баг, првиот во папката bugs."
            return (f"Го зачував како извештај за баг. "
                    f"Сега има {count} во папката bugs.")
        if count <= 1:
            return "Saved that as a bug report — the first one in the bugs folder."
        return f"Saved that as a bug report. That's {count} in the bugs folder."

    # -- the skill -------------------------------------------------------------

    async def execute(self, request: SkillRequest) -> SkillResult:
        speak_mk = mk.is_cyrillic(request.utterance)
        ctx = request.context or {}

        # Trust boundary: a fetched page or an imported document can contain
        # the words "that was a bug", and content must never drive an action.
        # The cost of being wrong here is only a stray file, but the rule is
        # the rule — content is never promoted to an instruction.
        if ctx.get("untrusted") or ctx.get("provenance") == "untrusted":
            return SkillResult(REFUSED_MK if speak_mk else REFUSED,
                               success=False, data={"refused": "untrusted"})

        if self._config is not None and not getattr(self._config, "enabled", True):
            return SkillResult(DISABLED_MK if speak_mk else DISABLED,
                               success=False)

        note = self._note_of(request)
        path = self._store.capture(
            note=note,
            reported_with=request.text,
            language="mk" if speak_mk else "en",
            fallback=self._fallback_turn(request),
            # Without this, a second "that was a bug" files the FIRST bug
            # report as the bug being reported.
            exclude_skill=self.name,
        )
        if path is None:
            # Two different failures land here: nothing recorded to report, and
            # a directory that could not be written. Telling them apart matters
            # — one is "say it after MEDO does something", the other is
            # "your disk or permissions are wrong".
            if self._fallback_turn(request) is None and not self._store.recent_turns(
                    1, exclude_skill=self.name):
                return SkillResult(NOTHING_MK if speak_mk else NOTHING,
                                   success=False)
            return SkillResult(FAILED_MK if speak_mk else FAILED, success=False)

        logger.info("bug report written: %s", path)
        return SkillResult(
            self._saved_speech(len(self._store.list()), speak_mk=speak_mk),
            data={"path": str(path), "file": path.name, "note": note})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Save the PREVIOUS turn as a bug report file, because the "
                    "user said it was wrong or broken. Call this only when the "
                    "user is reporting a defect in MEDO itself."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": ("What the user says went wrong, in "
                                            "their own words. Optional."),
                        },
                    },
                    "required": [],
                },
            },
        }
