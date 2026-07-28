"""Runtime voice-session modes shared across the loop, skills, and API.

A tiny mutable holder (like ``wake_event``) so a mode can be flipped from
anywhere — a voice command, a typed command, or the HUD — and the voice loop
reads it live each turn:

* ``continuous`` — after a reply MEDO keeps listening for a follow-up for a few
  seconds without the wake word (natural back-and-forth).
* ``interpreter`` — live translation: MEDO speaks each utterance back in the
  *other* language instead of answering it.

One instance is created in ``main.py`` and handed to the :class:`VoiceLoop`,
the companion API, and the modes skill, so all three see the same switches.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SessionModes:
    continuous: bool = False
    interpreter: bool = False
    # Interpreter language pair (source auto-detected; MEDO speaks the OTHER).
    interpreter_langs: tuple[str, str] = ("en", "mk")
    # ``dictating`` — everything heard is written to ``dictation_path`` instead
    # of being answered, until "stop dictation". Like interpreter mode, the
    # wake word is not needed between lines.
    dictating: bool = False
    dictation_path: str = ""
