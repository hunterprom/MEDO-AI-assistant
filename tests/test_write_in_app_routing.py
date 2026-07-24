"""Routing for "open an app AND write in it".

From a live transcript: "Could you open Notepad and summarize, type it out, the
Odysseus in Notepad?" answered "Opening notepad." in 9 ms and wrote nothing —
AppsSkill matched the leading "open notepad" and the writing half vanished.

Two rules fall out of that:
  * a launch request that ALSO asks for writing is not a plain launch, so
    AppsSkill must decline it;
  * text that has to be COMPOSED first ("summarize …") can't be served by a
    literal regex capture — it would type "it out, the Odysseus" — so
    WriteInAppSkill declines too and the brain writes it, then calls
    write_in_app(app=…, text=…) as a tool.
"""

from __future__ import annotations

import pytest

from skills.apps import AppsSkill
from skills.file_edit import WriteInAppSkill

_APPS = {
    "notepad": {"windows": "start notepad"},
    "editor": {"windows": "code"},
    "chrome": {"windows": "start chrome"},
}


def _route(text: str) -> str:
    """Registration order is WriteInApp -> Apps (see main.py)."""
    if WriteInAppSkill(_APPS).match(text):
        return "write_in_app"
    if AppsSkill(_APPS).match(text):
        return "apps"
    return "llm"


# --- plain launches still fast-path ------------------------------------------

@pytest.mark.parametrize("phrase", [
    "open notepad", "close chrome", "launch the editor", "start chrome",
])
def test_plain_launch_still_goes_to_apps(phrase):
    assert _route(phrase) == "apps"


# --- literal text goes straight to write_in_app ------------------------------

@pytest.mark.parametrize("phrase", [
    "type hello world in notepad",
    "write the shopping list in notepad",
    "put the address in notepad",
    "напиши го ова во нотепад",
])
def test_literal_text_writes_into_the_app(phrase):
    assert _route(phrase) == "write_in_app"


# --- composed text must reach the brain, not a regex -------------------------

@pytest.mark.parametrize("phrase", [
    "Could you open Notepad and summarize, type it out, the Odysseus in Notepad?",
    "open notepad and summarize the odyssey",
    "write a poem about the sea in notepad",
    "explain recursion and put it in notepad",
    "translate this and write it in notepad",
])
def test_composed_text_falls_through_to_the_llm(phrase):
    # Neither fast-path skill may claim it: AppsSkill would answer "Opening
    # notepad." and drop the writing; WriteInApp would type the raw capture.
    assert _route(phrase) == "llm", phrase


def test_the_transcript_no_longer_stops_at_opening():
    phrase = "Could you open Notepad and summarize, type it out, the Odysseus in Notepad?"
    assert AppsSkill(_APPS).match(phrase) is None, "apps must not claim it"
    assert WriteInAppSkill(_APPS).match(phrase) is None, "no literal capture"


def test_write_in_app_tool_takes_app_and_text():
    schema = WriteInAppSkill(_APPS).tool_schema()["function"]["parameters"]
    assert set(schema["required"]) == {"app", "text"}
    assert "notepad" in schema["properties"]["app"]["enum"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
