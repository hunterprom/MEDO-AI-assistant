"""QuitSkill: 'close yourself' actually ends the session (was an LLM hallucination).

"Close yourself" / "shut yourself down" used to miss every fast pattern and land
on the LLM, which cheerfully NARRATED a shutdown ("Shutting down — good day,
sir.") while nothing closed. QuitSkill makes the words true: it fast-matches only
SELF-directed close phrases, confirms first (like every non-lock power action),
and on "yes" hands the voice loop data['exit'] to quit for real. It must never
steal "close notepad" (a window) or a bare "shut down" (the machine).
"""

from __future__ import annotations

import asyncio

import pytest

from core.config import load_settings
from skills.base import SkillRegistry, SkillRequest
from skills.system import PowerSkill, QuitSkill


def _matches(skill, text: str) -> bool:
    return skill.match(text) is not None


@pytest.mark.parametrize("text", [
    "close yourself", "close your self", "quit medo", "exit medo",
    "shut yourself down", "shut yourself off", "turn yourself off",
    "power yourself down", "shut down medo", "turn off medo",
    "close the assistant", "goodbye medo", "goodbye, medo",
])
def test_quit_matches_self_directed_close(text):
    assert _matches(QuitSkill(), text)


@pytest.mark.parametrize("text", [
    "close notepad", "close the window", "close the tab", "close chrome",
    "shut down", "shut down the computer", "shut down the pc",
    "restart", "put the computer to sleep", "lock the screen",
    "what's the weather", "close the door",
])
def test_quit_ignores_other_closes_and_pc_power(text):
    assert not _matches(QuitSkill(), text)


def test_quit_asks_before_closing():
    skill = QuitSkill()
    r = asyncio.run(skill.execute(SkillRequest(
        text="close yourself", match=skill.match("close yourself"))))
    assert r.success and r.needs_confirmation is True
    assert not r.data.get("exit")            # nothing closes until confirmed


def test_quit_signals_exit_only_once_confirmed():
    skill = QuitSkill()
    r = asyncio.run(skill.execute(SkillRequest(
        text="close yourself", context={"confirmed": True})))
    assert r.data.get("exit") is True
    assert "shutting down" in r.speech.lower()


def test_quit_stays_off_the_semantic_and_learn_tiers():
    # requires_confirmation keeps this session-ender reachable ONLY by an
    # explicit phrase — never a fuzzy meaning match or a learned exemplar — while
    # controls_pc stays False so the PC-control switch can't hide it.
    from core.router import Router

    assert QuitSkill.requires_confirmation is True
    assert QuitSkill.controls_pc is False
    assert Router._is_learnable(QuitSkill()) is False


def test_registry_order_quit_beats_power_for_self_directed_shutdown():
    # main.py registers QuitSkill BEFORE PowerSkill, so "shut down medo" (the
    # assistant) resolves to quit while a bare "shut down" (the machine) still
    # belongs to power — first-match-wins on the fast path.
    reg = SkillRegistry()
    reg.register(QuitSkill())
    reg.register(PowerSkill())
    assert reg.find_match("shut down medo")[0].name == "quit"
    assert reg.find_match("close yourself")[0].name == "quit"
    assert reg.find_match("shut down the computer")[0].name == "power"
    assert reg.find_match("shut down")[0].name == "power"


def test_quit_routes_only_by_explicit_phrase_through_the_real_registry():
    from core.docindex import DocumentIndex
    from main import Announcer, build_registry

    settings = load_settings()
    reg = build_registry(settings, Announcer(),
                         doc_index=DocumentIndex(":memory:", None, []))
    assert reg.find_match("close yourself")[0].name == "quit"
    # "close notepad" is a window action, never a self-close.
    hit = reg.find_match("close notepad")
    assert hit is None or hit[0].name != "quit"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
