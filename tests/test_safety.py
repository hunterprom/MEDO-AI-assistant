"""Safety layer: path whitelist, yes/no parsing, and the confirmation gate."""

from __future__ import annotations

import re

import pytest

from core.config import load_settings
from core.events import EventBus, RoutePath
from core.router import CANCELLED_REPLY, Router
from core.safety import PathWhitelist, is_affirmative, is_negative
from llm.client import OllamaClient
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult


# --- whitelist ---------------------------------------------------------------
def test_whitelist_allows_inside_refuses_outside(tmp_path):
    inside = tmp_path / "docs"
    inside.mkdir()
    (inside / "a.txt").write_text("hi")
    wl = PathWhitelist([str(inside)])

    assert wl.is_allowed(inside / "a.txt")
    assert wl.is_allowed(inside)
    assert not wl.is_allowed("/etc/passwd")
    # Directory-traversal escape must be blocked (resolves outside the root).
    assert not wl.is_allowed(inside / ".." / "secret.txt")


def test_whitelist_skips_missing_dirs():
    wl = PathWhitelist(["/nonexistent/path/xyz"])
    assert wl.roots == []


@pytest.mark.parametrize("word", ["yes", "Yeah", "confirm", "do it", "okay."])
def test_affirmations(word):
    assert is_affirmative(word)
    assert not is_negative(word)


@pytest.mark.parametrize("word", ["no", "cancel", "nope", "never mind"])
def test_negations(word):
    assert is_negative(word)
    assert not is_affirmative(word)


# --- bilingual gate: Macedonian yes/no (Cyrillic + Latin transliterations) ----
@pytest.mark.parametrize("word", ["да", "може", "секако", "ајде", "важи",
                                  "потврди", "потврдувам", "да те молам"])
def test_macedonian_affirmations(word):
    assert is_affirmative(word)
    assert not is_negative(word)


@pytest.mark.parametrize("word", ["не", "откажи", "стоп", "прекини", "немој",
                                  "заборави", "не сакам"])
def test_macedonian_negations(word):
    assert is_negative(word)
    assert not is_affirmative(word)


@pytest.mark.parametrize("word", ["da", "moze", "ajde", "vazi"])
def test_transliterated_affirmations(word):
    assert is_affirmative(word)


@pytest.mark.parametrize("word", ["ne", "otkazi"])
def test_transliterated_negations(word):
    assert is_negative(word)


@pytest.mark.parametrize("messy,expect_yes", [
    ("Да.", True),          # Whisper adds a period + capital
    ("Да!", True),
    ("може…", True),
    ("Да, те молам.", True),  # comma dropped by normalization
    ("НЕ", False),
    ("Не!", False),
])
def test_whisper_punctuation_and_case(messy, expect_yes):
    assert is_affirmative(messy) is expect_yes
    if not expect_yes:
        assert is_negative(messy)


def test_ambiguous_reply_is_neither_yes_nor_no():
    # "не знам" = "I don't know" — contains "не" but must NOT count as a no
    # (whole-reply matching, never substring). The router then asks again.
    for phrase in ("не знам", "не баш", "yes and no", "не сум сигурен"):
        assert not is_affirmative(phrase)
        assert not is_negative(phrase)


# --- confirmation gate (with a safe dummy skill, never a real action) --------
class DummyDangerSkill(Skill):
    name = "danger"
    description = "test-only destructive skill"
    patterns = [re.compile(r"\bself\s+destruct\b", re.IGNORECASE)]

    def __init__(self) -> None:
        self.executed = False

    async def execute(self, request: SkillRequest) -> SkillResult:
        if not request.context.get("confirmed"):
            return SkillResult("Are you sure you want to self destruct?", needs_confirmation=True)
        self.executed = True
        return SkillResult("Done.")

    def tool_schema(self):
        return {}


def make_router(skill: Skill) -> Router:
    settings = load_settings()
    registry = SkillRegistry()
    registry.register(skill)
    return Router(settings, registry, OllamaClient(settings.llm), EventBus())


@pytest.mark.asyncio
async def test_destructive_requires_confirmation_then_runs():
    danger = DummyDangerSkill()
    router = make_router(danger)

    r1 = await router.route("self destruct")
    assert danger.executed is False           # not yet!
    assert router.awaiting_confirmation is True
    assert "sure" in r1.speech.lower()

    r2 = await router.route("yes")
    assert danger.executed is True            # confirmed -> executed
    assert router.awaiting_confirmation is False
    assert r2.path is RoutePath.FAST


@pytest.mark.asyncio
async def test_saying_no_cancels_without_running():
    danger = DummyDangerSkill()
    router = make_router(danger)

    await router.route("self destruct")
    r = await router.route("no")
    assert danger.executed is False
    assert r.speech == CANCELLED_REPLY
    assert router.awaiting_confirmation is False
