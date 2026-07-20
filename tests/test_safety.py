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


# --- bilingual (Macedonian) confirmation ------------------------------------


@pytest.mark.parametrize(
    "word",
    [
        "да", "може", "секако", "ајде", "важи", "потврди", "потврдувам",
        "Да.",                      # Whisper punctuation-wraps single words
        "Да, те молам.",            # punctuation stripped -> "да те молам"
        "da", "moze", "ajde", "vazi",  # Latin transliterations Whisper emits
    ],
)
def test_macedonian_affirmations(word):
    assert is_affirmative(word)
    assert not is_negative(word)


@pytest.mark.parametrize(
    "word",
    [
        "не", "откажи", "стоп", "прекини", "немој", "заборави", "не сакам",
        "Не!", "  не  сакам  ",     # punctuation + messy whitespace
        "ne", "otkazi",
    ],
)
def test_macedonian_negations(word):
    assert is_negative(word)
    assert not is_affirmative(word)


@pytest.mark.parametrize("phrase", ["не знам", "ne znam", "maybe", "што?"])
def test_ambiguous_replies_are_neither(phrase):
    # Whole-reply matching: "не знам" ("I don't know") contains "не" but must
    # NOT cancel — unknown answers go back to the router, which won't guess.
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


@pytest.mark.asyncio
async def test_confirmation_gate_speaks_macedonian():
    # End-to-end through the router: "Да." confirms, "не" cancels.
    danger = DummyDangerSkill()
    router = make_router(danger)
    await router.route("self destruct")
    r = await router.route("Да.")
    assert danger.executed is True
    assert router.awaiting_confirmation is False

    danger2 = DummyDangerSkill()
    router2 = make_router(danger2)
    await router2.route("self destruct")
    r2 = await router2.route("не")
    assert danger2.executed is False
    assert r2.speech == CANCELLED_REPLY
