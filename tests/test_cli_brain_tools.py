"""A CLI brain cannot call MEDO's skills — so it must not answer as if it could.

Reported from a live voice session, four bug reports in three minutes:

    You:  Can you build me a 3D model of a simple rocket?
    MEDO: I can't do that — I don't have a tool for generating 3D models...
    You:  So, can you build me a simple counter app?
    MEDO: Can't do that either — building an app is a coding task, not
          something I have a tool for here at the desk.
    You:  So, turn on pointer mode.
    MEDO: I don't have a pointer mode to switch on — no tool for that here, sir.
    You:  Pointer mode on.                     <- fast path
    MEDO: Pointer mode on — point with your index finger...   <- it works fine

Every denial was true of the ANSWERING AGENT and false of MEDO. The brain was
`claude-code`, and ``OllamaClient._chat_cli`` takes no ``tools`` argument at
all: a CLI agent is handed a flat prompt and read back as text, so MEDO's
sixty-odd skills are simply absent. The router already borrowed the local
tool-brain for live-info queries for exactly this reason; it just never did it
for a request to ACT.

The turn after that is worse than a denial: asked to turn pointer mode off,
MEDO said "Pointer mode off." having called nothing whatsoever.
"""

from __future__ import annotations

import pytest

from core.config import load_settings
from core.events import EventBus
from core.router import Router
from llm.client import CLI_PROVIDERS
from skills.base import SkillRegistry


def _router(provider: str = "claude-code") -> Router:
    settings = load_settings()
    settings.llm.provider = provider
    return Router(settings, SkillRegistry(), None, EventBus())  # type: ignore[arg-type]


# Verbatim from the bug reports, plus the everyday commands of the same shape.
ACTIONS = [
    "So, turn on pointer mode.",
    "turn off pointer mode",
    "Pointer mode on.",
    "So, can you build me a simple counter app?",
    "Can you build me a 3D model of a simple rocket?",
    "open youtube",
    "close spotify",
    "take a screenshot",
    "turn the volume up",
    "set a timer for five minutes",
    "remind me to call mum at six",
    "вклучи pointer mode",
    "отвори youtube",
]

# Knowledge questions from the SAME session. MEDO answered these well on the
# CLI brain and must keep doing so — handing them to a small local model would
# make the answers worse, which is the opposite of the fix.
QUESTIONS = [
    "So, can you recommend me any cologne perfume that I could use for this autumn?",
    "Could you tell me how can I make a motor driver?",
    "Can you tell me how can I make a simple AI model, specifically vision model?",
    "what is an H-bridge",
    "how do I build a counter app in python",
    "how to make a 3D model",
    "why is the sky blue",
    "tell me about the Mazda RX-7",
    "која е разликата помеѓу нив",
    "објасни ми што е RC коло",
]


@pytest.mark.parametrize("text", ACTIONS)
def test_a_request_to_act_leaves_the_cli_brain(text):
    assert _router()._needs_tool_brain(text) == "is a request to act"


@pytest.mark.parametrize("text", QUESTIONS)
def test_a_question_stays_on_the_selected_brain(text):
    """The verb is the same; the frame is not.

    "How can I make a motor driver" trips the same `make` as "make me an app".
    Only the explanatory frame separates them, and getting this wrong sends
    every how-to question to a 3B local model.
    """
    assert _router()._needs_tool_brain(text) == ""


def test_a_non_cli_brain_never_borrows_anything():
    """Ollama/OpenAI/Anthropic brains get MEDO's tools natively — nothing to fix."""
    router = _router("ollama")
    assert all(router._needs_tool_brain(t) == "" for t in ACTIONS + QUESTIONS)


def test_live_info_still_borrows_and_says_which_rule_fired():
    """The original reason must survive, and the log has to distinguish them."""
    router = _router()
    assert router._needs_tool_brain("what's the latest news") == "needs live info"
    assert router._needs_tool_brain("open youtube") == "is a request to act"


def test_every_cli_provider_is_covered():
    """codex has the same hole as claude-code — neither gets MEDO's tools."""
    for provider in CLI_PROVIDERS:
        assert _router(provider)._needs_tool_brain("open youtube")


def test_the_cli_chat_path_still_takes_no_tools():
    """A guard on the ASSUMPTION this whole fix rests on.

    If `_chat_cli` ever grows a `tools` parameter, a CLI agent can call MEDO's
    skills directly and the borrow above becomes unnecessary overhead rather
    than a fix. This is the tripwire that says so.
    """
    import inspect

    from llm.client import OllamaClient

    params = inspect.signature(OllamaClient._chat_cli).parameters
    assert "tools" not in params, (
        "_chat_cli now accepts tools — revisit Router._needs_tool_brain, which "
        "exists only because a CLI brain cannot call MEDO's skills")
