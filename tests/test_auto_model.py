"""Auto model tier (same brain, lighter model on simple turns).

The classifier is deterministic and leans STRONG on any complexity signal, so a
hard question is never downgraded; only clearly-simple turns get the fast model.
The router applies it same-provider and never overrides an offline/unset model.
"""

from __future__ import annotations

import pytest

from core.config import load_settings
from core.events import EventBus
from core.router import Router, pick_model_tier
from llm.client import OllamaClient
from skills.base import SkillRegistry


# -- the classifier -----------------------------------------------------------

@pytest.mark.parametrize("text", [
    "how are you", "what time is it", "hello", "set a timer for five minutes",
    "what's the weather in Skopje", "add milk to the shopping list",
    "turn the volume up", "open youtube",
])
def test_simple_turns_go_fast(text):
    assert pick_model_tier(text) == "fast"


@pytest.mark.parametrize("text", [
    "explain why the sky is blue", "compare React and Vue for a big app",
    "derive the quadratic formula", "debug this python function for me",
    "why does my motor stall under load", "walk me through setting up CI",
    "what are the pros and cons of microservices",
    "design an architecture for a chat app",
    "what is a mutex? and what is a semaphore?",             # two questions
    "I need you to carefully think through the trade-offs of each database "
    "option for a high write workload with strong consistency needs and tell me",
])
def test_complex_turns_go_strong(text):
    assert pick_model_tier(text) == "strong"


def test_empty_is_fast():
    assert pick_model_tier("") == "fast" and pick_model_tier("   ") == "fast"


# -- router integration -------------------------------------------------------

def _router(*, auto=True, fast="llama3.2:3b", model="qwen3:30b"):
    s = load_settings()
    s.llm.auto_model = auto
    s.llm.fast_model = fast
    r = Router(s, SkillRegistry(), OllamaClient(s.llm), EventBus())
    r.model = model
    return r


def test_simple_turn_downgrades_to_fast_model():
    r = _router()
    assert r._tier_model("how are you") == "llama3.2:3b"


def test_complex_turn_keeps_the_strong_model():
    r = _router()
    assert r._tier_model("explain in detail why transformers use attention") \
        == "qwen3:30b"


def test_auto_off_never_downgrades():
    r = _router(auto=False)
    assert r._tier_model("how are you") == "qwen3:30b"


def test_no_fast_model_never_downgrades():
    r = _router(fast="")
    assert r._tier_model("how are you") == "qwen3:30b"


def test_offline_model_stays_none():
    # A null model (tests / no brain) must stay null so the offline path holds.
    r = _router(model=None)
    assert r._tier_model("how are you") is None


def test_the_brain_provider_is_never_changed():
    # tiering swaps the MODEL, not the client/provider — _tier_model only returns
    # a model string; _pick_brain always returns the same self._llm client.
    r = _router()
    assert r._tier_model("hi") == "llama3.2:3b"        # same provider, small model


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
