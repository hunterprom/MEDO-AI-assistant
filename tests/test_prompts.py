"""System prompt: bilingual rule, brevity rules, facts injection."""

from __future__ import annotations

from core.config import PersonalityConfig
from llm.prompts import system_prompt


def test_language_rule_is_multilingual_and_names_none():
    """MEDO speaks sixteen languages now, so the prompt must not name two.

    Naming them is what produced "I understand Japanese, but I usually speak
    English or Macedonian" as a reply to こんにちは — the model volunteering a
    preference it should not have.
    """
    p = system_prompt(PersonalityConfig())
    assert "multilingual" in p and "bilingual" not in p
    assert "answer in the language the user" in p.lower()
    assert "spoken aloud" in p


def test_no_process_narration_and_app_routing_rules():
    """Guards the live misfires: narrated 'Searching…'/'I can access the web
    now…' aloud, and treated 'open a file in Arduino IDE' / 'find me tools' as a
    filename/web search instead of the app."""
    p = system_prompt(PersonalityConfig())
    assert "narrate the process" in p.lower()
    assert "open a file in <app>" in p.lower() or "open a file in" in p.lower()
    assert "not a web search" in p.lower()          # menu/tool inside an app


def test_address_and_name_are_configurable():
    p = system_prompt(PersonalityConfig(name="HAL", address_user_as="captain"))
    assert "HAL" in p and "'captain'" in p


def test_facts_block_numbered_and_optional():
    bare = system_prompt(PersonalityConfig())
    assert "Remembered facts" not in bare

    p = system_prompt(PersonalityConfig(), facts=["likes coffee", "sister is Ana"])
    assert "Remembered facts" in p
    assert "1. likes coffee" in p and "2. sister is Ana" in p
