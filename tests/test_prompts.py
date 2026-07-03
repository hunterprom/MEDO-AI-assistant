"""System prompt: bilingual rule, brevity rules, facts injection."""

from __future__ import annotations

from core.config import PersonalityConfig
from llm.prompts import system_prompt


def test_bilingual_rule_present():
    p = system_prompt(PersonalityConfig())
    assert "Macedonian" in p and "English" in p
    assert "spoken aloud" in p


def test_address_and_name_are_configurable():
    p = system_prompt(PersonalityConfig(name="HAL", address_user_as="captain"))
    assert "HAL" in p and "'captain'" in p


def test_facts_block_numbered_and_optional():
    bare = system_prompt(PersonalityConfig())
    assert "Remembered facts" not in bare

    p = system_prompt(PersonalityConfig(), facts=["likes coffee", "sister is Ana"])
    assert "Remembered facts" in p
    assert "1. likes coffee" in p and "2. sister is Ana" in p
