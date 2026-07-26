"""Tier-2 semantic-tier EXTENSION (2026-07): more query skills reachable by
meaning, and the safety gate that makes flipping the HUD SEMANTIC TIER to LIVE
safe.

The latent bug this closes: the semantic path dispatches a skill with a bare
SkillRequest (no regex match, no args). An arg-hungry skill reached that way
deflected ("What should I search for?"). ``Router._semantic_safe`` now keeps
such skills out of the index unless they promise (``semantic_from_text``) to
derive their argument from the utterance — which web_search now does.
"""

from __future__ import annotations

import pytest

from core.config import load_settings
from core.router import Router
from skills.base import Skill, SkillRequest, SkillResult


def _settings():
    return load_settings()


def _facts(tmp_path):
    from core.facts import FactsStore
    return FactsStore(str(tmp_path / "facts.db"))


# --- the safety gate ---------------------------------------------------------

def test_semantic_safe_allows_argfree_query_skills(tmp_path):
    from skills.memory_skill import RecallFactsSkill
    from skills.security import SecurityCheckSkill, FirewallAuditSkill, UpdateCheckSkill
    from skills.vision_skill import SeeCameraSkill, SeeScreenSkill, PointAtSkill
    s = _settings()
    for skill in (RecallFactsSkill(_facts(tmp_path)),
                  SecurityCheckSkill(s), FirewallAuditSkill(s), UpdateCheckSkill(s),
                  SeeCameraSkill(s), SeeScreenSkill(s), PointAtSkill(s)):
        assert Router._semantic_safe(skill) is True, skill.name


def test_semantic_safe_allows_semantic_from_text_skills():
    from skills.websearch import WebSearchSkill
    w = WebSearchSkill()
    assert w.semantic_from_text is True
    assert Router._semantic_safe(w) is True


def test_semantic_safe_rejects_arg_hungry_skills(tmp_path):
    """A skill with a REQUIRED tool param and no semantic_from_text can't run
    on the bare semantic request, so it must be kept out of the index."""
    from skills.memory_skill import ForgetFactSkill, RememberFactSkill
    for skill in (ForgetFactSkill(_facts(tmp_path)), RememberFactSkill(_facts(tmp_path))):
        assert Router._semantic_safe(skill) is False, skill.name


def test_semantic_safe_rejects_actuation_and_confirmation():
    from skills.web_open import OpenWebsiteSkill
    assert Router._semantic_safe(OpenWebsiteSkill()) is False   # controls_pc

    class _Confirmer(Skill):
        name = "confirmer"
        requires_confirmation = True
        routing_phrases = ["do the risky thing"]

        async def execute(self, request):  # pragma: no cover - never called
            return SkillResult("ok")

    assert Router._semantic_safe(_Confirmer()) is False


# --- web_search self-serves from the utterance on the semantic path ----------

@pytest.mark.asyncio
async def test_web_search_derives_query_from_text():
    from skills.websearch import WebSearchSkill
    w = WebSearchSkill(summarize=None)          # None -> returns the raw block
    w._search = lambda q: [{"title": "GT", "body": "Masahiro Andoh"}]
    # A bare semantic request: no match, no args.
    res = await w.execute(SkillRequest(text="who composed the music for gran turismo"))
    assert res.success
    assert res.data["query"] == "who composed the music for gran turismo"


@pytest.mark.asyncio
async def test_web_search_still_deflects_on_a_truly_empty_call():
    from skills.websearch import WebSearchSkill
    w = WebSearchSkill(summarize=None)
    res = await w.execute(SkillRequest(text="   "))
    assert res.success is False
    assert "search for" in res.speech.lower()


# --- every skill that opted into the tier is actually reachable & safe --------

def test_new_semantic_skills_are_indexable(tmp_path):
    """The skills we gave routing_phrases this round must ALL pass the gate,
    or they'd silently never route by meaning."""
    from skills.memory_skill import RecallFactsSkill
    from skills.security import SecurityCheckSkill, FirewallAuditSkill, UpdateCheckSkill
    from skills.vision_skill import SeeCameraSkill, SeeScreenSkill, PointAtSkill
    from skills.websearch import WebSearchSkill
    s = _settings()
    skills = [RecallFactsSkill(_facts(tmp_path)), WebSearchSkill(),
              SecurityCheckSkill(s), FirewallAuditSkill(s), UpdateCheckSkill(s),
              SeeCameraSkill(s), SeeScreenSkill(s), PointAtSkill(s)]
    for skill in skills:
        assert skill.routing_phrases, f"{skill.name} lost its routing_phrases"
        assert Router._semantic_safe(skill) is True, skill.name


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
