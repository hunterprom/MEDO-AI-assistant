"""Make the council "agents" actually work: reach a named specialist or the
whole panel by the phrasings people really use, on the fast path AND by meaning.

The audit's finding: the expert was addressed but never convened, because the
patterns only knew the literal verb "ask" / the word "council", and the skills
had no semantic surface at all.
"""

from __future__ import annotations

import pytest

from core.config import load_settings
from core.council import find_specialist
from core.router import Router
from skills.base import SkillRequest


def _ask():
    from skills.council import AskSpecialistSkill
    return AskSpecialistSkill(load_settings())


def _convene():
    from skills.council import ConveneCouncilSkill
    return ConveneCouncilSkill(load_settings())


def _who(skill, text):
    m = skill.match(text)
    if m is None:
        return None
    gd = m.groupdict()
    who = next((gd[k] for k in gd if k and k.startswith("who") and gd[k]), "")
    member = find_specialist(who, skill._council)
    return member.key if member else None


# --- ask_specialist: natural ways to address one expert ----------------------

def test_ask_specialist_address_phrasings():
    s = _ask()
    cases = [
        ("get the physicist's take on relativity", "physics"),
        ("what would the roboticist recommend for the gait", "robotics"),
        ("check with the lawyer about this contract", "law"),
        ("have the economist weigh in on inflation", "economics"),
        ("run this design by the electrical engineer", "electrical"),
    ]
    for text, key in cases:
        assert _who(s, text) == key, text


def test_ask_specialist_still_declines_a_non_specialist():
    s = _ask()
    # The greedy who-capture must not claim a non-expert.
    assert s.match("what does this page say about batteries") is None
    assert s.match("run this by the printer") is None


# --- convene_council: the synonyms its own description promises ---------------

def test_convene_council_synonyms():
    s = _convene()
    for text in ("convene the experts on the motor choice",
                 "assemble your specialists on this trade-off",
                 "what do your experts think about brushless motors",
                 "convene the panel of experts on the frame",
                 "gather a round-table on this design question"):
        assert s.match(text) is not None, text
    # the original "council" phrasings still work
    assert s.match("convene the council on the frame") is not None


def test_convene_does_not_grab_everyday_team_talk():
    """Bare 'team'/'panel' and generic 'the experts' are not the council."""
    s = _convene()
    for text in ("assemble the team", "gather the team for lunch",
                 "gather the panel for the interview",
                 "what do the experts think about climate change"):
        assert s.match(text) is None, text


# --- reachable by MEANING, and safe to be ------------------------------------

def test_council_skills_are_semantically_reachable():
    from skills.circuit import CircuitSkill
    s = load_settings()
    for skill in (_ask(), _convene(), CircuitSkill(s, None, None)):
        assert skill.routing_phrases, skill.name
        assert skill.semantic_from_text is True, skill.name
        assert Router._semantic_safe(skill) is True, skill.name


@pytest.mark.asyncio
async def test_ask_specialist_semantic_fallback_recovers_expert():
    async def fake_ask(system, user):
        return "here is the answer"
    from skills.council import AskSpecialistSkill
    s = AskSpecialistSkill(load_settings(), ask=fake_ask)
    # bare semantic request: no match, no args — the expert is named in the text
    res = await s.execute(SkillRequest(text="get the physicist's take on entanglement"))
    assert res.success
    assert "physicist" in res.speech.lower()
    assert res.data.get("specialist") == "physics"


def _roster():
    from skills.council import CouncilRosterSkill
    return CouncilRosterSkill(load_settings())


def test_council_roster_lists_and_probes():
    s = _roster()
    assert s.match("who's on the council") is not None
    assert s.match("which experts do you have") is not None
    assert s.match("do you have a lawyer") is not None
    # "do you have a X" is gated: a non-specialist falls through
    assert s.match("do you have a minute") is None
    assert s.match("do you have a plumber") is None
    # ...and a trailing-noun substring must NOT resolve via an alias
    assert s.match("do you have a legal pad") is None
    assert s.match("do you have an electrical outlet") is None
    assert s.match("do you have a data analyst") is None
    assert Router._semantic_safe(s) is True


@pytest.mark.asyncio
async def test_council_roster_execute():
    s = _roster()
    listed = await s.execute(SkillRequest(text="who's on the council",
                                          match=s.match("who's on the council")))
    assert listed.success
    assert "engineer" in listed.speech.lower()          # names real specialists
    probe = await s.execute(SkillRequest(text="do you have a lawyer",
                                         match=s.match("do you have a lawyer")))
    assert probe.success and "lawyer" in probe.speech.lower()


@pytest.mark.asyncio
async def test_convene_semantic_fallback_uses_text_as_question():
    async def fake_ask(system, user):
        return "the motor answer"
    from skills.council import ConveneCouncilSkill
    s = ConveneCouncilSkill(load_settings(), ask=fake_ask, synthesize=None)
    res = await s.execute(
        SkillRequest(text="get the whole team to weigh in on using a brushless motor"))
    assert res.success
    assert "put to the council" not in res.speech.lower()   # did NOT deflect


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
