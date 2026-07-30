"""Second opinion: the council audits MEDO's OWN last answer.

A fake `ask` returns canned tagged audit lines so the test proves the parsing,
the DETERMINISTIC fail-safe verdict (WRONG dominates, garbled/silent degrades to
unverified — never to false confidence), the clean declines, and that the
"full breakdown" follow-up reads the cache with no second model call.
"""

from __future__ import annotations

import pytest

from core.config import load_settings
from skills.base import SkillRequest
from skills.second_opinion import SecondOpinionSkill, _parse_audit, _verdict


def _skill(script, max_members=1):
    calls: list[tuple[str, str]] = []

    async def ask(system, user):
        calls.append((system, user))
        return script

    s = load_settings()
    s.council.max_members = max_members
    skill = SecondOpinionSkill(s, ask)
    skill._calls = calls                       # type: ignore[attr-defined]
    return skill


def _ctx(answer, question="what torque does the servo need"):
    return {"last_reply": answer, "last_question": question}


async def _run(skill, text="second opinion", ctx=None):
    return await skill.execute(SkillRequest(text=text, context=ctx or {}))


# --- the deterministic verdict ------------------------------------------------

def test_verdict_is_fail_safe():
    assert _verdict([]) == "unverified"                              # nothing parsed
    assert _verdict([{"tag": "SUPPORTED"}]) == "solid"
    assert _verdict([{"tag": "SUPPORTED"}, {"tag": "UNSUPPORTED"}]) == "shaky"
    assert _verdict([{"tag": "SUPPORTED"}, {"tag": "WRONG"}]) == "wrong"   # WRONG wins
    assert _verdict([{"tag": "CANNOT_VERIFY"}]) == "unverified"      # doubt != confidence


def test_parse_audit_reads_tagged_lines_and_ignores_prose():
    from core.council import COUNCIL
    reply = ("Here is my audit:\n"
             "CLAIM: the 2.5 Nm figure | UNSUPPORTED | no load estimate\n"
             "CLAIM: the 2 Hz gait | SUPPORTED | reasonable\n"
             "that's all")
    claims = _parse_audit(reply, COUNCIL[0])
    assert [c["tag"] for c in claims] == ["UNSUPPORTED", "SUPPORTED"]
    assert claims[0]["claim"] == "the 2.5 Nm figure"
    assert claims[0]["reason"] == "no load estimate"


# --- end to end ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_unsupported_claim_is_flagged_and_offers_breakdown():
    script = ("CLAIM: the 2.5 newton-metre figure | UNSUPPORTED | no load estimate given\n"
              "CLAIM: a 2 hertz gait | SUPPORTED | reasonable for the size")
    skill = _skill(script)
    r = await _run(skill, ctx=_ctx("Around 2.5 newton-metres per hip at a 2 hertz gait."))
    assert r.success and r.data["second_opinion"] == "shaky"
    assert r.await_reply is True
    # The "want the breakdown?" tail is a yes/no OFFER: a reply that is neither is
    # a fresh command the router re-routes, not a bland "okay" that drops it.
    assert r.reply_is_offer is True
    assert "unsupported" in r.speech.lower()
    assert "2.5 newton-metre" in r.speech


def test_tag_word_inside_the_claim_text_cannot_flip_the_verdict():
    from core.council import COUNCIL
    # A claim summary that echoes the answer's wording ("...supported by...") must
    # NOT be read as the tag — the real WRONG in the delimited slot has to win, or
    # the whole fail-safe promise breaks (a WRONG claim reported as "solid").
    claims = _parse_audit(
        "CLAIM: bracket supported by FEA | WRONG | no FEA was actually run", COUNCIL[0])
    assert claims[0]["tag"] == "WRONG"
    assert _verdict(claims) == "wrong"
    # And with no delimiters, the MOST SEVERE tag on the line wins (over-flag).
    c2 = _parse_audit("the 2.5 figure is UNSUPPORTED though partly SUPPORTED", COUNCIL[0])
    assert c2[0]["tag"] == "UNSUPPORTED"
    # A tag with an odd whitespace separator (tab / non-breaking space) must
    # normalize, not crash the parser — the fail-safe contract forbids throwing.
    c3 = _parse_audit("the melting point CANNOT\tVERIFY here", COUNCIL[0])
    assert c3[0]["tag"] == "CANNOT_VERIFY"
    c4 = _parse_audit("CLAIM: x | CANNOT\xa0VERIFY | outside my field", COUNCIL[0])
    assert c4[0]["tag"] == "CANNOT_VERIFY"


@pytest.mark.asyncio
async def test_all_cannot_verify_does_not_arm_reply_capture():
    # An all-CANNOT_VERIFY audit is "unverified" with claims, but its headline
    # offers no breakdown — so it must NOT set await_reply, or it would swallow
    # the user's next free-form utterance.
    skill = _skill("CLAIM: melting point of alloy X | CANNOT_VERIFY | outside my field")
    r = await _run(skill, ctx=_ctx("The alloy melts at 1400 C."))
    assert r.data["second_opinion"] == "unverified"
    assert r.await_reply is False
    assert r.reply_is_offer is False           # no offer armed -> nothing to swallow
    assert "breakdown" not in r.speech.lower()


@pytest.mark.asyncio
async def test_a_wrong_claim_dominates_the_verdict():
    skill = _skill("CLAIM: x | SUPPORTED | fine\nCLAIM: y | WRONG | actually false")
    r = await _run(skill, ctx=_ctx("some answer"))
    assert r.data["second_opinion"] == "wrong" and "wrong" in r.speech.lower()


@pytest.mark.asyncio
async def test_all_supported_reads_as_solid():
    skill = _skill("CLAIM: x | SUPPORTED | checks out\nCLAIM: y | SUPPORTED | correct")
    r = await _run(skill, ctx=_ctx("some answer"))
    assert r.data["second_opinion"] == "solid" and "rely on it" in r.speech.lower()


@pytest.mark.asyncio
async def test_specialist_is_chosen_by_the_answers_field_not_the_question():
    # The complaint this fixes: "are you sure?" audited an electrical wiring answer
    # with the SOFTWARE engineer because the generic question named no field. Now
    # the ANSWER drives the choice — an electrical answer gets the electrical
    # engineer.
    skill = _skill("CLAIM: the low-voltage run | SUPPORTED | fine", max_members=1)
    r = await _run(skill, ctx=_ctx(
        "Run low-voltage wiring on a separate path from the mains, and keep a "
        "solid ground at every box.",
        question="is this a good idea"))
    assert r.data["members"] == ["electrical"]


@pytest.mark.asyncio
async def test_falls_back_to_the_question_field_when_the_answer_is_generic():
    # If the answer names no field but the question does, use the question's.
    skill = _skill("CLAIM: it | SUPPORTED | fine", max_members=1)
    r = await _run(skill, ctx=_ctx("Yes, that should be fine.",
                                   question="what resistor do I need for the led"))
    assert r.data["members"] == ["electrical"]


@pytest.mark.asyncio
async def test_garbled_audit_degrades_to_unverified_not_confident():
    # No parseable tags -> fail-safe: unverified, never 'solid', and nothing to
    # break down.
    skill = _skill("I think the answer looks mostly fine to me honestly.")
    r = await _run(skill, ctx=_ctx("some answer"))
    assert r.data["second_opinion"] == "unverified"
    assert "unconfirmed" in r.speech.lower() and r.await_reply is False


@pytest.mark.asyncio
async def test_declines_with_no_prior_answer():
    skill = _skill("CLAIM: x | SUPPORTED | ok")
    r = await skill.execute(SkillRequest(text="are you sure", context={}))
    assert r.success is False and "haven't answered" in r.speech.lower()
    assert not skill._calls                    # never bothered the council


@pytest.mark.asyncio
async def test_declines_with_no_brain():
    skill = SecondOpinionSkill(load_settings(), None)
    r = await skill.execute(SkillRequest(text="second opinion", context=_ctx("a")))
    assert r.success is False and "offline" in r.speech.lower()


@pytest.mark.asyncio
async def test_declines_when_the_panel_is_silent():
    skill = _skill("")                          # every auditor returns nothing
    r = await _run(skill, ctx=_ctx("some answer"))
    assert r.success is False


@pytest.mark.asyncio
async def test_auditors_are_told_not_to_reanswer():
    skill = _skill("CLAIM: x | SUPPORTED | ok")
    await _run(skill, ctx=_ctx("some answer"))
    system, user = skill._calls[0]
    assert "do not re-answer" in system.lower()
    assert "audit" in user.lower()


@pytest.mark.asyncio
async def test_breakdown_reads_cache_without_a_second_model_call():
    script = ("CLAIM: the 2.5 nm figure | UNSUPPORTED | no basis\n"
              "CLAIM: the 2 hz gait | SUPPORTED | fine")
    skill = _skill(script)
    first = await _run(skill, ctx=_ctx("some answer"))
    assert first.await_reply is True
    n = len(skill._calls)
    detail = await skill.execute(SkillRequest(
        text="yes", context={"captured_reply": True}))
    assert detail.success and "2.5 nm figure" in detail.speech.lower()
    assert len(skill._calls) == n              # NO extra model call
    assert skill._cache is None                # cache consumed


@pytest.mark.asyncio
async def test_breakdown_dropped_on_a_non_affirmative():
    skill = _skill("CLAIM: x | UNSUPPORTED | no basis")
    await _run(skill, ctx=_ctx("some answer"))
    r = await skill.execute(SkillRequest(
        text="no thanks", context={"captured_reply": True}))
    assert r.success and "okay" in r.speech.lower()


def test_matches_expected_phrasings_and_declines_unrelated():
    s = SecondOpinionSkill(load_settings())
    for text in ("second opinion", "get a second opinion", "are you sure",
                 "are you sure about that", "poke holes in that", "fact-check that",
                 "double-check your answer", "дај второ мислење", "сигурен ли си"):
        assert s.match(text) is not None, text
    for text in ("check the time", "are you there", "double-check the address"):
        assert s.match(text) is None, text


def test_verdict_is_not_solid_when_most_claims_went_unverified():
    """1 SUPPORTED among 4 CANNOT_VERIFY must NOT read 'you can rely on it'."""
    from skills.second_opinion import _verdict
    assert _verdict([{"tag": "SUPPORTED"}] + [{"tag": "CANNOT_VERIFY"}] * 4) == "mixed"
    # a verified majority still reads solid; a wrong/unsupported claim dominates.
    assert _verdict([{"tag": "SUPPORTED"}, {"tag": "SUPPORTED"},
                     {"tag": "CANNOT_VERIFY"}]) == "solid"
    assert _verdict([{"tag": "SUPPORTED"}, {"tag": "WRONG"}]) == "wrong"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
