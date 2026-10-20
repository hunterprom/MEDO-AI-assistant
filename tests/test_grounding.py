"""A summary may only state figures its source stated.

Reproduced live, with the real DuckDuckGo results for "Mazda RX-7":

    Over 800,000 units of the RX-7 were produced during its lifetime.

None of the five snippets mentions a production figure. The rest of the answer
was accurate — 1978 to 2002, three generations, the Wankel rotary — which is
what makes this shape expensive: the invented number rides along inside true
prose, in the one register a listener cannot check.

The session that started this had the same failure with a different number:
"nearly half a million RX-7s sold across the model's seven-year run" (the car
ran 24 years; the FD alone ran 10).

Tightening the prompt was tried FIRST and did not work — told in plain words to
use only the results, llama3.2:3b still said 800,000. Hence a mechanical check.
"""

from __future__ import annotations

import pytest

from core.grounding import (
    drop_unsupported,
    ground_summary,
    repair_demand,
    ungrounded_figures,
)

# The real snippets, trimmed. No production figure anywhere in them.
SOURCE = (
    "- Mazda RX-7: The Mazda RX-7 is a sports car which was manufactured and "
    "marketed by Mazda from 1978 to 2002 across three generations. It has a "
    "front mid engine, rear wheel drive layout.\n"
    "- Mazda RX-7: renowned for its innovative Wankel rotary engine.\n"
    "- Mazda RX-7 FD (1992 - 2002): a '90s Japanese icon | evo: In Mazda's "
    "lineup, the RX badge means only one thing: a rotary engine."
)


# --- catching the invention --------------------------------------------------


def test_the_live_failure_is_caught():
    summary = ("The Mazda RX-7 is a sports car manufactured from 1978 to 2002, "
               "known for its Wankel rotary engine. Over 800,000 units of the "
               "RX-7 were produced during its lifetime.")
    assert ungrounded_figures(summary, SOURCE) == ["800,000"]


def test_the_original_transcript_failure_is_caught():
    # "nearly half a million" carries no digits at all — a digit-only check
    # would sail straight past it.
    summary = ("Nearly half a million RX-7s sold across the model's seven-year "
               "run in markets like the US, Australia, and Japan.")
    assert "million" in ungrounded_figures(summary, SOURCE)


def test_a_faithful_summary_passes_untouched():
    summary = ("The Mazda RX-7 is a sports car Mazda made from 1978 to 2002 "
               "across three generations, built around a Wankel rotary engine.")
    assert ungrounded_figures(summary, SOURCE) == []


def test_a_grounded_figure_passes_however_it_is_written():
    source = "- Sales: Mazda sold 811,634 RX-7s in total."
    assert ungrounded_figures("Mazda sold 811,634 of them.", source) == []
    assert ungrounded_figures("Mazda sold 811634 of them.", source) == []


def test_digits_are_not_grounded_by_accident_across_two_source_numbers():
    # The first cut concatenated every digit in the source into one blob, so
    # "5 GB" next to "13 units" spelled "513" and vouched for a figure nobody
    # wrote. Each figure has to match a whole figure.
    source = "- Specs: 5 GB of RAM, 13 units in stock, 8 ports."
    assert ungrounded_figures("It shipped 513 units.", source) == ["513"]


def test_a_rounding_is_flagged_and_that_is_the_intended_trade():
    # Deliberate: one re-ask costs less than an invented number spoken as fact.
    source = "- Sales: Mazda sold 811,634 RX-7s in total."
    assert ungrounded_figures("Mazda sold over 800,000.", source) == ["800,000"]


def test_a_grounded_magnitude_word_passes():
    source = "- Mazda built over eight hundred thousand of them."
    assert ungrounded_figures("Mazda built hundreds of thousands.", source) == []


# --- what it must NOT flag ---------------------------------------------------


@pytest.mark.parametrize("summary", [
    # Years are the numbers that legitimately get reworded.
    "It ran from 1978 until 2002.",
    "The FD arrived in 1992.",
    # Small counts paraphrase constantly.
    "There were three generations and two body styles.",
    "It has one rotor housing per chamber.",
    # No numbers at all.
    "A Japanese sports car with a rotary engine.",
    "",
])
def test_reworded_and_countable_prose_is_left_alone(summary):
    assert ungrounded_figures(summary, SOURCE) == []


def test_a_percentage_is_checked_even_though_it_is_small():
    # 40 is under the magnitude floor, but "40% lighter" is exactly the kind of
    # claim that gets invented, so a "%" makes any number a claim.
    assert ungrounded_figures("It was 40% lighter.", SOURCE) == ["40"]
    assert ungrounded_figures("It was 40% lighter.", "- 40% lighter than rivals") == []


def test_no_source_means_no_verdict():
    # Nothing to check against is not the same as everything being invented.
    assert ungrounded_figures("Over 800,000 units.", "") == []


# --- the fallback ------------------------------------------------------------


def test_dropping_the_sentence_keeps_the_accurate_ones():
    summary = ("The Mazda RX-7 is a sports car manufactured from 1978 to 2002. "
               "Over 800,000 units were produced. It used a Wankel rotary engine.")
    kept = drop_unsupported(summary, ["800,000"])
    assert kept == ("The Mazda RX-7 is a sports car manufactured from 1978 to "
                    "2002. It used a Wankel rotary engine.")


def test_dropping_everything_leaves_nothing_rather_than_a_lie():
    assert drop_unsupported("Over 800,000 units were produced.", ["800,000"]) == ""


def test_the_demand_names_the_figure():
    demand = repair_demand(["800,000"])
    assert "800,000" in demand and "do not contain" in demand


# --- the whole loop ----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_clean_summary_never_costs_a_second_call():
    async def _reask(demand):
        pytest.fail("re-asked a summary that invented nothing")

    good = "A sports car made from 1978 to 2002 with a rotary engine."
    assert await ground_summary(good, SOURCE, _reask) == good


@pytest.mark.asyncio
async def test_a_good_retry_replaces_the_invented_one():
    fixed = "A sports car Mazda made from 1978 to 2002, with a rotary engine."

    async def _reask(demand):
        return fixed

    bad = "Made from 1978 to 2002. Over 800,000 units were produced."
    assert await ground_summary(bad, SOURCE, _reask) == fixed


@pytest.mark.asyncio
async def test_a_retry_that_invents_again_gets_the_sentence_dropped():
    async def _reask(demand):
        return ("Mazda made it from 1978 to 2002. "
                "Roughly 900,000 were sold worldwide.")

    bad = "Made from 1978 to 2002. Over 800,000 units were produced."
    out = await ground_summary(bad, SOURCE, _reask)
    assert out == "Mazda made it from 1978 to 2002."


@pytest.mark.asyncio
async def test_an_offline_retry_falls_back_to_dropping():
    async def _reask(demand):
        return ""

    bad = "Made from 1978 to 2002. Over 800,000 units were produced."
    assert await ground_summary(bad, SOURCE, _reask) == "Made from 1978 to 2002."


@pytest.mark.asyncio
async def test_a_crashing_retry_never_costs_the_whole_answer():
    async def _reask(demand):
        raise RuntimeError("model died")

    bad = "Made from 1978 to 2002. Over 800,000 units were produced."
    assert await ground_summary(bad, SOURCE, _reask) == "Made from 1978 to 2002."
