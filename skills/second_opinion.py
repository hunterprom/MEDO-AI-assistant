"""Second opinion — the council red-teams MEDO's OWN last answer.

"Are you sure?" / "second opinion" / "poke holes in that" takes the answer MEDO
just gave (and the question that produced it) and hands it to the two or three
relevant specialists as a HOSTILE panel. They do not re-answer; for every claim
in the answer they emit a tag — SUPPORTED, UNSUPPORTED, WRONG or CANNOT_VERIFY —
with a one-clause reason. MEDO then computes a **deterministic, fail-safe**
confidence verdict in Python from those tags: a claim is only "confirmed" when
an auditor supported it and none flagged it, and ANY parse failure, empty reply,
or silence degrades to "treat as unverified" — never to false confidence. The
model can flag a claim but it can never talk MEDO into trusting one.

This is the one council feature that REMOVES a generative surface instead of
adding one: the verdict and the spoken headline are Python over structured tags,
so a wrong auditor drops out rather than fabricating agreement. The per-claim
breakdown is cached and read back on "yes" with no second model call.

Runs on the council brain (fast local model when the selected brain is a CLI
agent), so several auditors in parallel cost no extra processes on Ollama.
Declines cleanly when there is no prior answer or no brain, like RepeatSkill.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from core import mk
from core.config import Settings
from core.council import Specialist, find_specialist, rank_specialists
from core.safety import is_affirmative
from skills.base import SkillRequest, SkillResult
from skills.council import _CouncilBase

logger = logging.getLogger(__name__)

NO_ANSWER = "I haven't answered anything to check yet."
NO_ANSWER_MK = "Сè уште немам одговорено ништо за проверка."
NO_BRAIN = "My brain is offline, so the council can't meet."
NO_BRAIN_MK = "Мозокот е офлајн, па советот не може да работи."
NO_PANEL = "I couldn't get the council to weigh in, so I can't confirm that answer."
NO_PANEL_MK = "Не можев да го свикам советот, па не можам да го потврдам одговорот."

#: What each auditor is told to do. Deliberately NOT the spoken HOUSE_STYLE —
#: we want structured tag lines here, not prose.
_AUDIT_TASK = (
    "You are AUDITING an answer another assistant already gave — do NOT re-answer "
    "the question and do NOT introduce new claims. For each distinct factual or "
    "numeric claim in the answer, output exactly one line:\n"
    "CLAIM: <the claim in a few words> | <SUPPORTED|UNSUPPORTED|WRONG|CANNOT_VERIFY>"
    " | <one short reason>\n"
    "Rules: a number stated with no basis is UNSUPPORTED. If you cannot check a "
    "claim from your own knowledge, use CANNOT_VERIFY — never guess. Use WRONG only "
    "when the claim is actually incorrect. If the honest answer to the original "
    "question was 'I don't know', say the assistant should have declined. Output "
    "ONLY the CLAIM lines, nothing else."
)

_TAG = re.compile(
    r"\b(SUPPORTED|UNSUPPORTED|WRONG|INCORRECT|FALSE|"
    r"CANNOT[\s_-]?VERIFY|UNVERIFIABLE|UNKNOWN)\b", re.IGNORECASE)


#: Tag severity, most-severe-first. Used as the fail-safe tie-break: when a line
#: carries more than one tag word, the most severe wins so a tag word embedded in
#: the CLAIM TEXT (e.g. a claim that reads "...supported by...") can never mask a
#: real flag. The parser over-flags, never under-flags.
_SEVERITY = {"WRONG": 3, "UNSUPPORTED": 2, "CANNOT_VERIFY": 1, "SUPPORTED": 0}


def _norm_tag(raw: str) -> str:
    # Collapse ANY run of whitespace/underscore/hyphen — _TAG matches
    # "CANNOT<any \s>VERIFY" (tab, NBSP…), so a plain " "/"-" replace would leave
    # a stray separator and yield an unknown tag (a _SEVERITY KeyError downstream).
    t = re.sub(r"[\s_-]+", "_", raw.strip().upper())
    if t in ("INCORRECT", "FALSE"):
        return "WRONG"
    if t in ("UNVERIFIABLE", "UNKNOWN", "CANNOT_VERIFY"):
        return "CANNOT_VERIFY"
    return t                                # SUPPORTED | UNSUPPORTED | WRONG


def _clean(fragment: str) -> str:
    """Trim a 'CLAIM:' prefix and the separators around a claim/reason fragment."""
    f = re.sub(r"^\s*claim\s*[:\-]?\s*", "", fragment, flags=re.IGNORECASE)
    return f.strip(" \t|-–—:.•")


def _parse_audit(reply: str, member: Specialist) -> list[dict[str, str]]:
    """One auditor's raw reply -> a list of {claim, tag, reason, specialist,...}.

    Liberal: any line carrying a tag word is a claim; the text before the tag is
    the claim, the text after is the reason. Lines with no tag are ignored, so a
    stray preamble can't invent a claim.
    """
    claims: list[dict[str, str]] = []
    for line in reply.splitlines():
        if _TAG.search(line) is None:
            continue
        fields = [f.strip() for f in line.split("|")]
        tag: str | None = None
        claim = ""
        reason = ""
        if len(fields) >= 2:
            # Well-formed "CLAIM: <claim> | <TAG> | <reason>": read the tag from
            # its DELIMITED field, so a tag word inside the claim summary can't be
            # mistaken for the verdict. The claim is field 0; the tag is the first
            # later field that carries a tag word; the rest is the reason.
            claim = _clean(fields[0])
            tag_idx = next((i for i in range(1, len(fields))
                            if _TAG.search(fields[i])), None)
            if tag_idx is not None:
                tag = _norm_tag(_TAG.search(fields[tag_idx]).group(1))
                reason = _clean(" | ".join(fields[tag_idx + 1:]))
        if tag is None:
            # No usable delimiters — fall back to the MOST SEVERE tag on the line
            # (over-flag, never under-flag) with the text before the first tag as
            # the claim.
            hits = list(_TAG.finditer(line))
            # .get(..., 3): an unrecognised tag ranks most-severe (over-flag,
            # never crash) — the parser must never throw on odd auditor output.
            tag = max((_norm_tag(m.group(1)) for m in hits),
                      key=lambda t: _SEVERITY.get(t, 3))
            claim = _clean(line[:hits[0].start()])
            reason = _clean(line[hits[-1].end():])
        if not claim:                       # a bare tag with no claim text -> skip
            continue
        claims.append({
            "claim": claim,
            "tag": tag,
            "reason": reason,
            "specialist": member.key,
            "specialist_title": member.title,
        })
    return claims


def _verdict(claims: list[dict[str, str]]) -> str:
    """The DETERMINISTIC, fail-safe confidence status. Never optimistic on doubt.

    No claims parsed (empty/garbled auditor output) -> 'unverified', never
    'solid': silence must degrade to unconfirmed, not to confidence.
    """
    tags = {c["tag"] for c in claims}
    if not claims:
        return "unverified"
    if "WRONG" in tags:
        return "wrong"
    if "UNSUPPORTED" in tags:
        return "shaky"
    if "SUPPORTED" in tags:
        return "solid"
    return "unverified"                     # only CANNOT_VERIFY


def _join_titles(members: list[Specialist]) -> str:
    titles = [m.title for m in members]
    if len(titles) == 1:
        return titles[0]
    return ", ".join(titles[:-1]) + " and " + titles[-1]


class SecondOpinionSkill(_CouncilBase):
    """Convene the relevant specialists to audit MEDO's own last answer."""

    name = "second_opinion"
    controls_pc = False
    description = (
        "Fact-check MEDO's own previous answer: convene the relevant council "
        "specialists to audit each claim and return a fail-safe confidence "
        "verdict. Use for 'are you sure?', 'second opinion', 'poke holes in that'.")

    patterns = [
        re.compile(r"\bsecond opinion\b", re.IGNORECASE),
        re.compile(r"^\s*(?:but|and|so|wait|hmm|really)[,\s]*"
                   r"are you (?:sure|certain|positive|right)\b", re.IGNORECASE),
        re.compile(r"^\s*are you (?:sure|certain|positive|right)\b", re.IGNORECASE),
        re.compile(r"\bpoke holes in (?:that|it|this|your answer)\b", re.IGNORECASE),
        re.compile(r"\bfact[\s-]?check (?:that|it|this|your (?:last )?answer)\b",
                   re.IGNORECASE),
        re.compile(r"\bdouble[\s-]?check (?:that|it|your (?:last )?answer)\b",
                   re.IGNORECASE),
        re.compile(r"\b(?:can you |could you |please )?"
                   r"(?:check|verify) (?:that answer|your (?:last )?answer)\b",
                   re.IGNORECASE),
        re.compile(r"\bcouncil[,\s]+(?:please\s+)?(?:check|verify|fact[\s-]?check)\s+"
                   r"(?:that|your|the)\b", re.IGNORECASE),
        # MK: "дај второ мислење", "сигурен ли си", "провери го тоа/одговорот"
        re.compile(r"\bдај(?:те)?\s+(?:ми\s+)?второ мислење\b", re.IGNORECASE),
        re.compile(r"\bсигурен(?:\s+ли)?\s+си\b", re.IGNORECASE),
        re.compile(r"\bпровери\s+го\s+(?:тоа|одговорот|последниот одговор)\b",
                   re.IGNORECASE),
    ]

    def __init__(self, settings: Settings, ask=None) -> None:
        super().__init__(settings, ask)
        #: Last audit, cached for the "want the full breakdown?" follow-up so the
        #: detail is read back with NO second model call.
        self._cache: dict[str, Any] | None = None

    # -- audit one specialist -------------------------------------------------

    def _audit_system(self, member: Specialist, speak_mk: bool) -> str:
        prompt = f"{member.prompt}\n\n{_AUDIT_TASK}"
        if speak_mk:
            prompt += ("\n\nWrite the reason part in Macedonian, but keep the tag "
                       "word (SUPPORTED / UNSUPPORTED / WRONG / CANNOT_VERIFY) in "
                       "English so it can be read back.")
        return prompt

    async def _audit(self, member: Specialist, question: str, answer: str,
                     speak_mk: bool) -> tuple[Specialist, str]:
        """Ask one specialist to audit the answer. Never raises — a dead auditor
        comes back as an empty reply, exactly like _CouncilBase._consult."""
        user = f"Original question:\n{question}\n\nThe answer to audit:\n{answer}"
        try:
            reply = await self._ask(self._audit_system(member, speak_mk), user)
        except Exception:
            logger.warning("auditor %s failed", member.key, exc_info=True)
            return member, ""
        return member, (reply or "").strip()

    # -- execute --------------------------------------------------------------

    async def execute(self, request: SkillRequest) -> SkillResult:
        speak_mk = mk.is_cyrillic(request.text)
        if request.context.get("captured_reply"):
            return self._breakdown(request, speak_mk)

        answer = str(request.context.get("last_reply") or "").strip()
        question = str(request.context.get("last_question") or "").strip()
        if not answer:
            return SkillResult(NO_ANSWER_MK if speak_mk else NO_ANSWER, success=False)
        if self._ask is None:
            return SkillResult(NO_BRAIN_MK if speak_mk else NO_BRAIN, success=False)

        council = self._council
        # Pick who audits by the ANSWER's field, not the (often generic) question:
        # "are you sure?" should hand THIS answer to the major it's closest to —
        # an answer about low-voltage wiring goes to the electrical engineer, not a
        # default. Fall back to the question's field, then the default, only when
        # the answer names no discipline.
        picked = (rank_specialists(answer, council,
                                   self._settings.council.max_members)
                  or rank_specialists(question, council,
                                      self._settings.council.max_members))
        if not picked:
            fallback = find_specialist(self._settings.council.default_agent, council)
            picked = [fallback] if fallback else list(council[:1])
        if not picked:
            return SkillResult(NO_PANEL_MK if speak_mk else NO_PANEL, success=False)

        results = await asyncio.gather(
            *(self._audit(m, question or answer, answer, speak_mk) for m in picked))
        answered = [m for m, reply in results if reply]
        if not answered:
            return SkillResult(NO_PANEL_MK if speak_mk else NO_PANEL, success=False)

        claims: list[dict[str, str]] = []
        for member, reply in results:
            if reply:
                claims.extend(_parse_audit(reply, member))
        status = _verdict(claims)
        names = _join_titles(answered)
        self._cache = {"claims": claims, "status": status, "mk": speak_mk,
                       "names": names}

        headline = self._headline(status, names, claims, speak_mk)
        # Offer the per-claim detail only when the headline actually asks for it.
        # The "unverified" headline has no "want the breakdown?" tail, so it must
        # NOT arm reply-capture — otherwise the next free-form utterance is
        # silently swallowed into the (empty) breakdown branch.
        offer = bool(claims) and status != "unverified"
        return SkillResult(
            headline, success=True, await_reply=offer, reply_is_offer=offer,
            data={"second_opinion": status, "members": [m.key for m in answered],
                  "audit": claims})

    # -- spoken verdict -------------------------------------------------------

    def _headline(self, status: str, names: str, claims: list[dict[str, str]],
                  speak_mk: bool) -> str:
        flagged = [c for c in claims if c["tag"] in ("WRONG", "UNSUPPORTED")]
        top = next((c for c in claims if c["tag"] == "WRONG"),
                   next((c for c in claims if c["tag"] == "UNSUPPORTED"), None))
        tail = " Сакаш ли целосна анализа?" if speak_mk else " Want the full breakdown?"

        if status == "solid":
            body = (f"Го проверив со {names} — ништо не остана непоткрепено, "
                    f"можеш да се потпреш на тоа."
                    if speak_mk else
                    f"I had {names} cross-examine that — nothing came back "
                    f"unsupported, so you can rely on it.")
            return body + tail
        if status == "wrong" and top is not None:
            reason = f" — {top['reason']}" if top["reason"] else ""
            body = (f"Го проверив со {names}. Едно тврдење е погрешно: "
                    f"{top['claim']}{reason}. Третирај го остатокот како непотврден."
                    if speak_mk else
                    f"I had {names} check that. One claim is wrong: "
                    f"{top['claim']}{reason}. Treat the rest as unconfirmed until "
                    f"I can back it up.")
            return body + tail
        if status == "shaky" and top is not None:
            reason = f" — {top['reason']}" if top["reason"] else ""
            body = (f"Го проверив со {names}. Едно тврдење не се држи: "
                    f"{top['claim']} е непоткрепено{reason}. Третирај го како "
                    f"претпоставка, не факт."
                    if speak_mk else
                    f"I had {names} check that. One claim doesn't hold up: "
                    f"{top['claim']} is unsupported{reason}. Treat that as a guess, "
                    f"not a fact.")
            n = len(flagged)
            if n > 1:
                body += (f" ({n - 1} more like it.)" if not speak_mk
                         else f" (Уште {n - 1} слични.)")
            return body + tail
        # unverified (or a defensive fallthrough): fail-safe, no false confidence.
        return (f"Го прашав {names}, но не можеа да го потврдат — третирај го "
                f"одговорот како непотврден." if speak_mk else
                f"I ran that past {names}, but they couldn't verify it either way "
                f"— treat that answer as unconfirmed.")

    def _breakdown(self, request: SkillRequest, speak_mk: bool) -> SkillResult:
        """Read the cached per-claim audit back — no model call."""
        cache = self._cache
        self._cache = None
        if not cache or not cache.get("claims"):
            return SkillResult("Немам детали." if speak_mk
                               else "There's nothing left to break down.",
                               success=True)
        # Only dump the detail on an affirmative; anything else just drops it.
        if not (is_affirmative(request.text)
                or re.search(r"break\s*down|detail|full|the rest|go on|"
                             r"деталн|анализ|остатак",
                             request.text, re.IGNORECASE)):
            return SkillResult("Океј." if speak_mk else "Okay.", success=True)
        parts: list[str] = []
        for c in cache["claims"]:
            tag = c["tag"].replace("_", " ").lower()
            reason = f", {c['reason']}" if c["reason"] else ""
            parts.append(f"{c['specialist_title']} — {c['claim']}: {tag}{reason}.")
        return SkillResult(" ".join(parts), success=True,
                           data={"audit": cache["claims"],
                                 "second_opinion": cache.get("status")})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
