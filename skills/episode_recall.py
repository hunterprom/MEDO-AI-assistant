"""Recalling past conversations: "what did we talk about yesterday?"

The store (:mod:`core.episodes`) holds a few sentences per past session,
embedded. This skill is the way in: it works out WHEN the question is asking
about, searches that window by meaning, and reads back what it finds.

Written bilingually from the first line, not translated afterwards. The fast
path is regex, and a regex written in English is deaf to "за што зборувавме
вчера" — the request then falls through to the LLM, which cheerfully invents a
conversation that never happened. That failure mode is worse here than
anywhere else in MEDO: a fabricated memory is indistinguishable from a real
one. So the Macedonian patterns anchor on the verb ``зборувавме`` / ``разговаравме``
and the time words, and the time-window parser knows ``вчера``, ``синоќа``,
``минатата недела`` alongside their English counterparts.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

from core import mk
from core.config import Settings
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

NOTHING_EN = "I don't have anything remembered from then."
NOTHING_MK = "Немам ништо запаметено од тогаш."
DISABLED_EN = "I'm not keeping memories of past conversations right now."
DISABLED_MK = "Моментално не паметам минати разговори."


def _day(offset: int, now: datetime) -> tuple[datetime, datetime]:
    """The whole calendar day ``offset`` days back, as a half-open window."""
    start = (now - timedelta(days=offset)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def parse_window(text: str, now: datetime | None = None
                 ) -> tuple[datetime, datetime | None] | None:
    """The time window a question is asking about, or None for "any time".

    Windows are half-open and calendar-aligned, because that is what people
    mean: "yesterday" is yesterday's midnight to today's, not the twenty-four
    hours ending now. Bilingual, and deliberately small — a handful of phrases
    people actually say, rather than a date parser that will confidently
    mis-read something.
    """
    now = now or datetime.now().astimezone()
    low = text.lower()

    if re.search(r"\byesterday\b|\bвчера\b|\bсиноќа\b", low):
        return _day(1, now)
    if re.search(r"\btoday\b|\bденес\b|\bутрово\b|\bвечерва\b", low):
        return _day(0, now)
    if re.search(r"\bday\s+before\s+yesterday\b|\bзавчера\b", low):
        return _day(2, now)
    if re.search(r"\blast\s+week\b|\bминатата\s+недела\b|\bминатата\s+седмица\b", low):
        start = (now - timedelta(days=now.weekday() + 7)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=7)
    if re.search(r"\bthis\s+week\b|\bоваа\s+недела\b|\bоваа\s+седмица\b", low):
        start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0)
        return start, None
    if re.search(r"\blast\s+month\b|\bминатиот\s+месец\b", low):
        return now - timedelta(days=60), now - timedelta(days=25)
    match = re.search(r"\b(\d{1,3})\s+days?\s+ago\b|\bпред\s+(\d{1,3})\s+дена\b", low)
    if match:
        days = int(match.group(1) or match.group(2))
        return _day(days, now)
    if re.search(r"\brecently\b|\blately\b|\bодамна\b|\bнеодамна\b", low):
        return now - timedelta(days=14), None
    return None


class EpisodeRecallSkill(Skill):
    """Answer questions about what was discussed in earlier sessions."""

    name = "recall_conversation"
    description = (
        "Recall what the user and MEDO talked about in an EARLIER session "
        "('what did we discuss yesterday', 'what were we working on last "
        "week'). Use only for past CONVERSATIONS, not for remembered facts."
    )
    controls_pc = False
    semantic_from_text = True
    routing_phrases = [
        "what did we talk about yesterday",
        "what were we discussing last week",
        "remind me what we decided in our last conversation",
        "what have we been working on lately",
        "за што зборувавме вчера",
        "што решивме минатата недела",
    ]

    #: EN: the verb pair (talk/discuss/decide/work on) plus a past-time cue.
    #: MK: `зборувавме`/`разговаравме`/`решивме` — first person plural past,
    #: which is how this is actually said, and which no English pattern reaches.
    patterns = [
        re.compile(r"\bwhat\s+(?:did|were)\s+we\s+"
                   r"(?:talk(?:ing)?\s+about|discuss(?:ing)?|say(?:ing)?|"
                   r"decide|working\s+on)\b.*", re.IGNORECASE),
        re.compile(r"\bwhat\s+(?:have\s+)?we\s+(?:been\s+)?"
                   r"(?:talk(?:ed|ing)?\s+about|discuss(?:ed|ing)?|"
                   r"working\s+on)\b.*", re.IGNORECASE),
        re.compile(r"\b(?:remind\s+me\s+)?what\s+(?:we\s+)?"
                   r"(?:decided|agreed|concluded)\b.*", re.IGNORECASE),
        re.compile(r"\bour\s+last\s+conversation\b", re.IGNORECASE),
        re.compile(r"\bза\s+што\s+(?:зборувавме|разговаравме|дискутиравме)\b.*",
                   re.IGNORECASE),
        re.compile(r"\bшто\s+(?:зборувавме|разговаравме|решивме|договоривме)\b.*",
                   re.IGNORECASE),
        re.compile(r"\bна\s+што\s+рабо(?:тевме|тивме)\b.*", re.IGNORECASE),
    ]

    def __init__(self, settings: Settings, store=None) -> None:
        self._settings = settings
        self._store = store

    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "about": {
                            "type": "string",
                            "description": "What the conversation was about, if "
                                           "the user said (e.g. 'the roof'). "
                                           "Empty for 'what did we talk about'.",
                        },
                        "when": {
                            "type": "string",
                            "description": "The time the user named, verbatim "
                                           "('yesterday', 'last week', 'вчера').",
                        },
                    },
                },
            },
        }

    async def execute(self, request: SkillRequest) -> SkillResult:
        speak_mk = mk.is_cyrillic(request.utterance)
        if self._store is None or not self._settings.memory.episodes_enabled:
            return SkillResult(DISABLED_MK if speak_mk else DISABLED_EN,
                               success=False)
        when_text = str(request.args.get("when") or "")
        # The whole utterance is the fallback source for BOTH the window and the
        # topic: on the fast path there are no tool arguments, and on the
        # semantic path there is no regex match either.
        window = parse_window(when_text or request.text)
        about = str(request.args.get("about") or "").strip()
        named = about or self._named_topic(request.text)
        # A NAMED topic must be matched, not approximated. Asked "what did we say
        # about quantum physics", recall used to hand back the most recent
        # episode — the roof — which the listener hears as confirmation that the
        # conversation happened. A broad "what did we talk about" still gets the
        # most recent, because there recency IS the answer.
        query = named or self._topic_from(request.text)

        import asyncio

        episodes = await asyncio.to_thread(
            self._store.search, query, 3, window, bool(named))
        if not episodes:
            return SkillResult(NOTHING_MK if speak_mk else NOTHING_EN,
                               data={"episodes": 0})
        return SkillResult(self._speak(episodes, speak_mk),
                           data={"episodes": len(episodes),
                                 "ids": [e.id for e in episodes]})

    #: "…about the roof" — a topic said out loud. Anchored on the preposition,
    #: so the stock framing ("what did we talk about") yields nothing: "about"
    #: with no object is not a topic.
    #: Matches only the preposition, NOT the rest of the line: the topic is
    #: sliced off after the last one. A pattern that consumed to end-of-string
    #: would match once, from the leftmost preposition, making "…about it, and
    #: about the roof" yield "it, and about the roof" as the subject.
    _NAMED_EN = re.compile(
        r"\b(?:about|regarding|on the subject of)\s+(?=\S)", re.IGNORECASE)

    #: Macedonian needs the VERB as the anchor, not the preposition. ``за`` is
    #: the most common word in these questions — it opens the stock framing
    #: itself ("за што зборувавме"), so keying on it alone read every broad
    #: question as a named topic and then answered every one of them "I don't
    #: have anything". Requiring the verb FIRST means only a ``за`` that comes
    #: after it ("…зборувавме за покривот") is the subject.
    _NAMED_MK = re.compile(
        r"\b(?:зборувавме|разговаравме|дискутиравме|решивме|договоривме|"
        r"работевме|работивме)\b.*?\bза\s+(?=\S)", re.IGNORECASE)

    #: Words that follow "about" in the stock phrasings and are NOT a topic.
    _NOT_A_TOPIC = re.compile(
        r"^(?:yesterday|today|last\s+\w+|this\s+\w+|\d+\s+days?\s+ago|"
        r"recently|lately|вчера|денес|синоќа|завчера|минатата\s+\w+|"
        r"оваа\s+\w+|што|неодамна)\b\s*\??$", re.IGNORECASE)

    @classmethod
    def _named_topic(cls, text: str) -> str:
        """The subject the user named, or "" when they only asked broadly."""
        text = text.strip()
        pattern = cls._NAMED_MK if mk.is_cyrillic(text) else cls._NAMED_EN
        # The LAST preposition, not the first: "what did we talk about the roof"
        # has one, but "…about it, and about the roof" has two and the subject is
        # the later one. `search` would take the leftmost and swallow the rest of
        # the sentence into the topic.
        match = None
        for match in pattern.finditer(text):
            pass
        if match is None:
            return ""
        topic = text[match.end():].strip(" ?.!,")
        if not topic or cls._NOT_A_TOPIC.match(topic):
            return ""
        return topic

    @staticmethod
    def _topic_from(text: str) -> str:
        """The subject of the question, with the framing words removed.

        "What did we talk about regarding the roof" -> "the roof". Left as the
        whole utterance when there is no explicit topic: the embedding of the
        question is still a reasonable query, and a bad topic is worse than a
        broad one.
        """
        stripped = re.sub(
            r"^\s*(?:remind\s+me\s+)?(?:what|за\s+што|што|на\s+што)\s+"
            r"(?:did|were|have|has|we|ние)?\s*", "", text, flags=re.IGNORECASE)
        return stripped.strip(" ?.!") or text

    @staticmethod
    def _speak(episodes, speak_mk: bool) -> str:
        parts = []
        for episode in episodes:
            # Ask for the phrase in the reply's language rather than translating
            # a handful of English ones afterwards: the partial lookup this
            # replaces left "on saturday" and "3 weeks ago" untranslated in front
            # of a Macedonian summary.
            when = episode.when(language="mk" if speak_mk else "en")
            # `str.capitalize()` uppercases the first character AND LOWERCASES
            # THE REST, which turned "on Saturday" into "on saturday" in English
            # too. Only the first character should move.
            when = when[:1].upper() + when[1:]
            # A colon in both languages. A comma gave "On Saturday, We talked
            # about the roof." — the summary is a sentence and starts with a
            # capital, so joining it with a comma reads as a typo.
            parts.append(f"{when}: {episode.summary}")
        # One episode reads as a sentence; several want separating clearly, or
        # the summaries run into each other as one impossible paragraph.
        return " ".join(parts) if len(parts) == 1 else " … ".join(parts)
