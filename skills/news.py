"""Headlines from RSS feeds (configured in config.yaml) via feedparser.

Feeds are fetched with httpx so network failures are caught explicitly and turned
into a clean offline message; the bytes are then handed to feedparser to parse.
"""

from __future__ import annotations

import re
from typing import Any

from core import mk
from core.config import NewsConfig
from skills.base import Skill, SkillRequest, SkillResult

_OFFLINE = "I can't fetch the news right now. I appear to be offline."
_OFFLINE_MK = "Не можам да ги земам вестите сега — изгледа дека сум офлајн."


class NewsSkill(Skill):
    name = "news"
    description = "Read the latest news headlines."
    routing_phrases = [
        "what's the news", "catch me up on the headlines", "what's happening today",
        "anything new in the world", "give me the latest",
    ]

    patterns = [
        re.compile(r"\b(?:the\s+)?news\b", re.IGNORECASE),
        re.compile(r"\bheadlines?\b", re.IGNORECASE),
        re.compile(r"\bwhat(?:'?s| is)\s+happening\b", re.IGNORECASE),
        # Natural ways people ask for general/world headlines (no topic named —
        # a named topic is a web search, which WebSearchSkill owns).
        re.compile(r"\b(?:world|worldwide|global|international)\s+news\b", re.IGNORECASE),
        re.compile(r"\btell\s+me\s+about\s+(?:the\s+)?(?:world|worldwide|current\s+events?)\b", re.IGNORECASE),
        re.compile(r"\bwhat(?:'?s| is)\s+going\s+on\s+in\s+the\s+world\b", re.IGNORECASE),
        re.compile(r"\bcurrent\s+events\b", re.IGNORECASE),
        # MK: "вести", "најсвежи вести за Скопје", "наслови", "што има ново".
        # Deliberately NOT "што е ново за X" — naming a topic is a web search,
        # and WebSearchSkill owns that one.
        re.compile(r"\bвест(?:и|ите|ите)\b|\bновост(?:и|ите)\b", re.IGNORECASE),
        re.compile(r"\bнаслови(?:те)?\b", re.IGNORECASE),
        re.compile(r"\bшто\s+има\s+ново\b|\bшто\s+се\s+случува\b", re.IGNORECASE),
        re.compile(r"\bшто\s+се\s+случува\s+во\s+светот\b|\bсветски\s+вести\b", re.IGNORECASE),
    ]

    def __init__(self, config: NewsConfig, max_items: int = 5) -> None:
        self._feeds = config.feeds
        self._feeds_mk = config.feeds_mk
        self._max = max_items

    def _pick_feeds(self, speak_mk: bool) -> list[str]:
        """Macedonian question -> Macedonian sources, when any are configured.

        Reading BBC headlines back to someone who asked "најсвежи вести" is a
        worse answer than the language mismatch suggests: they want *local*
        news, which an English feed doesn't carry either.
        """
        if speak_mk and self._feeds_mk:
            return self._feeds_mk
        return self._feeds

    async def execute(self, request: SkillRequest) -> SkillResult:
        import feedparser
        import httpx

        speak_mk = mk.is_cyrillic(request.text)
        feeds = self._pick_feeds(speak_mk)
        if not feeds:
            return SkillResult(
                "Нема конфигурирани извори за вести." if speak_mk
                else "No news feeds are configured.", success=False)

        headlines: list[str] = []
        reached_any = False
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                for url in feeds:
                    try:
                        resp = await client.get(url)
                        resp.raise_for_status()
                    except httpx.HTTPError:
                        continue  # skip a single dead feed
                    reached_any = True
                    parsed = feedparser.parse(resp.content)
                    for entry in parsed.entries[: self._max]:
                        title = entry.get("title", "").strip()
                        if title:
                            headlines.append(title)
        except httpx.HTTPError:
            return SkillResult(_OFFLINE_MK if speak_mk else _OFFLINE, success=False)

        if not reached_any:
            return SkillResult(_OFFLINE_MK if speak_mk else _OFFLINE, success=False)
        if not headlines:
            return SkillResult(
                "Не најдов наслови во моментов." if speak_mk
                else "I couldn't find any headlines just now.", success=False)

        top = headlines[: self._max]
        spoken = ". ".join(f"{i}. {h}" for i, h in enumerate(top, 1))
        return SkillResult(
            f"Еве ги главните вести. {spoken}." if speak_mk
            else f"Here are the top headlines. {spoken}.",
            data={"count": len(top)})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer", "description": "how many headlines"}
                    },
                    "required": [],
                },
            },
        }
