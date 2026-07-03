"""Headlines from RSS feeds (configured in config.yaml) via feedparser.

Feeds are fetched with httpx so network failures are caught explicitly and turned
into a clean offline message; the bytes are then handed to feedparser to parse.
"""

from __future__ import annotations

import re
from typing import Any

from core.config import NewsConfig
from skills.base import Skill, SkillRequest, SkillResult

_OFFLINE = "I can't fetch the news right now. I appear to be offline."


class NewsSkill(Skill):
    name = "news"
    description = "Read the latest news headlines."

    patterns = [
        re.compile(r"\b(?:the\s+)?news\b", re.IGNORECASE),
        re.compile(r"\bheadlines?\b", re.IGNORECASE),
        re.compile(r"\bwhat(?:'?s| is)\s+happening\b", re.IGNORECASE),
    ]

    def __init__(self, config: NewsConfig, max_items: int = 5) -> None:
        self._feeds = config.feeds
        self._max = max_items

    async def execute(self, request: SkillRequest) -> SkillResult:
        import feedparser
        import httpx

        if not self._feeds:
            return SkillResult("No news feeds are configured.", success=False)

        headlines: list[str] = []
        reached_any = False
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                for url in self._feeds:
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
            return SkillResult(_OFFLINE, success=False)

        if not reached_any:
            return SkillResult(_OFFLINE, success=False)
        if not headlines:
            return SkillResult("I couldn't find any headlines just now.", success=False)

        top = headlines[: self._max]
        spoken = ". ".join(f"{i}. {h}" for i, h in enumerate(top, 1))
        return SkillResult(f"Here are the top headlines. {spoken}.", data={"count": len(top)})

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
