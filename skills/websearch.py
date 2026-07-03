"""Web search via DuckDuckGo (keyless), summarized by the local LLM.

Two routes, one skill:

* **Fast path** ("search for X"): fetch results, then summarize them with the
  injected ``summarize`` coroutine so the spoken answer is a paragraph, not a list.
* **LLM tool path**: the model calls ``web_search`` and gets the raw results back
  to summarize itself (``via=tool`` in the context), avoiding a double LLM hop.

Offline, DuckDuckGo raises and we return a clean spoken message.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any

from skills.base import Skill, SkillRequest, SkillResult

Summarize = Callable[[str, str], Awaitable[str]]
_OFFLINE = "I can't search the web right now. I appear to be offline."
_MAX_RESULTS = 5


class WebSearchSkill(Skill):
    name = "web_search"
    description = "Search the web and summarize the results for a query."

    patterns = [
        re.compile(r"\b(?:search|google|look\s+up)\s+(?:the\s+web\s+for\s+|for\s+)?(?P<q>.+)", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+the\s+latest\s+(?:on|about)\s+(?P<q2>.+)", re.IGNORECASE),
    ]

    def __init__(self, summarize: Summarize | None = None) -> None:
        self._summarize = summarize

    def _search(self, query: str) -> list[dict[str, str]] | None:
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=_MAX_RESULTS))
        except Exception:  # network down, rate limit, parser change, etc.
            return None

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        query = (request.args.get("query") or gd.get("q") or gd.get("q2") or "").strip(" ?.!")
        if not query:
            return SkillResult("What should I search for?", success=False)

        results = await asyncio.to_thread(self._search, query)
        if results is None:
            return SkillResult(_OFFLINE, success=False)
        if not results:
            return SkillResult(f"I couldn't find anything about {query}.", success=False)

        block = "\n".join(
            f"- {r.get('title', '')}: {r.get('body', '')}" for r in results
        )

        # Called by the model as a tool: hand back raw results for it to summarize.
        if request.context.get("via") == "tool" or self._summarize is None:
            return SkillResult(block, data={"query": query, "results": results})

        summary = await self._summarize(query, block)
        return SkillResult(summary, data={"query": query, "results": results})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "the search query"}
                    },
                    "required": ["query"],
                },
            },
        }
