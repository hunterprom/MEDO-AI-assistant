"""Ask questions over the user's own documents (RAG).

"What do my documents say about the apartment contract?" — semantic search over
the chunk index (:class:`core.docindex.DocumentIndex`), which covers .txt/.md/
.pdf files in the safety-whitelisted folders. Fast path speaks the best
passage; on the LLM tool path the model gets the top passages (with sources)
to synthesize a grounded answer.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from skills.base import Skill, SkillRequest, SkillResult

_NO_INDEX = ("Document search isn't available right now — the embedding "
             "model is offline.")


class DocumentsSkill(Skill):
    name = "search_documents"
    description = (
        "Search the user's own local documents, notes and PDFs for a topic "
        "and quote what they say. Use for questions about the user's files, "
        "contracts, notes, papers."
    )
    routing_phrases = [
        "what do my notes say about", "find it in my documents",
        "search my files for", "look through my pdfs for", "what did I write about",
        "does my contract mention", "what does my lease say about",
        "according to my documents", "is there anything in my files about",
    ]
    # Reached by MEANING, the whole utterance is the RAG query — so "what did I
    # write about the budget" searches the user's own files instead of letting
    # the LLM invent what they "wrote".
    semantic_from_text = True

    patterns = [
        re.compile(r"\b(?:search|look\s+in|check)\s+my\s+(?:documents|docs|notes|files)\s+(?:for\s+)?(?P<q>.+)", re.IGNORECASE),
        re.compile(r"\bwhat\s+do\s+my\s+(?:documents|docs|notes|files)\s+say\s+about\s+(?P<q2>.+)", re.IGNORECASE),
    ]

    def __init__(self, index) -> None:
        self._index = index

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        query = (request.args.get("query") or gd.get("q") or gd.get("q2") or "").strip(" ?.!")
        if not query and not request.match and not request.args:
            # Reached by MEANING (semantic tier): the utterance is the query.
            query = request.text.strip(" ?.!")
        if not query:
            return SkillResult("What should I look for in your documents?", success=False)

        hits = await asyncio.to_thread(self._index.search, query, 4)
        if not hits:
            stats = await asyncio.to_thread(self._index.stats)
            if not stats["enabled"]:
                return SkillResult(_NO_INDEX, success=False)
            if stats["chunks"] == 0:
                return SkillResult(
                    "I haven't indexed any documents yet — put text, markdown "
                    "or PDF files in your whitelisted folders.", success=False)
            return SkillResult(f"Nothing in your documents mentions {query}.",
                               success=False)

        # Tool path: hand the model the passages + sources to synthesize from.
        if request.context.get("via") == "tool":
            block = "\n\n".join(
                f"[{Path(h['path']).name}]\n{h['text']}" for h in hits
            )
            return SkillResult(block, data={"query": query, "hits": hits})

        # Fast path: speak the best passage, name the file it came from.
        top = hits[0]
        snippet = " ".join(top["text"].split())
        if len(snippet) > 280:
            snippet = snippet[:280].rsplit(" ", 1)[0] + "…"
        return SkillResult(
            f"From {Path(top['path']).name}: {snippet}",
            data={"query": query, "hits": hits},
        )

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "topic to find in the user's documents"}
                    },
                    "required": ["query"],
                },
            },
        }
