"""Long-term memory skills: remember / recall / forget user facts.

Facts persist in sqlite (:class:`core.facts.FactsStore`) and the router
injects the newest ones into the LLM system prompt every turn — so
"remember that my sister is called Ana" actually sticks across restarts
(the v1 jarvis-web feature the Python rewrite was missing).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from core.facts import FactsStore
from skills.base import Skill, SkillRequest, SkillResult

#: Pronouns that make "forget <needle>" too vague to act on safely.
_VAGUE_NEEDLES = {"it", "that", "this", "them"}


class RecallFactsSkill(Skill):
    """Registered before Remember so "what do you remember" is never stored."""

    name = "recall_facts"
    description = "List the facts MEDO has remembered about the user."

    patterns = [
        re.compile(r"\bwhat\s+do\s+you\s+(?:remember|know)\s+about\s+me\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+have\s+you\s+remembered\b", re.IGNORECASE),
        re.compile(r"\b(?:list|show|read)\s+(?:your\s+|my\s+)?(?:remembered\s+)?facts\b", re.IGNORECASE),
    ]

    def __init__(self, store: FactsStore, limit: int = 20) -> None:
        self._store = store
        self._limit = limit

    async def execute(self, request: SkillRequest) -> SkillResult:
        facts = await asyncio.to_thread(self._store.recent, self._limit)
        if not facts:
            return SkillResult("I haven't remembered anything about you yet.")
        spoken = "; ".join(facts[:10])
        more = f" — and {len(facts) - 10} more" if len(facts) > 10 else ""
        return SkillResult(
            f"I remember {len(facts)} things: {spoken}{more}.",
            data={"facts": facts},
        )

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }


class ForgetFactSkill(Skill):
    name = "forget_fact"
    description = "Forget remembered facts that contain the given words."

    patterns = [
        re.compile(r"^\s*(?:medo[,!\s]+)?forget\s+(?:that\s+|about\s+)?(?P<needle>.+?)\s*$", re.IGNORECASE),
        re.compile(r"\bdelete\s+(?:the\s+)?fact\s+(?:about\s+)?(?P<needle2>.+)$", re.IGNORECASE),
    ]

    def __init__(self, store: FactsStore) -> None:
        self._store = store

    async def execute(self, request: SkillRequest) -> SkillResult:
        needle = str(request.args.get("query") or "").strip()
        if not needle and request.match:
            gd = request.match.groupdict()
            needle = (gd.get("needle") or gd.get("needle2") or "").strip()
        if not needle or needle.lower() in _VAGUE_NEEDLES:
            return SkillResult(
                "Tell me what to forget — for example, 'forget that I like coffee'.",
                success=False,
            )
        count = await asyncio.to_thread(self._store.forget, needle)
        if count:
            plural = "fact" if count == 1 else "facts"
            return SkillResult(f"Forgotten — {count} {plural} about that.", data={"count": count})
        return SkillResult("I don't have anything like that remembered.")

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Words the facts to delete contain.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }


class RememberFactSkill(Skill):
    name = "remember_fact"
    description = "Remember a fact about the user permanently."

    # Anchored to the start so questions like "what do you remember" (recall)
    # or "did you remember to..." can never be stored as facts.
    patterns = [
        re.compile(
            r"^\s*(?:medo[,!\s]+)?(?:please\s+)?remember\s+(?:that\s+)?(?P<fact>.+?)\s*$",
            re.IGNORECASE,
        ),
    ]

    def __init__(self, store: FactsStore) -> None:
        self._store = store

    async def execute(self, request: SkillRequest) -> SkillResult:
        fact = str(request.args.get("fact") or "").strip()
        if not fact and request.match:
            fact = (request.match.groupdict().get("fact") or "").strip()
        if not fact:
            return SkillResult("What should I remember?", success=False)
        added = await asyncio.to_thread(self._store.add, fact)
        if added:
            return SkillResult("Got it — I'll remember that.", data={"fact": fact})
        return SkillResult("I already knew that.", data={"fact": fact})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fact": {
                            "type": "string",
                            "description": "The fact to store, phrased as a statement.",
                        }
                    },
                    "required": ["fact"],
                },
            },
        }
