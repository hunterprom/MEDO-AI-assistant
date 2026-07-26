"""Find and open files — restricted to the configured directory whitelist.

Every path this skill touches is checked against :class:`core.safety.PathWhitelist`,
so it can never open or reveal anything outside the directories you allow in
config.yaml. Search is a filename substring match across those trees.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core import mk
from core.platform import open_path
from core.safety import PathWhitelist
from skills.base import Skill, SkillRequest, SkillResult

_MAX_RESULTS = 25


def search_files(whitelist: PathWhitelist, query: str,
                 limit: int = _MAX_RESULTS) -> list[Path]:
    """Filename substring search across the whitelisted trees.

    Shared with :mod:`skills.file_edit`, which must resolve a spoken file name
    the same way this skill does — and, more importantly, must be confined to
    exactly the same directories.
    """
    query = query.lower().strip().strip("?.!")
    hits: list[Path] = []
    for root in whitelist.roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if query in path.name.lower() and whitelist.is_allowed(path):
                hits.append(path)
                if len(hits) >= limit:
                    return hits
    return hits

#: "file" in Macedonian, bare and with the definite article, plus the
#: colloquial borrowing. Longest first — "датотеката" must win over "датотека".
_MK_FILE = r"датотеките|датотеката|датотека|документот|документ|фајлот|фајл"

#: "on my computer" — the local-disk hint that separates a file search from a
#: web search. "drive" is left out on purpose: "search my drive" means Google
#: Drive to most people, and the site-search skill claims it first.
_EN_DISK = r"computer|pc|laptop|disk|hard\s+drive|machine"
_MK_DISK = r"компјутерот|компјутер|лаптопот|лаптоп|дискот|диск"


class FilesSkill(Skill):
    name = "files"
    controls_pc = True
    description = "Search for and open files within whitelisted folders."

    patterns = [
        re.compile(r"\b(?:find|search\s+for|locate)\s+(?:a\s+|the\s+)?file[s]?\s+(?:named\s+|called\s+)?(?P<query>.+)", re.IGNORECASE),
        re.compile(r"\bopen\s+(?:the\s+)?file\s+(?:named\s+|called\s+)?(?P<open>.+)", re.IGNORECASE),
        # "search my computer for the invoice"
        re.compile(rf"\b(?:search|look)\s+(?:on\s+|in\s+|through\s+)?(?:my|the)\s+"
                   rf"(?:{_EN_DISK})\s+(?:for\s+)?(?P<query>.+)", re.IGNORECASE),
        # "find the invoice on my computer", "pull up the invoice on my laptop".
        # The trailing disk hint keeps "pull up"/"bring up" unambiguous here (a
        # bare "pull up X" is still a website/app, not a file search).
        re.compile(rf"\b(?:find|locate|search\s+for|pull\s+up|bring\s+up)\s+"
                   rf"(?P<query>.+?)\s+on\s+(?:my|the)\s+(?:{_EN_DISK})\b",
                   re.IGNORECASE),
        # MK: "најди ја датотеката извештај", "барај фајл извештај"
        re.compile(rf"\b(?:{mk.SEARCH_OR_FIND}){mk.CLITICS}\s+(?:{_MK_FILE})\s+"
                   rf"(?:со\s+име\s+|со\s+наслов\s+)?(?P<query>.+)", re.IGNORECASE),
        # MK: "отвори ја датотеката извештај"
        re.compile(rf"\b(?:{mk.OPEN}){mk.CLITICS}\s+(?:{_MK_FILE})\s+"
                   rf"(?:со\s+име\s+|со\s+наслов\s+)?(?P<open>.+)", re.IGNORECASE),
        # MK: "најди извештај на компјутерот"
        re.compile(rf"\b(?:{mk.SEARCH_OR_FIND}){mk.CLITICS}\s+(?P<query>.+?)\s+"
                   rf"(?:на|во)\s+(?:мојот\s+|мојата\s+)?(?:{_MK_DISK})\b", re.IGNORECASE),
    ]

    def __init__(self, whitelist: PathWhitelist) -> None:
        self._whitelist = whitelist

    def _search(self, query: str) -> list[Path]:
        return search_files(self._whitelist, query, _MAX_RESULTS)

    async def execute(self, request: SkillRequest) -> SkillResult:
        speak_mk = mk.is_cyrillic(request.text)
        if not self._whitelist.roots:
            return SkillResult(
                "Нема дозволени папки, па не можам да барам датотеки." if speak_mk
                else "No file directories are whitelisted, so I can't search files.",
                success=False,
            )
        m = request.match
        gd = m.groupdict() if m else {}
        # LLM tool path: the model passes a `query` (and maybe action=open) that
        # the fast-path regex groups don't carry.
        args = request.args or {}
        query = (gd.get("query") or gd.get("open")
                 or str(args.get("query") or "")).strip().strip("?.!")
        wants_open = bool(gd.get("open")) or str(args.get("action") or "").lower() == "open"
        if not query:
            return SkillResult("Која датотека?" if speak_mk else "Which file?",
                               success=False)

        hits = self._search(query)
        if not hits:
            return SkillResult(
                f"Не најдов датотека со '{query}' во {self._whitelist.describe()}."
                if speak_mk else
                f"I couldn't find a file matching '{query}' in {self._whitelist.describe()}.",
                success=False,
            )

        # "open ..." -> open the single best match (after a final whitelist check).
        if wants_open:
            target = hits[0]
            if not self._whitelist.is_allowed(target):
                return SkillResult(
                    "Таа датотека е надвор од дозволените папки." if speak_mk
                    else "That file is outside the allowed folders.", success=False)
            open_path(target)
            return SkillResult(
                f"Отворам {target.name}." if speak_mk else f"Opening {target.name}.",
                data={"path": str(target)})

        names = ", ".join(h.name for h in hits[:8])
        more_mk = f" и уште {len(hits) - 8}" if len(hits) > 8 else ""
        more = f" and {len(hits) - 8} more" if len(hits) > 8 else ""
        if speak_mk:
            return SkillResult(f"Најдов {len(hits)} датотеки: {names}{more_mk}.",
                               data={"count": len(hits)})
        return SkillResult(
            f"Found {len(hits)} file{'s' if len(hits) != 1 else ''}: {names}{more}.",
            data={"count": len(hits)},
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
                        "action": {"type": "string", "enum": ["find", "open"]},
                        "query": {"type": "string", "description": "filename or substring"},
                    },
                    "required": ["action", "query"],
                },
            },
        }
