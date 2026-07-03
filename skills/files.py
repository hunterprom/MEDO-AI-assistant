"""Find and open files — restricted to the configured directory whitelist.

Every path this skill touches is checked against :class:`core.safety.PathWhitelist`,
so it can never open or reveal anything outside the directories you allow in
config.yaml. Search is a filename substring match across those trees.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.platform import open_path
from core.safety import PathWhitelist
from skills.base import Skill, SkillRequest, SkillResult

_MAX_RESULTS = 25


class FilesSkill(Skill):
    name = "files"
    description = "Search for and open files within whitelisted folders."

    patterns = [
        re.compile(r"\b(?:find|search\s+for|locate)\s+(?:a\s+|the\s+)?file[s]?\s+(?:named\s+|called\s+)?(?P<query>.+)", re.IGNORECASE),
        re.compile(r"\bopen\s+(?:the\s+)?file\s+(?:named\s+|called\s+)?(?P<open>.+)", re.IGNORECASE),
    ]

    def __init__(self, whitelist: PathWhitelist) -> None:
        self._whitelist = whitelist

    def _search(self, query: str) -> list[Path]:
        query = query.lower().strip().strip("?.!")
        hits: list[Path] = []
        for root in self._whitelist.roots:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if query in path.name.lower() and self._whitelist.is_allowed(path):
                    hits.append(path)
                    if len(hits) >= _MAX_RESULTS:
                        return hits
        return hits

    async def execute(self, request: SkillRequest) -> SkillResult:
        if not self._whitelist.roots:
            return SkillResult(
                "No file directories are whitelisted, so I can't search files.",
                success=False,
            )
        m = request.match
        gd = m.groupdict() if m else {}
        query = (gd.get("query") or gd.get("open") or "").strip().strip("?.!")
        if not query:
            return SkillResult("Which file?", success=False)

        hits = self._search(query)
        if not hits:
            return SkillResult(
                f"I couldn't find a file matching '{query}' in {self._whitelist.describe()}.",
                success=False,
            )

        # "open ..." -> open the single best match (after a final whitelist check).
        if gd.get("open"):
            target = hits[0]
            if not self._whitelist.is_allowed(target):
                return SkillResult("That file is outside the allowed folders.", success=False)
            open_path(target)
            return SkillResult(f"Opening {target.name}.", data={"path": str(target)})

        names = ", ".join(h.name for h in hits[:8])
        more = f" and {len(hits) - 8} more" if len(hits) > 8 else ""
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
