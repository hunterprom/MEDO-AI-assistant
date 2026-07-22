"""Obsidian plugin: capture notes into the vault, search it, open the app.

Uses the vault Obsidian itself registers in ``%APPDATA%/obsidian/obsidian.json``
(falling back to MEDO's ``docs/`` vault). Notes land in an ``Inbox/`` folder as
timestamped Markdown, so they sync/appear in Obsidian immediately. Searching
delegates to the documents RAG index — the vault lives under a whitelisted
folder, so its notes are already embedded and searchable by meaning.

Voice examples:
    "obsidian note buy a new webcam stand"
    "search obsidian for the roadmap"
    "open obsidian"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)


def find_vault() -> Path | None:
    """The most recently used Obsidian vault, or MEDO's docs vault, or None."""
    try:
        cfg = Path(os.environ.get("APPDATA", "")) / "obsidian" / "obsidian.json"
        vaults = json.loads(cfg.read_text(encoding="utf-8")).get("vaults", {})
        best = None
        for v in vaults.values():
            if best is None or v.get("ts", 0) > best.get("ts", 0):
                best = v
        if best and Path(best["path"]).exists():
            return Path(best["path"])
    except Exception:
        pass
    fallback = Path(__file__).resolve().parent.parent / "docs"
    return fallback if fallback.exists() else None


class ObsidianNoteSkill(Skill):
    name = "obsidian_note"
    description = (
        "Save a note into the user's Obsidian vault. Use when the user wants "
        "to capture a thought, idea, or note in Obsidian."
    )

    patterns = [
        re.compile(r"\bobsidian\s+note[:,]?\s+(?P<body>.+)", re.IGNORECASE),
        re.compile(r"\b(?:add|create|make)\s+(?:an?\s+)?(?:note\s+(?:in|to)\s+obsidian|obsidian\s+note)[:,]?\s*(?P<body2>.*)", re.IGNORECASE),
    ]

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        body = (request.args.get("text") or gd.get("body") or gd.get("body2") or "").strip()
        if not body:
            return SkillResult("What should the Obsidian note say?", success=False)
        vault = find_vault()
        if vault is None:
            return SkillResult("I couldn't find an Obsidian vault.", success=False)
        inbox = vault / "Inbox"
        inbox.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
        path = inbox / f"{stamp} voice note.md"
        path.write_text(f"{body}\n\n*— captured by MEDO {stamp}*\n", encoding="utf-8")
        return SkillResult(f"Noted in Obsidian: {body[:60]}", data={"path": str(path)})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string", "description": "the note body"}},
                    "required": ["text"],
                },
            },
        }


class ObsidianSearchSkill(Skill):
    name = "obsidian_search"
    description = ("Search the user's Obsidian vault notes for a topic and "
                   "quote the best match.")

    patterns = [
        re.compile(r"\bsearch\s+obsidian\s+(?:for\s+)?(?P<q>.+)", re.IGNORECASE),
        re.compile(r"\bwhat\s+do\s+my\s+obsidian\s+notes\s+say\s+about\s+(?P<q2>.+)", re.IGNORECASE),
    ]

    def __init__(self, doc_index) -> None:
        self._index = doc_index

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        query = (request.args.get("query") or gd.get("q") or gd.get("q2") or "").strip(" ?.!")
        if not query:
            return SkillResult("What should I look for in Obsidian?", success=False)
        if self._index is None:
            return SkillResult("Note search needs the embedding model, which is offline.",
                               success=False)
        vault = find_vault()
        hits = await asyncio.to_thread(self._index.search, query, 8)
        if vault is not None:  # prefer hits from inside the vault
            in_vault = [h for h in hits if str(vault).lower() in h["path"].lower()]
            hits = in_vault or hits
        if not hits:
            return SkillResult(f"Nothing in your notes mentions {query}.", success=False)
        top = hits[0]
        snippet = " ".join(top["text"].split())
        if len(snippet) > 260:
            snippet = snippet[:260].rsplit(" ", 1)[0] + "…"
        return SkillResult(f"From {Path(top['path']).name}: {snippet}",
                           data={"hits": hits[:4]})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string",
                                             "description": "topic to find in the notes"}},
                    "required": ["query"],
                },
            },
        }


class ObsidianOpenSkill(Skill):
    name = "obsidian_open"
    controls_pc = True
    description = "Open the Obsidian app (the user's note vault)."

    patterns = [re.compile(r"\b(?:open|start|launch)\s+obsidian\b", re.IGNORECASE)]

    async def execute(self, request: SkillRequest) -> SkillResult:
        """Launch the installed app; fall back to the obsidian:// URI.

        This used to go straight to the URI on the grounds that it works
        wherever the exe lives — but only if something registered the protocol,
        and a Store-less install may not. When it isn't registered Windows pops
        "Get an app to open this 'obsidian' link" while ``Popen`` returns
        happily, so MEDO announced success over a visible error dialog. The
        real executable is discoverable, so prefer it and keep the URI for the
        case where the app is installed somewhere discovery can't see.
        """
        from core.platform import open_path
        from skills.appfinder import find_app

        app = await asyncio.to_thread(find_app, "obsidian")
        if app is not None:
            try:
                await asyncio.to_thread(open_path, app.path)
                return SkillResult("Opening Obsidian.", data={"path": str(app.path)})
            except Exception:
                logger.warning("could not launch %s", app.path, exc_info=True)

        import subprocess

        try:
            subprocess.Popen(  # noqa: ASYNC220 - returns at once; never awaited
                ["cmd", "/c", "start", "", "obsidian://open"], shell=False)
            return SkillResult("Opening Obsidian.", data={"via": "uri"})
        except Exception:
            return SkillResult("I couldn't open Obsidian — is it installed?", success=False)


def setup(services: dict) -> list[Skill]:
    return [
        ObsidianNoteSkill(),
        ObsidianSearchSkill(services.get("doc_index")),
        ObsidianOpenSkill(),
    ]
