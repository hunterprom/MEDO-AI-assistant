"""Voice notes backed by SQLite (:class:`core.memory.NoteStore`).

    "take a note buy milk"      -> stored
    "read my notes"             -> reads them back
    "delete note 3"             -> removes one
"""

from __future__ import annotations

import re
from typing import Any

from core.memory import NoteStore
from skills.base import Skill, SkillRequest, SkillResult


class NotesSkill(Skill):
    name = "notes"
    description = "Take, read back, or delete voice notes."

    patterns = [
        re.compile(r"\b(?:take|make|write|jot(?:\s+down)?|add)\s+(?:a\s+)?note\b[:\s]*(?P<body>.*)", re.IGNORECASE),
        re.compile(r"\bnote\s+that\b\s*(?P<body2>.*)", re.IGNORECASE),
        re.compile(r"\b(?:read|list|show|what\s+are)\b.*\bnotes?\b", re.IGNORECASE),
        re.compile(r"\bdelete\s+note\s+(?P<id>\d+)\b", re.IGNORECASE),
    ]

    def __init__(self, store: NoteStore) -> None:
        self._store = store

    async def execute(self, request: SkillRequest) -> SkillResult:
        text = request.text
        m = request.match
        # LLM tool path: the model passes {action, text, id} the regex can't see.
        args = request.args or {}
        action = str(args.get("action") or "").strip().lower()

        # delete note N (regex group OR the LLM's id arg)
        del_id = m.group("id") if (m and m.groupdict().get("id")) else args.get("id")
        if action == "delete" or (del_id is not None and action != "add"):
            if del_id is None:
                return SkillResult("Which note number?", success=False)
            try:
                note_id = int(del_id)
            except (ValueError, TypeError):
                return SkillResult("Which note number?", success=False)
            ok = self._store.delete(note_id)
            return SkillResult(
                f"Deleted note {note_id}." if ok else f"There's no note {note_id}.",
                success=ok,
            )

        # read / list. Branch on which pattern actually matched, not on a
        # re-scan of the raw text: a dictated note whose BODY contains a
        # list-verb ("note that Bob will show up at 5") used to be read back
        # instead of stored, because "show" tripped the heuristic below.
        matched_add = bool(m and (m.groupdict().get("body") is not None
                                  or m.groupdict().get("body2") is not None))
        if action == "list" or (
                not matched_add
                and re.search(r"\b(read|list|show|what)\b", text, re.IGNORECASE)
                and "note" in text.lower()
                and not re.search(r"\b(take|make|write|jot|add)\b", text, re.IGNORECASE)):
            notes = self._store.list()
            if not notes:
                return SkillResult("You have no notes.")
            spoken = "; ".join(f"{n.id}: {n.text}" for n in notes[:10])
            return SkillResult(f"You have {len(notes)} notes. {spoken}.", data={"count": len(notes)})

        # take a note — body from the regex group OR the LLM's text arg
        body = ""
        if m:
            body = (m.groupdict().get("body") or m.groupdict().get("body2") or "").strip()
        if not body:
            body = str(args.get("text") or "").strip()
        if not body:
            return SkillResult("What should the note say?", success=False)
        note = self._store.add(body)
        return SkillResult(f"Noted. That's note {note.id}.", data={"id": note.id})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "list", "delete"]},
                        "text": {"type": "string", "description": "note body when adding"},
                        "id": {"type": "integer", "description": "note id when deleting"},
                    },
                    "required": ["action"],
                },
            },
        }
