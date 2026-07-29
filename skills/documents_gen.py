"""Create documents and presentations — MEDO writes the content, saves the file.

"Make a report about our Q3 results" / "create a presentation on solar batteries"
→ MEDO composes the content as Markdown with its language model, renders it to the
format that fits the ask (Word .docx, PowerPoint .pptx, or a self-contained web
page / slide deck — see core/documents.py), saves it under ~/Documents/MEDO, and
opens it. Degrades to the dependency-free formats when the office libs are absent,
and declines cleanly when there's no model to write the content.
"""

from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path

from core import documents
from core.platform import open_path
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

_KINDS = (
    r"presentation|slide\s*deck|slideshow|slides|power\s*point|powerpoint|pptx|deck|"
    r"spread\s*sheet|spreadsheet|worksheet|excel|xlsx|table|"
    r"word\s+document|word\s+doc|document|report|essay|letter|memo|article|"
    r"web\s*page|html\s+page|doc")
_MAKE = re.compile(
    r"\b(?:make|create|write|generate|draft|produce|build|prepare)\s+(?:me\s+)?"
    r"(?:a|an|the)\s+"
    r"(?:[\w-]+\s+){0,4}?"                      # optional "five-page", "short", …
    r"(?P<kind>" + _KINDS + r")\b"
    r"(?:\s+(?:about|on|for|of|titled|called|regarding|covering|listing|showing)"
    r"\s+(?P<topic>.+))?",
    re.IGNORECASE)
#: A length hint anywhere in the request, so "five-page" / "detailed" shapes the
#: content (the renderer doesn't paginate, but the model writes more/less).
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_PAGES = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
                    r"[-\s]?pages?\b", re.IGNORECASE)
_LENGTH_WORD = re.compile(
    r"\b(short|brief|quick|long|detailed|comprehensive|in-depth|thorough)\b",
    re.IGNORECASE)
_TOPIC_TAIL = re.compile(
    r"\b(?:about|on|for|of|titled|called|regarding|covering|listing|showing)"
    r"\s+(.+)", re.IGNORECASE)
_PRES_WORDS = ("presentation", "slide", "deck", "slideshow", "power point",
               "powerpoint", "pptx")
_SHEET_WORDS = ("spreadsheet", "spread sheet", "excel", "xlsx", "worksheet",
                "table")
_TRAILING_FORMAT = re.compile(
    r"\s+(?:as|in|to)\s+(?:an?\s+)?(?:pdf|html|web\s*page|webpage|"
    r"word(?:\s+doc(?:ument)?)?|power\s*point|powerpoint|pptx|docx|xlsx|excel|"
    r"spread\s*sheet|slides?|deck|presentation|document)\s*$", re.IGNORECASE)


def _slug(text: str, limit: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:limit].rstrip("-")) or "document"


def _title_from(topic: str) -> str:
    t = topic.strip().rstrip(".?! ")
    return t[:1].upper() + t[1:] if t else "Document"


def _ext(fmt: str) -> str:
    return "html" if fmt == "slides" else fmt


def _length_hint(text: str) -> str:
    """A sentence telling the model how much to write, from "five-page" / "short"."""
    m = _PAGES.search(text)
    if m:
        n = _NUM_WORDS.get(m.group(1).lower())
        if n is None:
            with contextlib.suppress(ValueError):
                n = int(m.group(1))
        if n:
            return (f"Aim for roughly {n} page(s) of substance — enough sections "
                    "and detail to fill it.")
    w = _LENGTH_WORD.search(text)
    if w:
        if w.group(1).lower() in ("short", "brief", "quick"):
            return "Keep it concise — about a page."
        return "Make it detailed and thorough — several pages of substance."
    return ""


class MakeDocumentSkill(Skill):
    name = "make_document"
    description = (
        "Create a document or presentation about a topic — Word (.docx), "
        "PowerPoint (.pptx), or a self-contained web page — MEDO writes the "
        "content and saves the file.")
    patterns = [_MAKE]

    def __init__(self, compose=None, out_dir: str | Path | None = None) -> None:
        # compose: async (instruction: str) -> markdown. None => no model wired.
        self._compose = compose
        self._out_dir = Path(out_dir) if out_dir else Path.home() / "Documents" / "MEDO"

    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string",
                                  "description": "what the document/presentation is about"},
                        "kind": {"type": "string", "enum": ["document", "presentation"],
                                 "description": "a prose document or a slide deck"},
                        "format": {"type": "string", "enum": list(documents.ALL_FORMATS),
                                   "description": "output file format (optional)"},
                    },
                    "required": ["topic"],
                },
            },
        }

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        args = request.args or {}
        topic = str(args.get("topic") or gd.get("topic") or "").strip().strip("?.!")
        if not topic:
            m = _TOPIC_TAIL.search(request.text)
            topic = m.group(1).strip(" ?.!") if m else ""
        # A trailing "…as a PDF" / "…in word" names the FORMAT, not the subject —
        # keep it out of the title/content (format detection still reads the full
        # text below).
        topic = _TRAILING_FORMAT.sub("", topic).strip()
        if not topic:
            return SkillResult("What should it be about?", success=False)

        kind_word = str(args.get("kind") or gd.get("kind") or "").lower()
        low = request.text.lower()
        # Sniff the KIND phrase only — not the whole utterance — so a subject that
        # happens to contain "table"/"deck"/"slide" doesn't override the ask.
        if any(w in kind_word for w in _SHEET_WORDS):
            kind = "spreadsheet"
        elif (args.get("kind") == "presentation"
              or any(w in kind_word for w in _PRES_WORDS)):
            kind = "presentation"
        else:
            kind = "document"
        fmt = self._choose_format(low, args.get("format"), kind)
        noun = kind

        if self._compose is None:
            return SkillResult(
                "I need my language model to write the content, and it's offline "
                "right now.", success=False)
        try:
            markdown = await self._compose(
                self._instruction(topic, kind, _length_hint(request.text)))
        except Exception:                       # a model failure must not crash the turn
            logger.exception("make_document: compose failed for %r", topic)
            markdown = ""
        if not markdown.strip():
            return SkillResult(
                f"I couldn't write the {noun} — my model didn't return any content.",
                success=False)

        title = _title_from(topic)
        out = self._out_dir / f"{_slug(title)}.{_ext(fmt)}"
        try:
            path = documents.write(markdown, out, fmt, title=title)
        except documents.DocumentError as exc:
            return SkillResult(f"I couldn't save that: {exc}", success=False)
        with contextlib.suppress(Exception):     # opening is a nicety, not the job
            open_path(str(path))
        logger.info("make_document: wrote %s (%s)", path, fmt)
        return SkillResult(
            f"I've made a {noun} about {topic} and saved it as {path.name}.",
            data={"path": str(path), "format": fmt, "kind": noun})

    # -- helpers ------------------------------------------------------------

    def _choose_format(self, text: str, explicit, kind: str) -> str:
        usable = documents.available_formats()
        fallback = {"spreadsheet": "md", "presentation": "slides"}.get(kind, "html")
        if explicit and str(explicit).lower() in documents.ALL_FORMATS:
            fmt = str(explicit).lower()
            return fmt if fmt in usable else fallback
        if kind == "spreadsheet" or any(
                w in text for w in ("excel", "xlsx", "spreadsheet")):
            return "xlsx" if "xlsx" in usable else "md"
        if "pdf" in text:
            # no PDF renderer bundled — a web page the user can print to PDF.
            return "slides" if kind == "presentation" else "html"
        if any(w in text for w in ("powerpoint", "power point", "pptx", "keynote")):
            return "pptx" if "pptx" in usable else "slides"
        if "word" in text or ".docx" in text:
            return "docx" if "docx" in usable else "html"
        if "web page" in text or "webpage" in text or "html" in text:
            return "slides" if kind == "presentation" else "html"
        if kind == "presentation":
            return "pptx" if "pptx" in usable else "slides"
        return "docx" if "docx" in usable else "html"

    def _instruction(self, topic: str, kind: str, length: str = "") -> str:
        tail = (" " + length) if length else ""
        if kind == "spreadsheet":
            return (
                f"Produce a data table for {topic} as a GitHub-style Markdown "
                "table: a header row of column names, then one row per item, with "
                "a |---|---| separator after the header. Keep it concise and "
                "factual. Output ONLY the table — no prose, no code fences.")
        if kind == "presentation":
            return (
                f"Create the content for a slide presentation about {topic}. "
                "Use '## ' for each slide's title and '- ' bullets for the points "
                f"on that slide.{tail or ' Aim for 6 to 8 slides with short, punchy bullets.'} "
                "Output only Markdown — no code fences, no preamble.")
        return (
            f"Write a well-structured document about {topic}. Use '## ' section "
            "headings, short paragraphs, and '- ' bullet lists where useful."
            f"{tail} Do not include a top-level title line (it's added "
            "separately). Output only Markdown — no code fences, no preamble.")
