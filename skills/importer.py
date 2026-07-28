"""Import a file or a picture so MEDO can answer questions about it later.

The gap: MEDO's document RAG only sees files that already live under a
whitelisted folder, and only notices them on a reindex. There was no way to say
"here, take this" about a specific file — and no way at all to hand it a
picture, because images carry no text to chunk.

Both land in the same place, deliberately:

* **Documents** (pdf/txt/md) are copied into the managed store and pushed
  through the existing chunk -> embed -> store pipeline
  (:meth:`DocumentIndex.index_file`), so the documents skill answers questions
  about them with no new retrieval path.
* **Images** have no text, so one is manufactured: the vision model describes
  the picture, and that description is written beside the image as a Markdown
  sidecar and indexed like any other document. "What was in that diagram I
  imported" is then an ordinary RAG query, not a special case.

Two rules the import obeys:

* The **source** path must be inside the configured whitelist. Importing is
  reading a file MEDO was not previously allowed to read, so the whitelist is
  exactly the check that matters.
* Imported files are **copied, never executed** — and the copy keeps its
  original suffix, so nothing here can turn a document into a program.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from core import mk
from core.config import Settings, expand_path
from core.safety import PathWhitelist
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

#: Documents the RAG pipeline can actually read. Mirrors docindex.INDEXED_EXTS
#: plus the formats extract_text handles.
DOCUMENT_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".rst", ".csv"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

#: A picture is described once, at import, and the description is what gets
#: indexed. Asking for structure pays off later: "the diagram with the 220 ohm
#: resistor" only finds anything if the description mentioned the resistor.
DESCRIBE_PROMPT = (
    "Describe this image in detail for someone who cannot see it and will "
    "search for it later by memory. Cover: what it is, every object or "
    "component you can identify, any text, labels, numbers or part markings "
    "you can read, and how things are arranged or connected. If it is a "
    "diagram, schematic or screenshot, describe its structure. Plain text."
)


def classify_suffix(path: Path) -> str:
    """"document", "image", or "" for something that cannot be imported."""
    suffix = path.suffix.lower()
    if suffix in DOCUMENT_SUFFIXES:
        return "document"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return ""


def unique_destination(directory: Path, name: str) -> Path:
    """A free path in the store for ``name``, never overwriting an existing file.

    Importing twice should give you two files, not silently replace the first —
    the second might be a different document that happens to share a name.
    """
    target = directory / name
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = directory / f"{stem} {stamp}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem} {stamp}-{counter}{suffix}"
        counter += 1
    return candidate


def sidecar_text(image_name: str, source: Path, description: str) -> str:
    """The Markdown written beside an imported image, and then indexed.

    The original path is included on purpose: it is the thing the user is most
    likely to remember and search for ("the screenshot from my downloads").
    """
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"# Imported image: {image_name}\n\n"
        f"- Imported: {stamp}\n"
        f"- Original location: {source}\n\n"
        f"## What MEDO saw\n\n{description.strip()}\n"
    )


class ImportFileSkill(Skill):
    """"import this file" / "внеси документ" — take a file into the store."""

    name = "import_file"
    controls_pc = True          # it reads and copies files on this machine
    description = (
        "Import a document or an image so MEDO can answer questions about it "
        "later. Documents are indexed for search; images are described by the "
        "vision model and the description is indexed. Takes a file path."
    )

    patterns = [
        re.compile(r"\bimport\s+(?:this\s+|the\s+)?(?:file|document|image|picture|photo)"
                   r"(?:\s+(?P<path>.+))?$", re.IGNORECASE),
        # ".+$" not "[^\s]+": a real path has spaces ("My Documents", "Program
        # Files"), and stopping at the first space imported "C:\Users\Me\My".
        re.compile(r"\bimport\s+(?P<path2>[A-Za-z]:[\\/].+|[~./].+)$", re.IGNORECASE),
        # "learn" only. "read this file" belongs to notes, and "remember this
        # file" to remember_fact — both are registered earlier and both are
        # right to claim their own verb. Anyone who wants those phrasings for
        # importing can add them under skills.triggers, which is what that
        # feature is for.
        re.compile(r"\blearn\s+this\s+"
                   r"(?:file|document|image|picture)(?:\s+(?P<path3>.+))?$", re.IGNORECASE),
        # MK: "внеси документ", "внеси ја сликата". "внеси" is unambiguous —
        # no other skill claims it — so no verb-stealing here.
        re.compile(r"\bвнеси\s+(?:(?:го|ја)\s+)?(?:документ(?:от)?|датотека(?:та)?|"
                   r"фајл(?:от)?|слика(?:та)?)(?:\s+(?P<pathm>.+))?$", re.IGNORECASE),
        re.compile(r"\bвнеси\s+(?P<pathm2>[A-Za-z]:[\\/].+|[~./].+)$",
                   re.IGNORECASE),
    ]

    def __init__(self, settings: Settings, whitelist: PathWhitelist,
                 doc_index=None, describe=None) -> None:
        self._settings = settings
        self._whitelist = whitelist
        self._index = doc_index
        #: async (image_b64: str, prompt: str) -> SkillResult — the same call
        #: the seeing skills make. Injected so tests need no Ollama.
        self._describe = describe or self._default_describe

    async def _default_describe(self, image_b64: str, prompt: str):
        """The vision model, via the seeing skills' own client.

        Importing an image and asking "what do you see" are the same operation
        with a different prompt, so this goes through vision_skill rather than
        opening a second path to Ollama.
        """
        from skills.vision_skill import _describe as describe_image

        return await describe_image(self._settings, image_b64, prompt)

    # -- store ----------------------------------------------------------------

    @property
    def store(self) -> Path:
        return expand_path(self._settings.memory.import_dir)

    def _resolve_source(self, spoken: str) -> tuple[Path | None, str | None]:
        """Spoken path -> a readable, whitelisted file, or (None, reason)."""
        raw = (spoken or "").strip().strip("\"'` ")
        if not raw:
            return None, "which file"
        try:
            path = expand_path(raw)
        except (OSError, RuntimeError, ValueError):
            return None, "not found"
        if not path.exists() or not path.is_file():
            return None, "not found"
        # The whitelist is the whole point: importing is reading a file MEDO
        # was not otherwise allowed to open.
        if not self._whitelist.is_allowed(path):
            return None, "outside"
        if not classify_suffix(path):
            return None, "unsupported"
        return path, None

    def _refusal(self, reason: str, raw: str, speak_mk: bool) -> SkillResult:
        messages = {
            "which file": ("Која датотека да внесам?", "Which file should I import?"),
            "not found": (f"Не најдов „{raw}“.", f"I couldn't find {raw!r}."),
            "outside": (
                "Таа датотека е надвор од дозволените папки.",
                "That file is outside the folders I'm allowed to read. Add its "
                "folder to safety.whitelist_dirs if you want me to import it."),
            "unsupported": (
                f"Не можам да внесам {raw} — поддржувам документи и слики.",
                f"I can't import {raw} — I handle documents and images."),
        }
        mk_msg, en_msg = messages[reason]
        return SkillResult(mk_msg if speak_mk else en_msg, success=False,
                           data={"imported": False, "reason": reason})

    # -- execution ------------------------------------------------------------

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        raw = (request.args.get("path") or gd.get("path") or gd.get("path2")
               or gd.get("path3") or gd.get("pathm") or gd.get("pathm2") or "")
        source, problem = self._resolve_source(raw)
        if problem is not None:
            return self._refusal(problem, raw.strip() or "that", speak_mk)

        try:
            store = self.store
            await asyncio.to_thread(store.mkdir, parents=True, exist_ok=True)
            destination = unique_destination(store, source.name)
            # copy2, never move and never execute: the original stays where it
            # is, and the copy keeps its suffix so a document cannot arrive as
            # something runnable.
            await asyncio.to_thread(shutil.copy2, source, destination)
        except OSError as exc:
            logger.warning("import copy failed: %s", exc)
            return SkillResult(
                f"Не успеав да ја копирам {source.name}." if speak_mk
                else f"I couldn't copy {source.name} into my store.",
                success=False, data={"imported": False})

        kind = classify_suffix(source)
        if kind == "image":
            return await self._import_image(source, destination, speak_mk)
        return await self._import_document(source, destination, speak_mk)

    async def _import_document(self, source: Path, destination: Path,
                               speak_mk: bool) -> SkillResult:
        chunks = 0
        if self._index is not None:
            try:
                chunks = await asyncio.to_thread(self._index.index_file, destination)
            except Exception:
                logger.warning("indexing the import failed", exc_info=True)
        data = {"imported": True, "kind": "document",
                "path": str(destination), "chunks": chunks}
        if not chunks:
            # Copied but not searchable — say so rather than implying it is
            # ready to answer questions.
            return SkillResult(
                f"Внесов {source.name}, но не можев да ја индексирам."
                if speak_mk else
                f"I've imported {source.name}, but couldn't index it — ask me "
                f"about it once the embedding model is available.", data=data)
        return SkillResult(
            f"Внесов {source.name}. Прашај ме за неа." if speak_mk
            else f"Imported {source.name}. Ask me anything about it.", data=data)

    async def _import_image(self, source: Path, destination: Path,
                            speak_mk: bool) -> SkillResult:
        from skills.vision_skill import _shrink

        description = ""
        try:
            raw = await asyncio.to_thread(destination.read_bytes)
            # Same downscale the screen path uses — a 12 MP phone photo would
            # otherwise blow the vision model's timeout.
            small = await asyncio.to_thread(_shrink, raw)
            result = await self._describe(base64.b64encode(small).decode(),
                                          DESCRIBE_PROMPT)
            if getattr(result, "success", True):
                description = (getattr(result, "speech", "") or "").strip()
        except Exception:
            logger.warning("describing the import failed", exc_info=True)
        if not description:
            return SkillResult(
                f"Внесов {source.name}, но не можев да ја опишам."
                if speak_mk else
                f"I've imported {source.name}, but couldn't describe it — my "
                f"vision model didn't answer.",
                data={"imported": True, "kind": "image",
                      "path": str(destination), "described": False})

        # The description IS the searchable document. Writing it beside the
        # image and indexing it means image recall needs no new retrieval path.
        sidecar = destination.with_suffix(destination.suffix + ".md")
        chunks = 0
        try:
            await asyncio.to_thread(
                sidecar.write_text,
                sidecar_text(source.name, source, description), encoding="utf-8")
            if self._index is not None:
                chunks = await asyncio.to_thread(self._index.index_file, sidecar)
        except Exception:
            logger.warning("storing the image description failed", exc_info=True)

        spoken = description if len(description) <= 240 else description[:240] + "…"
        return SkillResult(
            f"Внесов {source.name}. {spoken}" if speak_mk
            else f"Imported {source.name}. {spoken}",
            data={"imported": True, "kind": "image", "path": str(destination),
                  "sidecar": str(sidecar), "description": description,
                  "chunks": chunks, "described": True})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "Full path of the file to import."},
                    },
                    "required": ["path"],
                },
            },
        }
