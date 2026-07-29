"""Edit files by voice — dictate into them, replace text, or open them to edit.

Three shapes of "edit a file", because they're genuinely different asks:

* **Dictate** — "add to shopping.txt: milk and eggs". Additive, so it just runs.
* **Replace** — "in config.yaml change 8710 to 9000". Mutates existing content,
  so it reports what it will do and waits for a spoken yes.
* **Open to edit** — "edit notes.md" launches the configured editor on the file,
  which is the right answer when the change is easier done by hand than dictated.

Every path is checked against :class:`~core.safety.PathWhitelist`, the same
whitelist the file *search* skill uses, so MEDO can only touch directories you
listed in config.yaml. Two further guards, because writes are unlike anything
else here — they destroy information:

* Only text files. Binary content is refused rather than corrupted.
* Every write leaves a ``<name>.bak`` beside the file first, so any edit —
  including one you asked for and regretted — is one file-copy from undone.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from core import mk
from core.platform import pick_for_os, run_detached
from core.safety import PathWhitelist
from skills.base import Skill, SkillRequest, SkillResult
from skills.files import search_files

logger = logging.getLogger(__name__)

#: Extensions we're willing to rewrite. An allowlist, not a blocklist: the cost
#: of wrongly editing a .docx or .jpg (silent corruption) is far worse than the
#: cost of refusing one.
TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".htm", ".css", ".scss",
    ".xml", ".sql", ".sh", ".bat", ".ps1", ".c", ".h", ".cpp", ".java",
    ".gitignore", ".properties",
}

#: How a spoken file name can look. Deliberately narrow — no slashes, so a
#: pattern can never capture a path fragment and walk out of the whitelist.
_NAME = r"[\w][\w .()-]*"

#: A trailing "file"/"document" noun the greedy name group swallows from the
#: natural "the config FILE" phrasing — "config file" won't substring-match
#: "config.yaml". Strip it (but "makefile" has no space, so it's untouched).
_TRAILING_FILE = re.compile(r"\s+(?:file|files|document|doc)\s*$", re.IGNORECASE)


def _drop_file_noun(name: str) -> str:
    """'the config file' -> 'config'; never reduces the name to nothing."""
    return _TRAILING_FILE.sub("", name).strip() or name


def is_text_file(path: Path) -> bool:
    """True when this file is safe to rewrite as text.

    Suffix first (cheap and explicit), then an actual decode of the head — a
    ``.log`` full of binary noise should still be refused.
    """
    # A bare dotfile (".env", ".gitignore") has an EMPTY suffix, so also match
    # the whole name — otherwise those two entries in the set are dead.
    if path.suffix.lower() not in TEXT_SUFFIXES \
            and path.name.lower() not in TEXT_SUFFIXES:
        return False
    try:
        with path.open("rb") as fh:
            head = fh.read(8192)
    except OSError:
        return False
    if b"\x00" in head:            # NUL byte: not text, whatever the extension
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def backup(path: Path) -> Path | None:
    """Copy ``path`` to ``<name>.bak`` before it's written. Best effort."""
    target = path.with_suffix(path.suffix + ".bak")
    try:
        shutil.copy2(path, target)
        return target
    except OSError:
        logger.warning("could not back up %s", path)
        return None


def apply_replace(content: str, old: str, new: str) -> tuple[str, int]:
    """Replace ``old`` with ``new``; returns the new content and the hit count.

    Case-insensitive only as a fallback: an exact match is always preferred, so
    dictated case never silently rewrites the wrong thing when the exact string
    is present. Pure — the risky logic is testable without touching disk.
    """
    if old in content:
        return content.replace(old, new), content.count(old)
    pattern = re.compile(re.escape(old), re.IGNORECASE)
    found = len(pattern.findall(content))
    return (pattern.sub(new.replace("\\", "\\\\"), content), found) if found \
        else (content, 0)


class FileEditSkill(Skill):
    """Dictate into a file, or replace text inside it."""

    name = "edit_file"
    controls_pc = True
    description = (
        "Edit a text file in the user's whitelisted folders: append dictated "
        "text to it, or replace one string with another. Use when the user "
        "wants a file's CONTENT changed."
    )

    patterns = [
        # --- replace (destructive; asks first) ---------------------------
        # "in config.yaml replace 8710 with 9000"
        re.compile(rf"\bin\s+(?:the\s+|my\s+)?(?:file\s+)?(?P<rfile>{_NAME}?)[,\s]+"
                   r"(?:replace|change|swap)\s+(?P<old>.+?)\s+(?:with|to|for)\s+"
                   r"(?P<new>.+?)\s*$", re.IGNORECASE),
        # "replace 8710 with 9000 in config.yaml"
        re.compile(rf"\b(?:replace|change|swap)\s+(?P<old>.+?)\s+(?:with|to|for)\s+"
                   rf"(?P<new>.+?)\s+in\s+(?:the\s+|my\s+)?(?:file\s+)?(?P<rfile>{_NAME})\s*$",
                   re.IGNORECASE),
        # MK: "во config.yaml замени 8710 со 9000"
        re.compile(rf"\bво\s+(?:датотеката\s+|фајлот\s+)?(?P<rfile>{_NAME}?)[,\s]+"
                   r"замени\s+(?P<old>.+?)\s+со\s+(?P<new>.+?)\s*$", re.IGNORECASE),

        # --- dictate to the top ------------------------------------------
        re.compile(rf"\b(?:prepend|add)\s+(?P<text>.+?)\s+to\s+the\s+"
                   rf"(?:top|start|beginning)\s+of\s+(?:the\s+|my\s+)?(?:file\s+)?"
                   rf"(?P<pfile>{_NAME})\s*$", re.IGNORECASE),

        # --- dictate (append) --------------------------------------------
        # "add to shopping.txt: milk" — colon form, the natural dictation shape
        re.compile(rf"\b(?:add|append|write|save)\s+to\s+(?:the\s+|my\s+)?(?:file\s+)?"
                   rf"(?P<afile>{_NAME}?)\s*[:,]\s*(?P<text>.+)$", re.IGNORECASE),
        # "append milk to shopping.txt"
        re.compile(rf"\b(?:add|append|write|save)\s+(?P<text>.+?)\s+to\s+"
                   rf"(?:the\s+(?:end\s+of\s+)?|my\s+)?(?:file\s+)?(?P<afile>{_NAME})\s*$",
                   re.IGNORECASE),
        # MK: "додај во листа.txt: млеко" / "запиши млеко во листа.txt"
        re.compile(rf"\b(?:додај|запиши|напиши)\s+во\s+(?:датотеката\s+|фајлот\s+)?"
                   rf"(?P<afile>{_NAME}?)\s*[:,]\s*(?P<text>.+)$", re.IGNORECASE),
        re.compile(rf"\b(?:додај|запиши|напиши)\s+(?P<text>.+?)\s+во\s+"
                   rf"(?:датотеката\s+|фајлот\s+)?(?P<afile>{_NAME})\s*$",
                   re.IGNORECASE),
    ]

    def __init__(self, whitelist: PathWhitelist) -> None:
        self._whitelist = whitelist

    # -- resolution -----------------------------------------------------------

    def _resolve(self, name: str) -> tuple[Path | None, str | None]:
        """Spoken name -> a writable text file, or (None, spoken reason)."""
        name = _drop_file_noun(name.strip().strip(" .,:;\"'"))
        if not name:
            return None, "which file"
        if not self._whitelist.roots:
            return None, "no whitelist"
        hits = search_files(self._whitelist, name, limit=8)
        if not hits:
            return None, "not found"
        # Prefer an exact filename match over a substring one, then the
        # shortest name — "notes.md" should beat "notes-archive-2019.md".
        exact = [h for h in hits if h.name.lower() == name.lower()]
        best = min(exact or hits, key=lambda p: len(p.name))
        if not self._whitelist.is_allowed(best):
            return None, "outside"
        if not is_text_file(best):
            return None, "not text"
        return best, None

    def _refusal(self, reason: str, name: str, speak_mk: bool) -> SkillResult:
        messages = {
            "which file": ("Која датотека?", "Which file?"),
            "no whitelist": (
                "Нема дозволени папки, па не можам да менувам датотеки.",
                "No folders are whitelisted, so I can't edit any files."),
            "not found": (
                f"Не најдов датотека '{name}'.",
                f"I couldn't find a file called '{name}'."),
            "outside": ("Таа датотека е надвор од дозволените папки.",
                        "That file is outside the allowed folders."),
            "not text": (
                f"'{name}' не е текстуална датотека, па не ја менувам.",
                f"'{name}' isn't a text file, so I won't edit it."),
        }
        mk_msg, en_msg = messages[reason]
        return SkillResult(mk_msg if speak_mk else en_msg, success=False)

    # -- execution ------------------------------------------------------------

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        args = request.args
        speak_mk = mk.is_cyrillic(request.text)

        action = (args.get("action") or "").lower()
        name = (args.get("file") or gd.get("rfile") or gd.get("afile")
                or gd.get("pfile") or "")
        if not action:
            if gd.get("rfile"):
                action = "replace"
            elif gd.get("pfile"):
                action = "prepend"
            else:
                action = "append"

        # _resolve walks the whitelist (rglob) — thread it so a slow disk isn't
        # an audible gap in MEDO's voice.
        target, problem = await asyncio.to_thread(self._resolve, name)
        if problem is not None:
            return self._refusal(problem, name, speak_mk)

        if action == "replace":
            return await self._replace(
                target, args.get("old") or gd.get("old") or "",
                args.get("new") or gd.get("new") or "",
                request.context.get("confirmed", False), speak_mk)
        text = (args.get("text") or gd.get("text") or "").strip()
        if not text:
            return SkillResult("Што да запишам?" if speak_mk
                               else "What should I write?", success=False)
        return await self._dictate(target, text, action == "prepend", speak_mk)

    async def _dictate(self, path: Path, text: str, to_top: bool,
                       speak_mk: bool) -> SkillResult:
        def _write() -> None:
            # Read, back up and write in ONE worker hop. The event loop also
            # drives TTS streaming and the HUD's event feed, so a slow disk
            # here is an audible gap in MEDO's voice.
            content = path.read_text(encoding="utf-8")
            backup(path)
            if to_top:
                new_content = f"{text}\n{content}"
            else:
                sep = "" if not content or content.endswith("\n") else "\n"
                new_content = f"{content}{sep}{text}\n"
            path.write_text(new_content, encoding="utf-8")

        try:
            await asyncio.to_thread(_write)
        except OSError as exc:
            logger.warning("dictate failed on %s: %s", path, exc)
            return SkillResult(
                f"Не успеав да запишам во {path.name}." if speak_mk
                else f"I couldn't write to {path.name}.", success=False)
        where_mk, where_en = ("на почетокот на", "to the top of") if to_top \
            else ("во", "to")
        return SkillResult(
            f"Запишав '{text}' {where_mk} {path.name}." if speak_mk
            else f"Added '{text}' {where_en} {path.name}.",
            data={"path": str(path), "action": "prepend" if to_top else "append"})

    async def _replace(self, path: Path, old: str, new: str, confirmed: bool,
                       speak_mk: bool) -> SkillResult:
        old, new = old.strip().strip("\"'"), new.strip().strip("\"'")
        if not old:
            return SkillResult("Што да заменам?" if speak_mk
                               else "What should I replace?", success=False)
        try:
            content = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except OSError:
            return SkillResult(
                f"Не можам да ја прочитам {path.name}." if speak_mk
                else f"I can't read {path.name}.", success=False)

        updated, count = apply_replace(content, old, new)
        if count == 0:
            return SkillResult(
                f"Не најдов '{old}' во {path.name}." if speak_mk
                else f"I couldn't find '{old}' in {path.name}.", success=False)

        if not confirmed:
            # Say exactly what will change before changing it — this is the one
            # action here that destroys information.
            times_mk = "пат" if count == 1 else "пати"
            return SkillResult(
                f"Ќе заменам '{old}' со '{new}' {count} {times_mk} во "
                f"{path.name}. Да продолжам?" if speak_mk else
                f"That replaces '{old}' with '{new}' {count} "
                f"time{'s' if count != 1 else ''} in {path.name}. Shall I?",
                needs_confirmation=True,
                data={"path": str(path), "count": count})

        saved = await asyncio.to_thread(backup, path)
        try:
            await asyncio.to_thread(path.write_text, updated, encoding="utf-8")
        except OSError:
            return SkillResult(
                f"Не успеав да запишам во {path.name}." if speak_mk
                else f"I couldn't write to {path.name}.", success=False)
        note_mk = f" Оригиналот е во {saved.name}." if saved else ""
        note_en = f" The original is in {saved.name}." if saved else ""
        return SkillResult(
            f"Заменив {count} во {path.name}.{note_mk}" if speak_mk
            else f"Replaced {count} in {path.name}.{note_en}",
            data={"path": str(path), "count": count,
                  "backup": str(saved) if saved else None})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string",
                                   "enum": ["append", "prepend", "replace"]},
                        "file": {"type": "string",
                                 "description": "File name, e.g. 'notes.md'."},
                        "text": {"type": "string",
                                 "description": "Text to add (append/prepend)."},
                        "old": {"type": "string", "description": "Text to replace."},
                        "new": {"type": "string", "description": "Its replacement."},
                    },
                    "required": ["action", "file"],
                },
            },
        }


class WriteInAppSkill(Skill):
    """"write this in notepad" — open an app and type into it.

    The gap this closes: ``edit_file`` needs a file that already exists, and
    ``type_text`` types into whatever happens to be focused. Neither answers
    "put this in Notepad", which means *launch it, then type*. The app must be
    one from the config table, so this can't be talked into launching anything
    arbitrary.
    """

    name = "write_in_app"
    controls_pc = True
    description = (
        "Open an application (Notepad, the editor) and type text into it. Use "
        "when the user wants text written INTO a program rather than a file."
    )

    #: Seconds to wait for the window before typing. Too short and the first
    #: characters land on the desktop instead of in the app.
    LAUNCH_WAIT_S = 1.8

    def __init__(self, apps_table: dict[str, dict[str, str]] | None = None) -> None:
        self._apps = apps_table or {}
        names = sorted({*self._apps.keys(), *_APP_ALIASES.keys()},
                       key=len, reverse=True)
        alt = "|".join(re.escape(n) for n in names) if names else r"(?!x)x"
        self._alt = alt
        self.patterns = [
            # "write this in notepad" / "type the shopping list into notepad"
            re.compile(rf"\b(?:write|type|put|jot)\s+(?P<text>.+?)\s+"
                       rf"(?:in|into|on)\s+(?:the\s+|a\s+)?(?P<app>{alt})\b",
                       re.IGNORECASE),
            # MK: "напиши го ова во нотепад"
            re.compile(rf"\b(?:напиши|искуцај|запиши)\s+(?:го\s+|ја\s+)?"
                       rf"(?P<text>.+?)\s+во\s+(?P<app>{alt})\b", re.IGNORECASE),
        ]

    #: Asking MEDO to COMPOSE something ("open Notepad and summarize the
    #: Odyssey, type it out") can't be served by a literal regex capture — the
    #: group would grab "it out, the Odysseus" and type THAT. Decline the fast
    #: path for these so the brain writes the text first and calls this skill as
    #: a tool with the finished prose.
    _NEEDS_COMPOSING = re.compile(
        r"\b(?:summari[sz]e|summary|explain|describe|translate|compose|draft|"
        r"rewrite|paraphrase|tell\s+me\s+about|write\s+(?:me\s+)?an?\b)",
        re.IGNORECASE)

    def match(self, text: str):
        if self._NEEDS_COMPOSING.search(text):
            return None          # let the LLM compose, then call us as a tool
        return super().match(text)

    def _resolve(self, app: str) -> str | None:
        app = app.lower().strip()
        key = app if app in self._apps else _APP_ALIASES.get(app)
        return key if key in self._apps else None

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        app = (request.args.get("app") or gd.get("app") or "").strip()
        text = (request.args.get("text") or gd.get("text") or "").strip()
        key = self._resolve(app)
        if key is None:
            return SkillResult(
                f"Немам конфигурирано {app}." if speak_mk
                else f"I don't have {app} configured.", success=False)
        # "write THE PROMPT in notepad" — a reference, not the words to type.
        if not text or text.lower() in _VAGUE:
            return SkillResult(
                f"Што точно да напишам во {key}?" if speak_mk
                else f"What exactly should I write in {key}?", success=False)

        command = pick_for_os(self._apps[key])
        if not command:
            return SkillResult(
                f"{key} не е поставен за овој систем." if speak_mk
                else f"{key} isn't set up for this OS.", success=False)
        run_detached(command)
        await asyncio.sleep(self.LAUNCH_WAIT_S)   # let the window take focus
        try:
            from skills.desktop import type_into_focused

            await asyncio.to_thread(type_into_focused, text)
        except Exception:
            logger.exception("typing into %s failed", key)
            return SkillResult(
                f"Го отворив {key}, но не успеав да напишам." if speak_mk
                else f"I opened {key} but couldn't type into it.", success=False)
        return SkillResult(
            f"Напишав во {key}." if speak_mk else f"Written in {key}.",
            data={"app": key, "chars": len(text)})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "enum": sorted(self._apps.keys())},
                        "text": {"type": "string",
                                 "description": "The exact text to type."},
                    },
                    "required": ["app", "text"],
                },
            },
        }


#: Spoken app names -> config keys, for the write-into-an-app skill.
_APP_ALIASES = {
    "notepad": "notepad", "note pad": "notepad", "нотепад": "notepad",
    "text editor": "editor", "the editor": "editor", "уредувач": "editor",
    "vs code": "editor", "vscode": "editor", "code": "editor",
}

#: Placeholders that mean "the thing we were talking about", not literal text.
#: Typing the words "the prompt" into Notepad is never what was wanted.
_VAGUE = {
    "this", "that", "it", "the prompt", "prompt", "the text", "the same",
    "ова", "тоа", "истото",
}


class OpenInEditorSkill(Skill):
    """"edit notes.md" — open the file in the configured editor.

    The third shape of "edit a file": some changes are simply faster by hand.
    Launches ``skills.apps.editor`` from config.yaml on the resolved path,
    rather than driving the keyboard blind.
    """

    name = "open_in_editor"
    controls_pc = True
    description = (
        "Open a file in the user's text editor so they can change it by hand."
    )

    patterns = [
        re.compile(rf"\bedit\s+(?:the\s+|my\s+)?(?:file\s+)?(?P<file>{_NAME})\s*$",
                   re.IGNORECASE),
        re.compile(rf"\bopen\s+(?:the\s+|my\s+)?(?:file\s+)?(?P<file>{_NAME}?)\s+in\s+"
                   r"(?:the\s+|my\s+)?(?:editor|text\s+editor|vs\s?code|notepad)\b",
                   re.IGNORECASE),
        # MK: "уреди белешки.md", "отвори белешки.md во уредувач"
        re.compile(rf"\bуреди\s+(?:ја\s+)?(?:датотеката\s+)?(?P<file>{_NAME})\s*$",
                   re.IGNORECASE),
        re.compile(rf"\bотвори\s+(?:ја\s+)?(?:датотеката\s+)?(?P<file>{_NAME}?)\s+во\s+"
                   r"(?:уредувач(?:от)?|едитор(?:от)?)\b", re.IGNORECASE),
    ]

    def __init__(self, whitelist: PathWhitelist,
                 apps_table: dict[str, dict[str, str]] | None = None) -> None:
        self._whitelist = whitelist
        self._apps = apps_table or {}

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        name = _drop_file_noun(
            (request.args.get("file") or gd.get("file") or "").strip(" .,:;\"'"))
        if not name:
            return SkillResult("Која датотека?" if speak_mk else "Which file?",
                               success=False)
        hits = await asyncio.to_thread(search_files, self._whitelist, name, limit=8)
        hits = [h for h in hits if self._whitelist.is_allowed(h)]
        if not hits:
            return SkillResult(
                f"Не најдов датотека '{name}'." if speak_mk
                else f"I couldn't find a file called '{name}'.", success=False)
        exact = [h for h in hits if h.name.lower() == name.lower()]
        target = min(exact or hits, key=lambda p: len(p.name))

        command = pick_for_os(self._apps.get("editor", {}))
        if not command:
            return SkillResult(
                "Немам поставен уредувач во конфигурацијата." if speak_mk
                else "I don't have an editor configured.", success=False)
        run_detached(f'{command} "{target}"')
        return SkillResult(
            f"Отворам {target.name} во уредувачот." if speak_mk
            else f"Opening {target.name} in your editor.",
            data={"path": str(target)})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "File to open."},
                    },
                    "required": ["file"],
                },
            },
        }
