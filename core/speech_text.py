"""Make model output safe to say out loud.

MEDO's replies are spoken, so anything that is punctuation-as-structure —
markdown, a code fence, a JSON blob, a bare tool name — is noise at best and
gibberish at worst. The system prompt forbids all of it, and a small local model
still produces it: asked what feature it would like, MEDO read a function schema
aloud, quotes, braces, ``"type": "function"`` and all.

Two jobs, deliberately separate:

* :func:`strip_markup` always runs — it is the last thing between a reply and
  the speakers, and it never fails or throws text away that carries meaning.
* :func:`looks_like_markup` and :func:`leaked_tool_names` are *diagnoses*. They
  tell the router the model ignored its instructions, so it can re-ask and get a
  real spoken answer rather than voice the stripped remains of a code block.
"""

from __future__ import annotations

import re

#: A fenced block, with or without a language tag, closed or running to the end
#: of the reply (a truncated stream leaves the fence open).
_FENCE = re.compile(r"```[^\n]*\n?.*?(?:```|\Z)", re.DOTALL)
#: `inline code` — the ticks go, the words stay.
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
#: Leading heading hashes, blockquote markers, and list bullets.
_LINE_PREFIX = re.compile(r"^[ \t]*(?:#{1,6}\s+|>\s+|[-*+]\s+|\d+[.)]\s+)",
                          re.MULTILINE)
#: **bold**, __bold__, *italic*, _italic_ — markers only, never the words.
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
#: [label](https://…) -> label
_LINK = re.compile(r"\[([^\]\n]+)\]\([^)\s]*\)")
#: A bullet marker that survived line-joining ("Would you like: * meditation?
#: * a story?"). Digits either side are spared so "2 * 3" keeps its arithmetic.
_INLINE_BULLET = re.compile(r"(?<!\d)\s+[*•]\s+(?!\d)")
#: Backticks with no partner — a fence the stream cut in half.
_STRAY_TICKS = re.compile(r"`+")
#: A JSON-ish object: a brace, then a quoted key with a colon, anywhere inside.
_JSON_BLOB = re.compile(r"\{[^{}]*\"[\w-]+\"\s*:", re.DOTALL)
#: snake_case words — how every MEDO tool is named (open_website, locate_in_app).
_SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def strip_markup(text: str) -> str:
    """Return ``text`` with anything unspeakable removed.

    Fenced blocks go entirely — a code listing has no spoken form, and reading
    the fence contents is what produced the schema-out-loud failure. Everything
    else keeps its words and loses only its markers.
    """
    if not text:
        return ""
    out = _FENCE.sub(" ", text)
    out = _LINK.sub(r"\1", out)
    out = _INLINE_CODE.sub(r"\1", out)
    out = _LINE_PREFIX.sub("", out)
    # Emphasis nests (**_x_**), so peel until it stops changing — bounded, since
    # each pass strictly shortens the string.
    for _ in range(3):
        peeled = _EMPHASIS.sub(r"\2", out)
        if peeled == out:
            break
        out = peeled
    out = _INLINE_BULLET.sub(" ", out)
    out = _STRAY_TICKS.sub("", out)
    return " ".join(out.split()).strip()


class FenceFilter:
    """Drop fenced blocks from a STREAM, where no chunk sees the whole fence.

    :func:`strip_markup` needs the finished reply; sentence-streaming TTS speaks
    the first sentence while the rest is still generating, so by the time the
    closing fence arrives the code has already been read aloud. This carries the
    open/closed state across chunks and simply never emits what is inside.
    """

    def __init__(self) -> None:
        self._inside = False
        self._tail = ""      # a partial fence marker split across chunks

    def feed(self, chunk: str) -> str:
        """Return the speakable part of ``chunk`` (may be empty)."""
        buf, self._tail = self._tail + (chunk or ""), ""
        out: list[str] = []
        while buf:
            marker = buf.find("```")
            if marker < 0:
                # Hold back up to two trailing backticks: they may be the start
                # of a fence whose third tick is in the next chunk.
                keep = len(buf) - len(buf.rstrip("`"))
                if keep:
                    self._tail, buf = buf[-keep:], buf[:-keep]
                if not self._inside:
                    out.append(buf)
                break
            if not self._inside:
                out.append(buf[:marker])
            buf = buf[marker + 3:]
            if not self._inside:
                # Swallow the language tag on the opening fence ("```json").
                newline = buf.find("\n")
                buf = buf[newline + 1:] if newline >= 0 else ""
            self._inside = not self._inside
        return "".join(out)

    def flush(self) -> str:
        """Anything held back at end of stream (never a real fence after all)."""
        tail, self._tail = self._tail, ""
        return "" if self._inside else tail


def looks_like_markup(text: str) -> bool:
    """True when the model answered in writing rather than in speech.

    A fence or a JSON object means the reply was never spoken language; the
    right response is to ask again, not to salvage the prose around it.
    """
    if not text:
        return False
    return "```" in text or bool(_JSON_BLOB.search(text))


def leaked_tool_names(text: str, names: object) -> list[str]:
    """Tool names the reply says out loud, e.g. "I can locate_in_app weather app".

    Only names MEDO actually has count, so ordinary snake_case in a quoted
    filename or a code answer isn't mistaken for a leak. ``names`` is any
    iterable of tool names; a non-iterable is treated as none.
    """
    if not text:
        return []
    try:
        known = {str(n).lower() for n in names}          # type: ignore[union-attr]
    except TypeError:
        return []
    if not known:
        return []
    seen: list[str] = []
    for word in _SNAKE.findall(text.lower()):
        if word in known and word not in seen:
            seen.append(word)
    return seen
