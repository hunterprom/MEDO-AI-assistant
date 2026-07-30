"""The code generator — the active brain writes real library code for a brief.

Injectable so the engine is fully testable without a model: the engine hands the
Brain a domain system prompt + a user prompt, and gets back CODE (the source of
truth it will execute). The default calls MEDO's LLM (local by default; a
stronger/cloud brain per ``studio.model`` + the privacy gate). Models wrap code
in fences — ``extract_code`` pulls it out.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Tolerant of a language tag with trailing spaces and of CRLF line endings —
# a model that emits "```py \r\n…" must still have its code extracted, or the
# whole reply (backticks and all) gets written and fails to run.
_FENCE = re.compile(r"```[^\n`]*\r?\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> str:
    """The code from a model reply: the first fenced block, else the whole text
    (stripped). Always ends in a newline."""
    if not reply:
        return ""
    blocks = _FENCE.findall(reply)
    code = blocks[0] if blocks else reply
    code = code.strip("\n")
    return (code + "\n") if code else ""


class Brain:
    """``write_code(system, user) -> code``. ``chat`` is an injected async
    callable ``(model, messages) -> message-dict`` (e.g. ``OllamaClient.chat``);
    tests pass a fake."""

    def __init__(self, *, chat=None, model: str = "") -> None:
        self._chat = chat
        self._model = model

    async def write_code(self, system: str, user: str) -> str:
        if self._chat is None:
            raise RuntimeError("no brain is wired for Studio code generation")
        message = await self._chat(self._model, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        content = (message.get("content", "") if isinstance(message, dict)
                   else str(message))
        return extract_code(content or "")
