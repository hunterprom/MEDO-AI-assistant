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
    """``write_code(system, user) -> code``. Either inject ``chat`` (an async
    ``(model, messages) -> message-dict`` like ``OllamaClient.chat``; tests pass a
    fake) or pass ``settings`` and the Brain builds the active LLM client lazily
    (so the skill layer needn't hold the client). ``model`` defaults to the
    active default brain."""

    def __init__(self, *, chat=None, model: str = "", settings=None) -> None:
        self._chat = chat
        self._model = model
        self._settings = settings
        self._client = None

    async def write_code(self, system: str, user: str) -> str:
        chat = self._chat or self._lazy_chat
        model = self._model or (self._settings.llm.default_model
                                if self._settings is not None else "")
        message = await chat(model, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        content = (message.get("content", "") if isinstance(message, dict)
                   else str(message))
        return extract_code(content or "")

    async def _lazy_chat(self, model, messages):
        if self._settings is None:
            raise RuntimeError("no brain is wired for Studio code generation")
        if self._client is None:
            from llm.client import OllamaClient

            # Writing a part is a BUILD, not a spoken turn, so it gets the
            # studio's own allowance rather than the conversation's. Measured:
            # a parametric phone stand is 195 s of genuine code writing, and
            # llm.cli_timeout_s (180 s, sized so nobody waits three minutes for
            # an answer out loud) killed it just short — every time, then three
            # more times through the retry loop.
            cfg = self._settings.llm.model_copy()
            budget = float(getattr(self._settings.studio, "generate_timeout_s", 0)
                           or 0)
            if budget > 0:
                cfg.cli_timeout_s = max(cfg.cli_timeout_s, budget)
                cfg.request_timeout_s = max(cfg.request_timeout_s, budget)
            self._client = OllamaClient(cfg)
        # NOT for speech. On a CLI-agent brain the flattened prompt otherwise
        # ends "Reply with only your spoken answer, as plain text" — the last
        # line the agent reads, and it beats the domain's "output ONLY the
        # Python code". That is how a request for a phone stand came back as a
        # paragraph describing one, which was then executed as Python.
        return await self._client.chat(model, messages, for_speech=False)
