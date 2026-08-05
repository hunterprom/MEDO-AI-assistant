"""Screen agent (EXPERIMENTAL): MEDO performs a multi-step GUI task by looking.

"Do this for me: …" — MEDO screenshots the desktop, asks the local vision
model (qwen2.5-vl) for the single next action, executes it (click / type /
scroll / key), and repeats until the model says done or a step cap is hit.

Safety by construction:
- ``controls_pc`` — the HUD PC-control switch blocks it entirely.
- ``requires_confirmation`` — one spoken yes/no before ANY clicking starts.
- Bounded to ``vision.agent_max_steps`` actions per task.

Honest limits: reliability rests on the local model's ability to point at the
right pixel. It's genuinely useful for simple, unambiguous tasks and will
misfire on dense UIs — hence the confirmation gate and step cap. Capture,
model call, and action execution are all injectable so the loop is fully
unit-tested without a screen, Ollama, or pyautogui.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from typing import Any

from core.config import Settings
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are controlling a Windows computer to accomplish the user's task. "
    "You are shown a screenshot. Decide the SINGLE next action and reply with "
    "ONLY a JSON object — no prose, no markdown. Schema:\n"
    '{"action": "click|double_click|type|scroll|key|done", '
    '"x": int 0-1000, "y": int 0-1000, "text": str, '
    '"amount": int, "keys": str, "success": bool, "say": str}\n'
    "x/y are the cursor target as coordinates on THIS image, normalized to "
    "0-1000 (x left→right, y top→bottom). Include x/y for click and "
    "double_click. text is what to type. amount is scroll distance (positive "
    "= down). keys is a key or chord like \"enter\" or \"ctrl a\". When you have "
    "actually COMPLETED the task, use action \"done\" with success true and a "
    "short spoken result in say. If the task is impossible or you cannot make "
    "progress, use action \"done\" with success FALSE and say why — never claim "
    "success for something you did not finish."
)


def scale_point(x_norm: float, y_norm: float, w: int, h: int) -> tuple[int, int]:
    """Normalized 0-1000 point → clamped absolute screen pixels. Pure."""
    x = round(max(0.0, min(1000.0, x_norm)) / 1000.0 * w)
    y = round(max(0.0, min(1000.0, y_norm)) / 1000.0 * h)
    return max(0, min(w - 1, x)), max(0, min(h - 1, y))


#: A "done" whose `say` reads like a give-up. Small models often omit the
#: success flag entirely while plainly saying they failed — defaulting those to
#: success would announce a completion that never happened.
_GAVE_UP_RE = re.compile(
    r"\b(?:can'?t|cannot|couldn'?t|could not|unable|impossible|not (?:able|possible)|"
    r"fail(?:ed|ure)?|gave up|no way|didn'?t work|unsuccessful)\b", re.IGNORECASE)


def done_succeeded(action: dict) -> bool:
    """Whether a model's ``done`` action really means COMPLETED.

    Honours an explicit ``success`` flag; when it's absent, a give-up phrasing in
    ``say`` counts as failure (pure, so both directions are unit-tested).
    """
    flag = action.get("success")
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str) and flag.strip().lower() in ("true", "false"):
        return flag.strip().lower() == "true"
    return not _GAVE_UP_RE.search(str(action.get("say") or ""))


def parse_action(raw: str) -> dict | None:
    """Extract the action JSON from a model reply (tolerates surrounding text)."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)   # first {...} blob
    if m:
        try:
            return json.loads(m.group(0))
        except (ValueError, TypeError):
            return None
    return None


class ScreenAgentSkill(Skill):
    name = "operate_screen"
    description = (
        "Perform a multi-step task on the user's screen by looking at it and "
        "clicking, typing, and scrolling (e.g. 'open my email and read the "
        "newest'). Experimental — use only when the user asks MEDO to DO "
        "something on screen, not to describe it."
    )
    controls_pc = True
    requires_confirmation = True

    patterns = [
        re.compile(r"\bdo\s+(?:this|it|that)\s+for\s+me\b", re.IGNORECASE),
        re.compile(r"\boperate\s+(?:my\s+)?(?:screen|computer|pc)\b", re.IGNORECASE),
        re.compile(r"\bfor\s+me\s+on\s+(?:my\s+|the\s+)?screen\b", re.IGNORECASE),
    ]

    def __init__(self, settings: Settings, capture=None, ask=None, act=None) -> None:
        self._settings = settings
        self._capture = capture or self._default_capture
        self._ask = ask or self._default_ask
        self._act = act or self._default_act

    # -- task extraction ------------------------------------------------------

    @staticmethod
    def _task_from(request: SkillRequest) -> str:
        if request.args.get("task"):
            return str(request.args["task"]).strip()
        text = request.text.strip()
        # "open my email for me on the screen" — the task is BEFORE the wrapper.
        # The old after-the-first-keyword capture returned "on the screen" here
        # and the agent then pursued that as the task.
        trailing = re.search(r"^(.*?)\s+for\s+me\s+on\s+(?:my\s+|the\s+)?screen\b.*$",
                             text, re.IGNORECASE)
        if trailing and trailing.group(1).strip(" .?!,"):
            return trailing.group(1).strip(" .?!,")
        # "do this for me: open notepad" / "operate my screen and open notepad"
        # — strip the leading wrapper; the task is what follows.
        lead = re.sub(r"^(?:do\s+(?:this|it|that)\s+for\s+me|"
                      r"operate\s+(?:my\s+)?(?:screen|computer|pc))\b[:,]?\s*"
                      r"(?:and\s+|to\s+|please\s+)?", "", text, flags=re.IGNORECASE)
        if lead != text and lead.strip(" .?!,"):
            return lead.strip(" .?!,")
        # Fallback: the original after-the-first-keyword capture.
        m = re.search(r"(?:for\s+me|screen|computer|pc)\b[:,]?\s*"
                      r"(?:and\s+|to\s+|please\s+)?(.+)$", text, re.IGNORECASE)
        return (m.group(1).strip(" .?!") if m else text.strip())

    # -- the agent loop -------------------------------------------------------

    async def execute(self, request: SkillRequest) -> SkillResult:
        if not self._settings.vision.agent_enabled:
            return SkillResult("The screen agent is disabled in config.", success=False)
        task = self._task_from(request)
        if not task:
            return SkillResult("What would you like me to do on screen?", success=False)
        if not request.context.get("confirmed"):
            return SkillResult(
                f"I'll take control of the screen to: {task}. Shall I go ahead?",
                needs_confirmation=True)

        history: list[str] = []
        max_steps = self._settings.vision.agent_max_steps
        for step in range(max_steps):
            try:
                image, (w, h) = await asyncio.to_thread(self._capture)
            except Exception as exc:
                return SkillResult(f"I couldn't capture the screen: {exc}", success=False)
            # Guard the model call + its reply: Ollama being down, or a reply
            # that's a JSON array/scalar (a common LLM habit), must degrade to a
            # spoken error, not crash the turn out of the router.
            try:
                raw = await self._ask(image, task, history)
            except Exception as exc:
                logger.warning("screen-agent vision call failed: %s", exc)
                return SkillResult(
                    "I couldn't reach my vision model, so I stopped — is Ollama "
                    "running?", success=False)
            action = parse_action(raw)
            if not isinstance(action, dict):
                return SkillResult(
                    "I couldn't work out the next step, so I stopped.", success=False)
            kind = str(action.get("action", "")).lower()
            if kind == "done":
                # 'done' means finished OR impossible — honour the success flag
                # (and a give-up `say` when the flag is missing) so a gave-up run
                # isn't reported as a completed one.
                ok = done_succeeded(action)
                say = action.get("say") or (
                    "Done." if ok else "I couldn't finish that on screen.")
                return SkillResult(say, success=ok, data={"steps": step})
            try:
                desc = await asyncio.to_thread(self._act, action, w, h)
            except Exception:
                logger.exception("screen action failed")
                return SkillResult("That step failed, so I stopped.", success=False)
            history.append(desc)
            await asyncio.sleep(0.6)   # let the UI settle before the next look
        return SkillResult(
            f"I ran {max_steps} steps on '{task}' and stopped at the limit.",
            data={"steps": max_steps})

    # -- default (real) backends ---------------------------------------------

    @staticmethod
    def _default_capture():
        import pyautogui

        shot = pyautogui.screenshot()
        return shot, shot.size

    async def _default_ask(self, image, task: str, history: list[str]) -> str:
        from skills.vision_skill import _shrink

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        small = _shrink(buf.getvalue())
        prompt = SYSTEM_PROMPT + f"\n\nTask: {task}"
        if history:
            prompt += "\nSteps done so far: " + "; ".join(history)
        import httpx

        host = self._settings.llm.host.rstrip("/")
        model = self._settings.vision_llm.model
        async with httpx.AsyncClient(timeout=self._settings.vision_llm.timeout_s) as client:
            resp = await client.post(f"{host}/api/generate", json={
                "model": model, "prompt": prompt,
                "images": [base64.b64encode(small).decode()],
                "stream": False, "options": {"temperature": 0.0},
            })
            resp.raise_for_status()
            return (resp.json().get("response") or "").strip()

    @staticmethod
    def _default_act(action: dict, w: int, h: int) -> str:
        from skills.desktop import _pyautogui

        gui = _pyautogui()
        kind = str(action.get("action", "")).lower()
        if kind in ("click", "double_click"):
            x, y = scale_point(float(action.get("x", 500)),
                               float(action.get("y", 500)), w, h)
            (gui.doubleClick if kind == "double_click" else gui.click)(x, y)
            return f"{kind} at ({x},{y})"
        if kind == "type":
            text = str(action.get("text", ""))
            gui.write(text, interval=0.02)
            return f"typed {text!r}"
        if kind == "scroll":
            amount = int(action.get("amount", 3))
            gui.scroll(-amount * 120)               # +amount = down
            return f"scrolled {amount}"
        if kind == "key":
            keys = action.get("keys", "")
            if isinstance(keys, list):           # model may return ["ctrl","a"]
                parts = [str(k).strip() for k in keys]
            else:
                parts = re.split(r"[+\s]+", str(keys).strip())
            parts = [p for p in parts if p]
            if len(parts) == 1:
                gui.press(parts[0])
            elif parts:
                gui.hotkey(*parts)
            return f"pressed {'+'.join(parts)}"
        return f"ignored unknown action {kind!r}"

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string",
                                 "description": "What to do on screen, e.g. "
                                                "'open Notepad and type hello'."},
                    },
                    "required": ["task"],
                },
            },
        }
