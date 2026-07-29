"""First-run setup wizard — a testable state machine + plain-language copy (S3).

The user sees web pages (served by the HUD); this module is the LOGIC behind
them: an ordered set of steps, each of which advances / fails / retries, with the
setup OPERATIONS (install Ollama, pull models, test the mic) injected so every
transition is unit-tested with fakes — no real downloads. Two rules baked in:

* **Resumable.** The model step pulls only what's MISSING, so an interrupted
  download just picks up where it left off on retry.
* **No tracebacks.** Every failure carries a friendly, localized message; the
  technical detail goes to ``Step.detail`` for the log, never the screen.

Copy is EN + MK (MEDO is bilingual). ``to_dict`` is what the HUD renders.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Protocol

logger = logging.getLogger(__name__)

STEP_ORDER = ["welcome", "ollama", "models", "microphone", "done"]

Progress = Callable[[float], None]


class SetupOps(Protocol):
    """The real installers/probes (app.setup_ops); faked in tests."""

    def ollama_installed(self) -> bool: ...
    def install_ollama(self, progress: Progress) -> bool: ...
    def model_present(self, tag: str) -> bool: ...
    def pull_model(self, tag: str, progress: Progress) -> bool: ...
    def list_microphones(self) -> List[str]: ...
    def test_microphone(self, index: int) -> bool: ...


# -- plain-language copy (EN + MK) --------------------------------------------

COPY = {
    "en": {
        "welcome": {"title": "Welcome to MEDO",
                    "body": ("MEDO is your private assistant that runs on THIS "
                             "computer — your voice never leaves it. Let's get "
                             "it set up; this takes a few minutes."),
                    "action": "Get started"},
        "ollama": {"title": "Setting up MEDO's engine",
                   "body": ("MEDO uses a small engine called Ollama to think. "
                            "We'll install it for you — nothing for you to do."),
                   "action": "Install"},
        "models": {"title": "Downloading MEDO's brain",
                   "body": ("This one-time download lets MEDO think without the "
                            "internet. It can be large depending on your "
                            "computer, so it may take a while — you can keep "
                            "using your PC while it works."),
                   "action": "Download"},
        "microphone": {"title": "Let's hear you",
                       "body": ("Pick your microphone and say a few words so "
                                "MEDO can check it's working."),
                       "action": "Test microphone"},
        "done": {"title": "You're all set",
                 "body": ("MEDO is ready. Try saying your wake word to start — "
                          "for example, 'Hey MEDO'."),
                 "action": "Finish"},
    },
    "mk": {
        "welcome": {"title": "Добредојде во MEDO",
                    "body": ("MEDO е твојот приватен асистент што работи на ОВОЈ "
                             "компјутер — твојот глас никогаш не го напушта. Ајде "
                             "да го поставиме; ова трае неколку минути."),
                    "action": "Започни"},
        "ollama": {"title": "Го подготвуваме моторот на MEDO",
                   "body": ("MEDO користи мал мотор наречен Ollama за да "
                            "размислува. Ќе го инсталираме за тебе — ти немаш "
                            "што да правиш."),
                   "action": "Инсталирај"},
        "models": {"title": "Го преземаме мозокот на MEDO",
                   "body": ("Ова еднократно преземање му овозможува на MEDO да "
                            "размислува без интернет. Може да биде големо зависно "
                            "од твојот компјутер, па може да потрае — можеш да го "
                            "користиш компјутерот во меѓувреме."),
                   "action": "Преземи"},
        "microphone": {"title": "Да те чуеме",
                       "body": ("Избери го микрофонот и кажи неколку зборови за "
                                "MEDO да провери дали работи."),
                       "action": "Тестирај микрофон"},
        "done": {"title": "Сè е подготвено",
                 "body": ("MEDO е спремен. Кажи ја активирачката фраза за да "
                          "започнеш — на пример, 'Еј MEDO'."),
                 "action": "Заврши"},
    },
}

FAIL_COPY = {
    "en": {
        "ollama": ("We couldn't install MEDO's engine. Please check your "
                   "internet connection and try again."),
        "models": ("The download didn't finish. It'll pick up where it left "
                   "off — please try again."),
        "microphone": ("We couldn't hear your microphone. Check it's plugged in "
                       "and selected, then try again."),
        "generic": "Something went wrong. Please try again.",
    },
    "mk": {
        "ollama": ("Не можевме да го инсталираме моторот на MEDO. Провери ја "
                   "интернет-врската и обиди се повторно."),
        "models": ("Преземањето не заврши. Ќе продолжи од каде застана — обиди "
                   "се повторно."),
        "microphone": ("Не можевме да го слушнеме микрофонот. Провери дали е "
                       "приклучен и избран, па обиди се повторно."),
        "generic": "Нешто не успеа. Обиди се повторно.",
    },
}


@dataclass
class Step:
    name: str
    status: str = "pending"          # pending | running | ok | failed
    progress: float = 0.0            # 0..1
    message: str = ""                # user-facing (localized)
    detail: str = ""                 # technical, for the log only


@dataclass
class Wizard:
    ops: SetupOps
    profile: object                  # app.profiles.Profile (has ollama_models())
    lang: str = "en"
    selected_mic: Optional[int] = None
    steps: List[Step] = field(default_factory=lambda: [Step(n) for n in STEP_ORDER])
    index: int = 0

    def __post_init__(self) -> None:
        if self.lang not in COPY:
            self.lang = "en"

    # -- accessors ------------------------------------------------------------

    @property
    def current(self) -> Step:
        return self.steps[self.index]

    def copy(self, step_name: str) -> dict:
        return COPY[self.lang][step_name]

    def _fail_message(self, step_name: str) -> str:
        return FAIL_COPY[self.lang].get(step_name, FAIL_COPY[self.lang]["generic"])

    # -- driving the flow -----------------------------------------------------

    def run_current(self) -> Step:
        """Execute the current step. Never raises — a crash becomes a failed step
        with a friendly message and the detail logged."""
        step = self.current
        step.status = "running"
        step.progress = 0.0
        step.detail = ""
        try:
            ok = self._execute(step)
        except Exception as exc:  # a broken op must not crash the wizard
            logger.warning("wizard step %s raised", step.name, exc_info=True)
            step.detail = repr(exc)
            ok = False
        step.status = "ok" if ok else "failed"
        step.message = (self.copy(step.name)["body"] if ok
                        else self._fail_message(step.name))
        if ok:
            step.progress = 1.0
        return step

    def _execute(self, step: Step) -> bool:
        name = step.name
        if name in ("welcome", "done"):
            return True
        if name == "ollama":
            if self.ops.ollama_installed():
                return True
            return bool(self.ops.install_ollama(lambda p: self._set_progress(step, p)))
        if name == "models":
            return self._download_models(step)
        if name == "microphone":
            mics = self.ops.list_microphones()
            if not mics:
                return False
            idx = self.selected_mic if self.selected_mic is not None else 0
            if not (0 <= idx < len(mics)):
                return False
            return bool(self.ops.test_microphone(idx))
        return False

    def _download_models(self, step: Step) -> bool:
        models = list(self.profile.ollama_models())
        total = len(models) or 1
        for done, tag in enumerate(models):
            if self.ops.model_present(tag):        # resume: skip what we have
                self._set_progress(step, (done + 1) / total)
                continue
            ok = self.ops.pull_model(
                tag, lambda p, d=done: self._set_progress(step, (d + p) / total))
            if not ok:
                return False
            self._set_progress(step, (done + 1) / total)
        return True

    @staticmethod
    def _set_progress(step: Step, p: float) -> None:
        step.progress = max(0.0, min(1.0, float(p)))

    def retry(self) -> Step:
        self.current.status = "pending"
        return self.run_current()

    def advance(self) -> bool:
        """Move to the next step, only if the current one succeeded."""
        if self.current.status != "ok":
            return False
        if self.index < len(self.steps) - 1:
            self.index += 1
            return True
        return False

    def is_complete(self) -> bool:
        return self.current.name == "done" and self.current.status == "ok"

    # -- HUD payload ----------------------------------------------------------

    def to_dict(self) -> dict:
        step = self.current
        return {
            "lang": self.lang,
            "index": self.index,
            "total": len(self.steps),
            "step": asdict(step),
            "copy": self.copy(step.name),
            "steps": [s.name for s in self.steps],
            "complete": self.is_complete(),
        }
