"""First-run wizard state machine (S3): advance / fail / retry / resume.

A FakeOps stands in for the real installers/downloads, so every path is pinned
without touching the network: the happy path, an install failure then retry, a
resumed (partial) model download, a mic with no devices, and that copy exists in
both languages with no traceback ever surfacing.
"""

from __future__ import annotations

import pytest

from app import profiles
from app.wizard import COPY, FAIL_COPY, STEP_ORDER, Wizard


class FakeOps:
    def __init__(self, *, installed=True, install_ok=True, present=(),
                 pull_ok=True, mics=("Built-in Mic",), mic_ok=True):
        self.installed = installed
        self.install_ok = install_ok
        self.present = set(present)
        self.pull_ok = pull_ok
        self.mics = list(mics)
        self.mic_ok = mic_ok
        self.pulled: list[str] = []
        self.install_called = False

    def ollama_installed(self):
        return self.installed

    def install_ollama(self, progress):
        self.install_called = True
        progress(0.5)
        if self.install_ok:
            self.installed = True
            progress(1.0)
        return self.install_ok

    def model_present(self, tag):
        return tag in self.present

    def pull_model(self, tag, progress):
        progress(0.5)
        if not self.pull_ok:
            return False
        self.pulled.append(tag)
        self.present.add(tag)
        progress(1.0)
        return True

    def list_microphones(self):
        return self.mics

    def test_microphone(self, index):
        return self.mic_ok


def _wiz(ops=None, lang="en", profile=None):
    return Wizard(ops or FakeOps(), profile or profiles.LITE, lang=lang)


def _run_all(w):
    while True:
        w.run_current()
        if not w.advance():
            break
    return w


# -- happy path ---------------------------------------------------------------

def test_full_run_completes():
    w = _wiz()
    _run_all(w)
    assert w.is_complete()
    assert all(s.status == "ok" for s in w.steps)


def test_ollama_already_installed_skips_install():
    ops = FakeOps(installed=True)
    w = _wiz(ops)
    w.run_current(); w.advance()          # welcome
    step = w.run_current()                 # ollama
    assert step.status == "ok" and ops.install_called is False


def test_ollama_missing_triggers_install():
    ops = FakeOps(installed=False, install_ok=True)
    w = _wiz(ops)
    w.run_current(); w.advance()
    step = w.run_current()
    assert step.status == "ok" and ops.install_called is True


# -- failure + retry ----------------------------------------------------------

def test_install_failure_is_friendly_and_retryable():
    ops = FakeOps(installed=False, install_ok=False)
    w = _wiz(ops)
    w.run_current(); w.advance()
    step = w.run_current()
    assert step.status == "failed"
    assert step.message == FAIL_COPY["en"]["ollama"]     # no traceback
    assert w.advance() is False                          # can't move past a failure
    ops.install_ok = True                                # user fixed their wifi
    step = w.retry()
    assert step.status == "ok" and w.advance() is True


def test_a_crashing_op_becomes_a_failed_step_not_an_exception():
    class Boom(FakeOps):
        def install_ollama(self, progress):
            raise RuntimeError("network stack died")
    ops = Boom(installed=False)
    w = _wiz(ops)
    w.run_current(); w.advance()
    step = w.run_current()
    assert step.status == "failed" and "network stack died" in step.detail
    assert step.message == FAIL_COPY["en"]["ollama"]


# -- resume -------------------------------------------------------------------

def test_model_download_resumes_only_the_missing():
    # llama3.2:3b already pulled last time; only nomic-embed-text should download.
    ops = FakeOps(present=["llama3.2:3b"])
    w = _wiz(ops)
    for _ in range(STEP_ORDER.index("models")):
        w.run_current(); w.advance()
    step = w.run_current()
    assert step.status == "ok"
    assert ops.pulled == ["nomic-embed-text"]            # 3b was skipped
    assert step.progress == 1.0


def test_model_download_failure_then_retry():
    ops = FakeOps(pull_ok=False)
    w = _wiz(ops)
    for _ in range(STEP_ORDER.index("models")):
        w.run_current(); w.advance()
    assert w.run_current().status == "failed"
    ops.pull_ok = True
    assert w.retry().status == "ok"


# -- microphone ---------------------------------------------------------------

def test_no_microphone_fails_cleanly():
    ops = FakeOps(mics=())
    w = _wiz(ops)
    for _ in range(STEP_ORDER.index("microphone")):
        w.run_current(); w.advance()
    step = w.run_current()
    assert step.status == "failed" and step.message == FAIL_COPY["en"]["microphone"]


# -- guards + payload ---------------------------------------------------------

def test_advance_requires_current_success():
    w = _wiz(FakeOps(installed=False, install_ok=False))
    w.run_current(); w.advance()
    w.run_current()                                       # ollama fails
    assert w.current.name == "ollama" and w.advance() is False


def test_to_dict_shape():
    w = _wiz()
    d = w.to_dict()
    assert d["index"] == 0 and d["total"] == len(STEP_ORDER)
    assert d["copy"]["title"] == "Welcome to MEDO"
    assert d["step"]["name"] == "welcome"


# -- localization + copy completeness -----------------------------------------

def test_macedonian_copy_is_used():
    w = _wiz(lang="mk")
    assert w.copy("welcome")["title"] == "Добредојде во MEDO"
    w = _wiz(lang="fr")                                   # unknown -> en fallback
    assert w.lang == "en"


def test_every_step_has_copy_in_both_languages():
    for lang in ("en", "mk"):
        for step in STEP_ORDER:
            c = COPY[lang][step]
            assert c["title"] and c["body"] and c["action"], (lang, step)
    for lang in ("en", "mk"):
        for step in ("ollama", "models", "microphone", "generic"):
            assert FAIL_COPY[lang][step]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
