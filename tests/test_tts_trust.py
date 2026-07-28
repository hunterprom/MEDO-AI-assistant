"""Multilingual TTS: use the OS trust store so edge-tts works behind a
TLS-intercepting proxy, and never read non-English text with the English voice.
"""

from __future__ import annotations

import builtins
import sys
import types

import pytest


def test_use_os_trust_store_injects_when_present(monkeypatch):
    called = {}
    mod = types.ModuleType("truststore")
    mod.inject_into_ssl = lambda: called.setdefault("injected", True)
    monkeypatch.setitem(sys.modules, "truststore", mod)

    import main
    main._use_os_trust_store()
    assert called.get("injected") is True


def test_use_os_trust_store_is_safe_without_package(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "truststore":
            raise ImportError("no truststore here")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import main
    main._use_os_trust_store()   # best-effort: must never raise


@pytest.mark.asyncio
async def test_non_english_reply_is_not_gibberished_through_english_piper(monkeypatch):
    """If edge-tts can't voice a non-English reply, Piper (English-only here)
    must NOT be used — that's the 'speaks Slavic in English' bug."""
    from core.config import load_settings
    from voice.loop import VoiceLoop

    loop = VoiceLoop.__new__(VoiceLoop)          # skip heavy __init__
    loop._settings = load_settings()
    loop._turn_language = "mk"
    loop._speak_lock = __import__("asyncio").Lock()

    class _FailingEdge:
        async def synthesize(self, text, language=None):
            raise RuntimeError("edge unreachable (TLS)")

    class _Piper:
        def __init__(self):
            self.called = False

        def synthesize(self, text):
            self.called = True
            import numpy as np
            return np.zeros(10, dtype=np.int16), 22050

    piper = _Piper()
    loop._edge = _FailingEdge()
    loop._tts = piper

    played = {"n": 0}
    monkeypatch.setattr(loop, "_play_interruptible", lambda wav, sr: played.__setitem__("n", played["n"] + 1) or False)

    await loop._synthesize_and_play("Здраво, како си?", "mk")
    assert piper.called is False, "Macedonian must not be read by the English Piper voice"
    assert played["n"] == 0, "nothing should have been played (reply is shown, not voiced)"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
