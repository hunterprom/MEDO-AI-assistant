"""Live interpreter mode: translation direction, captions, exit phrase."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import load_settings
from core.events import Event, EventType
from core.modes import SessionModes
from voice.loop import VoiceLoop


class _Bus:
    def __init__(self):
        self.events = []

    async def emit(self, event: Event):
        self.events.append(event)


class _SM:
    def __init__(self):
        self.bus = _Bus()

    async def transition(self, state):
        pass


class _LLM:
    def __init__(self, reply="TRANSLATED"):
        self._reply = reply
        self.calls = []

    async def chat(self, model, messages, **kw):
        self.calls.append(messages)
        return {"content": self._reply}


class _Router:
    def __init__(self, reply="TRANSLATED"):
        self.model = "test-model"
        self.llm = _LLM(reply)
        self.routed = []

    async def route(self, text, **kw):
        self.routed.append(text)
        from core.router import RouteResult
        from core.events import RoutePath
        return RouteResult(path=RoutePath.FAST, speech="Interpreter mode off.")


class _UI:
    def __init__(self):
        self.lines = []

    def transcript(self, text):
        self.lines.append(text)


def _loop(modes, router, stt_return):
    settings = load_settings()
    loop = VoiceLoop(settings, router=router, sm=_SM(), announcer=object(),
                     ui=_UI(), log=object(), modes=modes)

    class _STT:
        def transcribe_with_language(self, audio):
            return stt_return

    loop._stt = _STT()
    spoken = []

    async def _fake_speak(text, language=None):
        spoken.append(text)
        return 0.0

    loop._speak = _fake_speak
    loop._spoken = spoken
    return loop


AUDIO = np.ones(1600, dtype=np.float32)


@pytest.mark.asyncio
async def test_translate_uses_llm_and_directions():
    modes = SessionModes(interpreter=True, interpreter_langs=("en", "mk"))
    router = _Router(reply="Здраво")
    loop = _loop(modes, router, ("Hello there", "en"))
    keep = await loop._interpret_turn(AUDIO)
    assert keep is True
    assert loop._spoken == ["Здраво"]                 # spoke the translation
    caption = loop._sm.bus.events[-1]
    assert caption.type is EventType.CAPTION
    assert caption.payload["src"] == "en" and caption.payload["dst"] == "mk"
    assert caption.payload["src_text"] == "Hello there"
    assert caption.payload["dst_text"] == "Здраво"


@pytest.mark.asyncio
async def test_direction_flips_with_detected_language():
    modes = SessionModes(interpreter=True, interpreter_langs=("en", "mk"))
    router = _Router(reply="Take the bus.")
    loop = _loop(modes, router, ("Земи автобус", "mk"))
    await loop._interpret_turn(AUDIO)
    caption = loop._sm.bus.events[-1].payload
    assert caption["src"] == "mk" and caption["dst"] == "en"
    assert loop._spoken == ["Take the bus."]


@pytest.mark.asyncio
async def test_unknown_language_defaults_to_first_of_pair():
    modes = SessionModes(interpreter=True, interpreter_langs=("en", "mk"))
    router = _Router()
    loop = _loop(modes, router, ("bonjour", "fr"))   # not in the pair
    await loop._interpret_turn(AUDIO)
    caption = loop._sm.bus.events[-1].payload
    assert caption["src"] == "en" and caption["dst"] == "mk"


@pytest.mark.asyncio
async def test_exit_phrase_routes_and_stops():
    modes = SessionModes(interpreter=True, interpreter_langs=("en", "mk"))
    router = _Router()
    loop = _loop(modes, router, ("stop interpreting", "en"))
    keep = await loop._interpret_turn(AUDIO)
    assert keep is False                              # back to standby
    assert router.routed == ["stop interpreting"]     # routed, not translated
    assert router.llm.calls == []                     # never called the translator


@pytest.mark.asyncio
async def test_macedonian_exit_phrase():
    modes = SessionModes(interpreter=True)
    router = _Router()
    loop = _loop(modes, router, ("прекини со преведување", "mk"))
    keep = await loop._interpret_turn(AUDIO)
    assert keep is False and router.routed


@pytest.mark.asyncio
async def test_empty_transcription_keeps_listening():
    modes = SessionModes(interpreter=True)
    router = _Router()
    loop = _loop(modes, router, ("   ", "en"))
    keep = await loop._interpret_turn(AUDIO)
    assert keep is True and loop._spoken == []


@pytest.mark.asyncio
async def test_translate_without_model_echoes():
    modes = SessionModes(interpreter=True)
    router = _Router()
    router.model = None
    loop = _loop(modes, router, ("hello", "en"))
    out = await loop._translate("hello", "en", "mk")
    assert out == "hello"
