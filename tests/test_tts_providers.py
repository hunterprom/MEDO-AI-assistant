"""The TTS provider layer — S1 of the natural-voice work.

MEDO spoke fifteen of its sixteen languages through Microsoft's cloud. These
tests pin the replacement: one interface, a per-language voice map, local
first, and — the part that actually bit during the build — a refusal to voice
a language in some other language's voice.

Three real bugs are frozen here as regressions, all found by USING the map
rather than reading it:

* ``kokoro-multi-lang-v1_1`` was picked for Japanese on a summary saying Kokoro
  covers ja/ko. Its ONNX metadata says "Kokoro v1.1-zh … English, Chinese".
* Kokoro without an explicit ``lang`` phonemizes everything as en-us: Japanese
  came out 12.9 s long for a sentence that takes 4.2 s.
* Voice models arrive at wildly different levels — es_ES peaked at 3870 and
  tr_TR at 32746 on the same sentence, an 8x swing that reads as a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from voice import voices
from voice.providers import (
    PEAK_CEILING,
    TARGET_RMS,
    EdgeProvider,
    LegacyPiperProvider,
    TTSProvider,
    VoiceBox,
    normalize,
)
from voice.voices import VOICES, VoiceSpec

#: Every language MEDO claims to speak, from the one table that defines them.
ALL_CODES = [lang.code for lang in __import__(
    "core.languages", fromlist=["LANGUAGES"]).LANGUAGES]


class _Fake:
    """A provider that speaks the languages it was told to, and records calls."""

    def __init__(self, name: str, langs: set[str], *, audio: bool = True) -> None:
        self.name = name
        self._langs = langs
        self._audio = audio
        self.calls: list[str | None] = []

    def supports(self, language):
        return (language or "") in self._langs

    async def synthesize(self, text, language=None):
        self.calls.append(language)
        if not self._audio:
            return np.zeros(0, dtype=np.int16), 22050
        return np.ones(100, dtype=np.int16), 22050


# --- the map -----------------------------------------------------------------


def test_every_language_medo_claims_has_a_voice():
    missing = [c for c in ALL_CODES if voices.spec_for(c) is None]
    assert missing == [], f"languages with no voice at all: {missing}"


def test_only_macedonian_uses_the_cloud():
    cloud = sorted(c for c in ALL_CODES
                   if (voices.spec_for(c) or VoiceSpec("", "")).engine == "edge")
    assert cloud == ["mk"], (
        "the cloud is the declared exception for languages with NO local "
        f"voice; these drifted onto it: {cloud}")


def test_offline_only_never_returns_the_cloud():
    for code in ALL_CODES:
        spec = voices.spec_for(code, offline_only=True)
        assert spec is None or spec.engine != "edge", code


def test_the_stand_in_voice_is_never_handed_out_unasked():
    # THE BUG: with one flag doing both jobs, refusing the cloud silently
    # returned the Serbian voice, and the local provider then answered "yes, I
    # speak Macedonian" and would have read it aloud in Serbian. Refusing the
    # cloud must mean None until someone explicitly opts in.
    assert voices.spec_for("mk", offline_only=True) is None
    spec = voices.spec_for("mk", offline_only=True, allow_fallback=True)
    assert spec is not None and spec.fallback and "sr_RS" in spec.archive


def test_the_local_provider_does_not_claim_macedonian(tmp_path):
    from voice.providers import SherpaProvider

    assert not SherpaProvider(tmp_path).supports("mk")
    assert SherpaProvider(tmp_path, use_fallback_voices=True).supports("mk")


def test_kokoro_v1_1_is_not_used_for_japanese():
    # It is "Kokoro v1.1-zh": 103 speakers, all English or Chinese. Picking it
    # for ja was a summary-vs-metadata error; v1_0 is the nine-language one.
    ja = voices.spec_for("ja")
    assert ja is not None and ja.archive == "kokoro-multi-lang-v1_0"


def test_multilingual_models_declare_their_phonemizer_language():
    # A single model serving nine languages defaults to en-us, which made
    # Japanese 3x too long. Per-language models carry it in their own json.
    for code, spec in VOICES.items():
        if spec.engine == "sherpa-kokoro":
            assert spec.lang, f"{code} on a multi-language model needs lang"


def test_quality_tiers_are_honest_about_the_weak_ones():
    # Greek and Korean genuinely only have "low" voices available. The tier is
    # surfaced in the HUD and --demo, so it must not be quietly inflated.
    assert VOICES["el"].quality == "low"
    assert VOICES["ko"].quality == "low"
    assert VOICES["mk"].quality == "cloud"


def test_model_paths_are_derived_not_guessed(tmp_path):
    spec = VOICES["en"]
    assert voices.model_path(tmp_path, spec).name.endswith(".onnx")
    assert not voices.is_installed(tmp_path, spec)   # nothing downloaded yet


def test_an_unknown_language_has_no_voice():
    assert voices.spec_for("xx") is None
    assert voices.spec_for(None) is None
    assert voices.spec_for("") is None


# --- loudness ----------------------------------------------------------------


def test_normalize_brings_a_quiet_voice_up():
    quiet = (np.random.default_rng(0).standard_normal(8000) * 0.01).astype(np.float32)
    out = normalize(quiet)
    assert np.sqrt(np.mean(out ** 2)) == pytest.approx(TARGET_RMS, rel=0.02)


def test_normalize_brings_a_loud_voice_down():
    loud = (np.random.default_rng(1).standard_normal(8000) * 0.5).astype(np.float32)
    out = normalize(loud)
    assert np.sqrt(np.mean(out ** 2)) == pytest.approx(TARGET_RMS, rel=0.02)


def test_normalize_never_clips():
    # A quiet clip with one big transient: scaling to the RMS target would push
    # the peak past full scale and crack.
    spiky = np.full(8000, 0.001, dtype=np.float32)
    spiky[100] = 0.9
    out = normalize(spiky)
    assert np.abs(out).max() <= PEAK_CEILING + 1e-6


def test_normalize_leaves_silence_alone():
    assert normalize(np.zeros(0, dtype=np.float32)).size == 0
    silent = np.zeros(1000, dtype=np.float32)
    assert np.array_equal(normalize(silent), silent)


# --- the chain ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_first_provider_that_can_speak_it_wins():
    local = _Fake("local", {"de"})
    cloud = _Fake("cloud", {"de", "mk"})
    box = VoiceBox([local, cloud])
    audio, _ = await box.synthesize("hallo", "de")
    assert audio.size and local.calls == ["de"] and cloud.calls == []


@pytest.mark.asyncio
async def test_a_language_only_the_cloud_covers_reaches_the_cloud():
    local = _Fake("local", {"de"})
    cloud = _Fake("cloud", {"mk"})
    box = VoiceBox([local, cloud])
    await box.synthesize("здраво", "mk")
    assert cloud.calls == ["mk"] and local.calls == []


@pytest.mark.asyncio
async def test_a_provider_that_returns_silence_hands_on():
    broken = _Fake("broken", {"de"}, audio=False)
    backup = _Fake("backup", {"de"})
    box = VoiceBox([broken, backup])
    audio, _ = await box.synthesize("hallo", "de")
    assert audio.size and broken.calls == ["de"] and backup.calls == ["de"]


@pytest.mark.asyncio
async def test_nothing_can_speak_it_returns_silence_not_the_wrong_voice():
    # The caller must SHOW the reply. Handing Japanese to an English voice is
    # the "reads it out character by character" bug.
    box = VoiceBox([_Fake("local", {"en"})])
    audio, _ = await box.synthesize("こんにちは", "ja")
    assert audio.size == 0
    assert not box.can_speak("ja")


def test_provider_for_names_who_would_speak_it():
    local = _Fake("local", {"de"})
    box = VoiceBox([local, _Fake("cloud", {"mk"})])
    assert box.provider_for("de") is local
    assert box.provider_for("zz") is None


# --- individual providers ----------------------------------------------------


def test_edge_provider_is_refused_when_cloud_is_off():
    class _Edge:
        pass

    assert not EdgeProvider(_Edge(), allow_cloud=False).supports("mk")
    assert EdgeProvider(_Edge(), allow_cloud=True).supports("mk")


def test_edge_provider_only_offers_languages_with_no_local_voice():
    class _Edge:
        pass

    edge = EdgeProvider(_Edge(), allow_cloud=True)
    assert edge.supports("mk")
    # German has a local voice; routing it to the cloud is the exact drift the
    # provider split exists to prevent.
    assert not edge.supports("de")


def test_edge_provider_without_an_engine_supports_nothing():
    assert not EdgeProvider(None, allow_cloud=True).supports("mk")


def test_legacy_piper_is_english_only():
    # It produced the "speaks Japanese with English phonemes" bug when handed
    # text it had no voice for. It is last in the chain AND narrow.
    piper = LegacyPiperProvider(object())
    assert piper.supports("en")
    for code in ("de", "ja", "mk", "ru"):
        assert not piper.supports(code), code


def test_legacy_piper_without_a_voice_supports_nothing():
    assert not LegacyPiperProvider(None).supports("en")


def test_the_providers_satisfy_the_interface():
    for provider in (EdgeProvider(None), LegacyPiperProvider(None),
                     _Fake("x", set())):
        assert isinstance(provider, TTSProvider)
