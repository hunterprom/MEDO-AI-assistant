"""Which voice speaks which language, and where that voice comes from.

MEDO spoke sixteen languages by sending fifteen of them to Microsoft's cloud
(``edge-tts``). That was an honest trade at the time — the alternative was
shipping sixteen voice models — but it means MEDO's voice is not local, which
is the one thing MEDO is supposed to be. This table is what replaces it.

Every voice here was read from the **sherpa-onnx release asset list** and, for
the multi-speaker ones, from the ONNX metadata itself — not from a summary.
That distinction cost something: ``kokoro-multi-lang-v1_1`` was picked first
on a claim that Kokoro covers Japanese and Korean, and its metadata says
"Kokoro v1.1-zh … supporting English, Chinese" with 103 speakers that are all
English or Chinese. Check the model, not the write-up.

Three families cover fifteen of the sixteen:

* **Piper** (VITS) for thirteen. Fast, tiny, CPU-only — measured RTF 0.057 on
  this machine, i.e. seventeen times faster than real time, with no VRAM at
  all. That matters: MEDO's GPU is already shared with the LLM and vision.
* **Kokoro v1_0** for Japanese: 53 speakers over nine languages, of which
  ``jf_alpha`` (id 37) is the Japanese female read. Piper has no Japanese
  voice at any quality.
* **mimic3** for Korean, at the "low" tier, because it is the only Korean TTS
  in sherpa-onnx's 272 models. Neither Kokoro version has a Korean speaker.

**Macedonian has no local voice, and no amount of configuration invents one.**
Piper has none, Kokoro has none, Chatterbox's 23 languages have none, and
Meta's MMS is CC-BY-NC so it can't ship in something sold. mk therefore stays
on edge-tts as a single, declared, config-gated cloud exception until a
fine-tuned voice exists — see docs/Voice.md. Serbian (the nearest Cyrillic
Slavic voice Piper does have) is registered as ``fallback`` so the offline
path is a documented choice rather than silence.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)

#: Where sherpa-onnx publishes its packaged voices. One tag, 600+ assets.
RELEASE_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
               "tts-models/{archive}.tar.bz2")

#: espeak-ng-data ships inside EVERY piper archive and is identical in all of
#: them (18 MB of phonemizer rules for every language espeak knows). Extracted
#: once to this shared directory, the thirteen Piper voices cost 18 MB between
#: them instead of 234 MB.
SHARED_ESPEAK = "espeak-ng-data"


#: Kokoro v1_0's speakers, name -> id, read from the model's own ONNX
#: metadata (``speaker2id``). Embedded so config.yaml can say ``bf_emma``
#: instead of ``21``: nobody should have to look up a magic number to change
#: how their assistant sounds, and the id ordering is an implementation
#: detail of the checkpoint.
#:
#: The prefix is the language and gender: ``af_``/``am_`` American English,
#: ``bf_``/``bm_`` British, ``ef_``/``em_`` Spanish, ``ff_`` French, ``hf_``/
#: ``hm_`` Hindi, ``if_``/``im_`` Italian, ``jf_``/``jm_`` Japanese, ``pf_``/
#: ``pm_`` Portuguese, ``zf_``/``zm_`` Chinese.
KOKORO_SPEAKERS: dict[str, int] = {
    "af_alloy": 0, "af_aoede": 1, "af_bella": 2, "af_heart": 3,
    "af_jessica": 4, "af_kore": 5, "af_nicole": 6, "af_nova": 7,
    "af_river": 8, "af_sarah": 9, "af_sky": 10, "am_adam": 11,
    "am_echo": 12, "am_eric": 13, "am_fenrir": 14, "am_liam": 15,
    "am_michael": 16, "am_onyx": 17, "am_puck": 18, "am_santa": 19,
    "bf_alice": 20, "bf_emma": 21, "bf_isabella": 22, "bf_lily": 23,
    "bm_daniel": 24, "bm_fable": 25, "bm_george": 26, "bm_lewis": 27,
    "ef_dora": 28, "em_alex": 29, "ff_siwis": 30, "hf_alpha": 31,
    "hf_beta": 32, "hm_omega": 33, "hm_psi": 34, "if_sara": 35,
    "im_nicola": 36, "jf_alpha": 37, "jf_gongitsune": 38, "jf_nezumi": 39,
    "jf_tebukuro": 40, "jm_kumo": 41, "pf_dora": 42, "pm_alex": 43,
    "pm_santa": 44, "zf_xiaobei": 45, "zf_xiaoni": 46, "zf_xiaoxiao": 47,
    "zf_xiaoyi": 48, "zm_yunjian": 49, "zm_yunxi": 50, "zm_yunxia": 51,
    "zm_yunyang": 52,
}

#: The archive each named engine lives in, so config can say ``engine: kokoro``
#: without naming a release asset.
ENGINE_ARCHIVES = {
    "kokoro": ("sherpa-kokoro", "kokoro-multi-lang-v1_0", "model.onnx"),
}


@dataclass(frozen=True)
class VoiceSpec:
    """One language's voice: what plays it, which file, and how good it is.

    ``quality`` is the honest tier, not marketing. It is surfaced in the HUD
    and in ``--demo`` so a language that only has a weak voice available says
    so rather than quietly sounding worse than its neighbours.
    """

    engine: str            # "sherpa-vits" | "sherpa-kokoro" | "edge"
    archive: str           # sherpa-onnx release asset name, or the edge voice
    model: str = ""        # the .onnx inside the archive; "" => derive it
    quality: str = "medium"   # "high" | "medium" | "low" | "cloud"
    speaker: int = 0       # multi-speaker models (Kokoro) pick a voice by id
    #: espeak voice for the phonemizer. Only MULTI-language models need it: a
    #: Piper/mimic3 voice carries its own in the .onnx.json, but Kokoro is one
    #: model for nine languages and defaults to en-us. Left unset, Japanese was
    #: phonemized as English — 12.9 s of audio for a sentence that takes 4.2 s,
    #: i.e. the "reads it in the wrong language" bug, just less obviously.
    lang: str = ""
    #: True when this is a stand-in from another language rather than a real
    #: voice for this one. Never selected automatically — it exists so the
    #: offline choice for Macedonian is a documented option, not a surprise.
    fallback: bool = False


#: The map. Voice choices favour a clear, single-speaker, female-or-neutral
#: read at the highest tier each language actually has.
VOICES: dict[str, VoiceSpec] = {
    # --- Kokoro for English -------------------------------------------------
    # Piper's lessac was here and it sounds like a 2010 satnav. That is not a
    # tuning problem: Piper is a small 2021-era VITS model chosen for being
    # tiny and fast, and flat delivery is what it trades away. Kokoro costs
    # ~8x the synthesis time (410 ms -> 3.3 s on a 6.8 s sentence, RTF 0.061
    # -> 0.48) and is still twice as fast as real time, so streaming stays
    # ahead of playback. Worth a second of startup not to sound robotic.
    "en": VoiceSpec("sherpa-kokoro", "kokoro-multi-lang-v1_0",
                    "model.onnx", "high", speaker=KOKORO_SPEAKERS["bf_emma"],
                    lang="en"),
    # --- Piper (VITS), local, CPU ------------------------------------------
    # thorsten-MEDIUM, not -high. Measured on the same sentence: high took
    # 966 ms against medium's 181 ms for identical audio length — 5.3x, on the
    # one number that decides how long MEDO stands there before it starts
    # talking. (Not a rule about tiers: es/pt miro-HIGH run at 179/228 ms, so
    # they stay high. It is this specific model that is heavy.)
    "de": VoiceSpec("sherpa-vits", "vits-piper-de_DE-thorsten-medium",
                    "de_DE-thorsten-medium.onnx", "medium"),
    "fr": VoiceSpec("sherpa-vits", "vits-piper-fr_FR-siwis-medium",
                    "fr_FR-siwis-medium.onnx", "medium"),
    "es": VoiceSpec("sherpa-vits", "vits-piper-es_ES-miro-high",
                    "es_ES-miro-high.onnx", "high"),
    "it": VoiceSpec("sherpa-vits", "vits-piper-it_IT-paola-medium",
                    "it_IT-paola-medium.onnx", "medium"),
    "pt": VoiceSpec("sherpa-vits", "vits-piper-pt_PT-miro-high",
                    "pt_PT-miro-high.onnx", "high"),
    "nl": VoiceSpec("sherpa-vits", "vits-piper-nl_NL-ronnie-medium",
                    "nl_NL-ronnie-medium.onnx", "medium"),
    "pl": VoiceSpec("sherpa-vits", "vits-piper-pl_PL-darkman-medium",
                    "pl_PL-darkman-medium.onnx", "medium"),
    "ru": VoiceSpec("sherpa-vits", "vits-piper-ru_RU-irina-medium",
                    "ru_RU-irina-medium.onnx", "medium"),
    "tr": VoiceSpec("sherpa-vits", "vits-piper-tr_TR-fahrettin-medium",
                    "tr_TR-fahrettin-medium.onnx", "medium"),
    # The ONLY Greek voice Piper publishes, and it is a "low" model. Marked
    # honestly rather than left to sound mysteriously worse than the others.
    "el": VoiceSpec("sherpa-vits", "vits-piper-el_GR-rapunzelina-low",
                    "el_GR-rapunzelina-low.onnx", "low"),
    "zh": VoiceSpec("sherpa-vits", "vits-piper-zh_CN-huayan-medium",
                    "zh_CN-huayan-medium.onnx", "medium"),
    "hi": VoiceSpec("sherpa-vits", "vits-piper-hi_IN-priyamvada-medium",
                    "hi_IN-priyamvada-medium.onnx", "medium"),
    # --- the two Piper has no voice for at all -----------------------------
    # Kokoro **v1_0**, not v1_1. The newer archive is "Kokoro v1.1-zh" and its
    # 103 speakers are all af_/bf_ (English) and zf_/zm_ (Chinese) — read off
    # the ONNX metadata, after picking it on a summary that claimed Kokoro
    # covered ja and ko. v1_0 carries 53 speakers across nine languages, of
    # which jf_alpha (37) is the Japanese female read.
    "ja": VoiceSpec("sherpa-kokoro", "kokoro-multi-lang-v1_0",
                    "model.onnx", "medium", speaker=37, lang="ja"),
    # Korean is the thinnest language MEDO speaks. Kokoro has no Korean
    # speaker in EITHER version, Piper publishes no Korean voice, and this
    # mimic3 model is the only Korean TTS in sherpa-onnx's 272 — at the "low"
    # tier. Marked honestly; it is the fine-tune candidate after Macedonian.
    "ko": VoiceSpec("sherpa-vits", "vits-mimic3-ko_KO-kss_low",
                    "ko_KO-kss_low.onnx", "low"),
    # --- the one cloud exception ------------------------------------------
    "mk": VoiceSpec("edge", "mk-MK-MarijaNeural", quality="cloud"),
}

#: Registered, never auto-selected: the offline stand-in for Macedonian.
#: Serbian is the nearest Cyrillic Slavic voice Piper ships. It will sound
#: Serbian — that is the point of making it an explicit choice.
OFFLINE_FALLBACKS: dict[str, VoiceSpec] = {
    "mk": VoiceSpec("sherpa-vits", "vits-piper-sr_RS-serbski_institut-medium",
                    "sr_RS-serbski_institut-medium.onnx", "low", fallback=True),
}


def from_config(code: str, spec: dict) -> VoiceSpec | None:
    """One ``tts.voices`` entry from config.yaml -> a :class:`VoiceSpec`.

    Three shapes, in the order people reach for them::

        en: {engine: kokoro, voice: bf_emma}       # a Kokoro speaker by NAME
        de: {engine: piper,  voice: de_DE-thorsten-high}
        mk: {engine: edge,   voice: mk-MK-AleksandarNeural}

    Returns None for an entry that names nothing usable, so one typo costs
    that language its override rather than crashing the assistant at startup.
    """
    if not isinstance(spec, dict):
        return None
    engine = str(spec.get("engine") or "").strip().lower()
    voice = str(spec.get("voice") or "").strip()
    quality = str(spec.get("quality") or "")
    if not voice:
        return None
    if engine in ("kokoro", "sherpa-kokoro"):
        kind, archive, model = ENGINE_ARCHIVES["kokoro"]
        speaker = KOKORO_SPEAKERS.get(voice)
        if speaker is None:
            if not voice.isdigit():
                logger.warning("unknown Kokoro voice %r for %s — keeping the "
                               "built-in one", voice, code)
                return None
            speaker = int(voice)
        lang = str(spec.get("lang") or _KOKORO_LANGS.get(voice[:2], ""))
        return VoiceSpec(kind, archive, model, quality or "high",
                         speaker=speaker, lang=lang)
    if engine in ("piper", "vits", "sherpa-vits"):
        # Accept either the bare voice name or the full release-asset name.
        archive = voice if voice.startswith("vits-") else f"vits-piper-{voice}"
        model = str(spec.get("model") or f"{voice.split('/')[-1]}.onnx")
        return VoiceSpec("sherpa-vits", archive, model, quality or "medium",
                         lang=str(spec.get("lang") or ""))
    if engine == "edge":
        return VoiceSpec("edge", voice, quality=quality or "cloud")
    logger.warning("unknown tts.voices engine %r for %s", engine, code)
    return None


#: Which espeak voice a Kokoro speaker prefix implies, so config only has to
#: name the speaker. Without the right one Kokoro phonemizes as en-us — the
#: bug that made Japanese three times too long.
#: Every code here was checked against the espeak-ng-data actually shipped
#: (lang/*/<code>), not guessed. "en-gb" is NOT one of them — espeak's base
#: "en" IS British, and asking for en-gb fails the whole synthesis with
#: "Failed to set eSpeak-ng voice", which surfaces as a silent voice.
_KOKORO_LANGS = {
    "af": "en-us", "am": "en-us", "bf": "en", "bm": "en",
    "ef": "es", "em": "es", "ff": "fr", "hf": "hi", "hm": "hi",
    "if": "it", "im": "it", "jf": "ja", "jm": "ja",
    "pf": "pt-BR", "pm": "pt-BR", "zf": "cmn", "zm": "cmn",
}


def load_voices(overrides: dict | None = None) -> dict[str, VoiceSpec]:
    """:data:`VOICES` with any ``tts.voices`` entries from config applied.

    Same contract as ``skills.sites.load_sites``: a malformed entry is skipped
    with a warning rather than taken as a reason to have no voices at all.
    """
    merged = dict(VOICES)
    for code, spec in (overrides or {}).items():
        built = from_config(str(code), spec)
        if built is not None:
            merged[str(code).strip().lower()[:2]] = built
    return merged


def spec_for(language: str | None, *, offline_only: bool = False,
             allow_fallback: bool = False,
             table: dict[str, VoiceSpec] | None = None) -> VoiceSpec | None:
    """The voice that speaks ``language``. None when nothing covers it.

    ``offline_only`` refuses the cloud engine. ``allow_fallback`` — and ONLY
    ``allow_fallback`` — permits the registered stand-in from another language.

    These are two flags rather than one because folding them together is a
    bug I shipped and caught: with the stand-in returned automatically
    whenever the cloud was refused, the local provider answered "yes, I can
    speak Macedonian" and would have read it aloud in a Serbian voice without
    anyone choosing that. None is the honest answer — the caller shows the
    reply instead of voicing it in the wrong language.
    """
    code = (language or "").strip().lower()[:2]
    spec = (table if table is not None else VOICES).get(code)
    if spec is None:
        return None
    if offline_only and spec.engine == "edge":
        return OFFLINE_FALLBACKS.get(code) if allow_fallback else None
    return spec


def model_path(root: Path, spec: VoiceSpec) -> Path:
    """Where ``spec``'s .onnx lives once downloaded."""
    return root / spec.archive / (spec.model or f"{spec.archive}.onnx")


def is_installed(root: Path, spec: VoiceSpec) -> bool:
    """Is this voice already on disk and usable?"""
    if spec.engine == "edge":
        return True                       # nothing to install; it's a service
    return model_path(root, spec).exists() and (root / SHARED_ESPEAK).is_dir()


def download(root: Path, spec: VoiceSpec, *, timeout: float = 300.0) -> Path:
    """Fetch and unpack one voice, returning the path to its .onnx.

    Downloads to a temp file and only moves the unpacked directory into place
    at the end, so an interrupted download can never leave a half-voice that
    ``is_installed`` would then vouch for.
    """
    root.mkdir(parents=True, exist_ok=True)
    target = root / spec.archive
    url = RELEASE_URL.format(archive=spec.archive)
    logger.info("downloading voice %s", spec.archive)
    with tempfile.TemporaryDirectory(dir=str(root)) as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "voice.tar.bz2"
        with urlopen(url, timeout=timeout) as response, \
                archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(tmp_path, filter="data")
        unpacked = tmp_path / spec.archive
        if not unpacked.is_dir():          # some archives use a different root
            candidates = [p for p in tmp_path.iterdir() if p.is_dir()]
            unpacked = candidates[0] if candidates else unpacked
        # Hoist espeak-ng-data out to the shared copy, once.
        espeak = unpacked / SHARED_ESPEAK
        shared = root / SHARED_ESPEAK
        if espeak.is_dir():
            if not shared.is_dir():
                shutil.move(str(espeak), str(shared))
            else:
                shutil.rmtree(espeak, ignore_errors=True)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(unpacked), str(target))
    return model_path(root, spec)


def ensure(root: Path, spec: VoiceSpec) -> Path | None:
    """The voice's .onnx, downloading it once if needed. None on failure.

    Never raises: a language whose voice won't download must cost that
    language its voice, not the whole turn.
    """
    if spec.engine == "edge":
        return None
    if is_installed(root, spec):
        return model_path(root, spec)
    try:
        return download(root, spec)
    except Exception:
        logger.warning("couldn't fetch voice %s", spec.archive, exc_info=True)
        return None
