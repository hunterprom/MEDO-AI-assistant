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
    # --- Piper (VITS), local, CPU ------------------------------------------
    "en": VoiceSpec("sherpa-vits", "vits-piper-en_US-lessac-medium",
                    "en_US-lessac-medium.onnx", "medium"),
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


def spec_for(language: str | None, *, offline_only: bool = False,
             allow_fallback: bool = False) -> VoiceSpec | None:
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
    spec = VOICES.get(code)
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
