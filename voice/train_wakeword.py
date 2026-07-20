"""Turnkey helper for a custom "hey MEDO" wake word.

The MEDO code already supports a custom model (``wakeword.phrase`` pointing at
a ``.onnx`` file — see ``voice/wakeword.py``); the only missing piece is the
trained model. This script does the tedious part — generating a diverse set of
synthetic positive samples — so the actual training run (local or Colab) is
one step.

    # 1) generate a diverse "hey medo" sample set (edge-tts many voices + Piper)
    .venv\\Scripts\\python -m voice.train_wakeword --generate

    # 2) train:
    #    - locally IF the openWakeWord training extras + background data exist:
    .venv\\Scripts\\python -m voice.train_wakeword --train
    #    - otherwise upload the samples to the openWakeWord Colab notebook
    #      (the script prints the exact steps; see docs/Wake Word Training.md).

    # 3) drop hey_medo.onnx in models/wakeword/ and flip config.yaml:
    #      wakeword.phrase: "models/wakeword/hey_medo.onnx"

Diversity matters: a wake model trained on ONE voice overfits, so we synthesize
"hey medo" (and pronunciation variants) across many edge-tts voices at varied
speaking rates. edge-tts is online; Piper is the offline fallback.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import wave
from pathlib import Path

import numpy as np

from core.config import PROJECT_ROOT, load_settings

logger = logging.getLogger("train_wakeword")

SAMPLE_RATE = 16000  # openWakeWord trains on 16 kHz mono

#: Phrase spellings — MEDO isn't an English word, so give the TTS voices a few
#: pronunciations to cover how it's actually said.
DEFAULT_SPELLINGS = ["hey medo", "hey meh doh", "hey may doh", "hey meadow"]

#: A spread of edge-tts English voices (accents + genders) for diversity.
EDGE_VOICES = [
    "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
    "en-US-ChristopherNeural", "en-US-EricNeural", "en-US-MichelleNeural",
    "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-AU-NatashaNeural",
    "en-AU-WilliamNeural", "en-IE-ConnorNeural", "en-CA-LiamNeural",
]


def _to_float(audio: np.ndarray) -> np.ndarray:
    """int16 (what Piper/edge-tts return) or float -> float32 in [-1, 1]."""
    a = np.asarray(audio)
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.float32) / 32768.0
    return a.astype(np.float32)


def _resample(audio: np.ndarray, sr_in: int) -> np.ndarray:
    a = _to_float(audio)
    if sr_in == SAMPLE_RATE or a.size == 0:
        return a
    n = int(len(a) * SAMPLE_RATE / sr_in)
    x_in = np.linspace(0.0, 1.0, len(a), endpoint=False)
    x_out = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(x_out, x_in, a).astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


async def _edge_samples(out: Path, spellings: list[str]) -> int:
    """Synthesize positives across many edge-tts voices/rates. Returns count."""
    try:
        from voice.tts import EdgeTTS
    except Exception as exc:
        logger.warning("edge-tts unavailable (%s) — skipping online voices", exc)
        return 0
    made = 0
    for voice in EDGE_VOICES:
        try:
            tts = EdgeTTS(voice)
        except Exception:
            continue
        for spelling in spellings:
            try:
                wav, sr = await tts.synthesize(spelling)
            except Exception:
                continue
            audio = _resample(wav, sr)
            if audio.size:
                name = f"edge_{voice}_{spelling.replace(' ', '-')}_{made}.wav"
                _write_wav(out / name, audio)
                made += 1
    return made


def _piper_samples(out: Path, spellings: list[str]) -> int:
    """Offline Piper positives (single voice; diversity is limited)."""
    try:
        from voice.tts import TextToSpeech

        tts = TextToSpeech(load_settings().tts)
    except Exception as exc:
        logger.warning("Piper unavailable (%s) — skipping offline voice", exc)
        return 0
    made = 0
    for spelling in spellings:
        try:
            wav, sr = tts.synthesize(spelling)
        except Exception:
            continue
        audio = _resample(wav, sr)
        if audio.size:
            _write_wav(out / f"piper_{spelling.replace(' ', '-')}_{made}.wav", audio)
            made += 1
    return made


def generate(out_dir: Path, spellings: list[str], online: bool = True) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = _piper_samples(out_dir, spellings)
    if online:
        total += asyncio.run(_edge_samples(out_dir, spellings))
    print(f"generated {total} positive sample(s) in {out_dir}")
    return total


def _training_hint(samples: Path) -> None:
    print(
        "\n--- training ---\n"
        "Full openWakeWord training needs the background/negative feature\n"
        "datasets (several GB) and ~1 h on a GPU — easiest on the free Colab:\n"
        "  https://github.com/dscripka/openWakeWord "
        "-> notebooks/automatic_model_training.ipynb\n"
        f"  * upload the positives from: {samples}\n"
        "  * set the target phrase to 'hey medo'\n"
        "  * export hey_medo.onnx, drop it in models/wakeword/, then set\n"
        "    wakeword.phrase: \"models/wakeword/hey_medo.onnx\" in config.yaml\n"
        "  * re-tune wakeword.threshold with:  python -m voice.wakeword\n"
        "See docs/Wake Word Training.md for the full runbook.\n"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Custom 'hey MEDO' wake-word helper")
    ap.add_argument("--generate", action="store_true", help="synthesize positive samples")
    ap.add_argument("--train", action="store_true", help="attempt local training (or print steps)")
    ap.add_argument("--offline", action="store_true", help="Piper only, no edge-tts")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "models" / "wakeword" / "samples"))
    args = ap.parse_args()
    samples = Path(args.out)

    if args.generate or not args.train:
        generate(samples, DEFAULT_SPELLINGS, online=not args.offline)

    if args.train:
        try:
            import openwakeword.train  # noqa: F401
            # The training entry point needs the negative feature datasets to be
            # present; we don't bundle those (multi-GB). Point the user at the
            # notebook rather than silently doing a bad single-source train.
            print("openWakeWord training module found, but the background/negative")
            print("datasets are not bundled here.")
        except Exception:
            pass
        _training_hint(samples)


if __name__ == "__main__":
    main()
