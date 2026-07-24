#!/usr/bin/env python3
"""Capture REAL wake-word audio from your mic, to sharpen the trained model.

Synthetic TTS training makes the model fire on any clean speech because it never
heard YOUR voice, mic, or room. This records a handful of real "medo" positives
and a couple of minutes of your normal speech as negatives, so a retrain can
learn "medo vs the way *I* actually talk" instead of "speech vs silence".

    # 1) capture (say "medo" ~20x, then just talk for ~90s):
    .venv\\Scripts\\python.exe scripts\\capture_wake_samples.py

    # 2) retrain WITH the real clips folded in:
    #    (features stage — MEDO venv)
    .venv\\Scripts\\python.exe scripts\\train_wakeword_deep.py --stage features \\
        --offline --real-dir models\\wakeword\\real --workdir D:\\wakeword_train\\run3
    #    (train stage — torch venv)  D:\\medo-sim\\.venv\\Scripts\\python.exe ...

Uses the SAME mic/device MEDO uses (from config.yaml), at 16 kHz mono, so the
captured acoustics match what the wake detector sees at runtime.
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SR = 16000
CHUNK_S = 2.5


def _write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = np.clip(audio, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def _record_seconds(mic, seconds: float) -> np.ndarray:
    need = int(seconds * SR)
    frames, got = [], 0
    while got < need:
        f = mic.read_frame()
        frames.append(f)
        got += len(f)
    return np.concatenate(frames)[:need]


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture real wake-word samples")
    ap.add_argument("--out", default=str(ROOT / "models" / "wakeword" / "real"))
    ap.add_argument("--positives", type=int, default=20)
    ap.add_argument("--neg-seconds", type=int, default=90, dest="neg_seconds")
    ap.add_argument("--pos-window", type=float, default=2.0, dest="pos_window")
    args = ap.parse_args()

    from core.config import load_settings
    from voice.audio import Microphone

    s = load_settings()
    device = s.audio.input_device
    out = Path(args.out)
    (out / "pos").mkdir(parents=True, exist_ok=True)
    (out / "neg").mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(" MEDO wake-word capture — using your configured mic")
    print(f" device: {device if device is not None else 'system default'}")
    print("=" * 64)

    mic = Microphone(SR, device=device)
    mic.open()
    try:
        # --- warm-up level check so a dead/quiet mic is obvious immediately ----
        print("\nQuick mic check — say anything for 1 second...")
        chk = _record_seconds(mic, 1.0)
        peak = float(np.abs(chk).max()) / 32768.0
        print(f"  peak level: {peak:.3f}  " + (
            "OK" if peak > 0.02 else "!! VERY LOW — check your mic/device before continuing"))

        # --- positives --------------------------------------------------------
        print(f"\n--- POSITIVES ({args.positives}) — say the wake word each time ---")
        print("Say it the way you naturally would. Vary distance/loudness a bit.\n")
        made = 0
        for i in range(args.positives):
            phrase = "hey medo" if (i % 3 == 2) else "medo"   # ~2/3 bare, ~1/3 hey
            input(f"[{i + 1}/{args.positives}] Press Enter, then say  >>> {phrase.upper()} <<<")
            print(f"    recording {args.pos_window:.1f}s — say it now...")
            audio = _record_seconds(mic, args.pos_window)
            pk = float(np.abs(audio).max()) / 32768.0
            if pk < 0.02:
                print("    (that was near-silent — not saved, let's redo this one)")
                # redo the same index
                i_redo = i
                while pk < 0.02:
                    input(f"[{i_redo + 1}/{args.positives}] Press Enter, then say  >>> {phrase.upper()} <<<")
                    print(f"    recording {args.pos_window:.1f}s — say it now...")
                    audio = _record_seconds(mic, args.pos_window)
                    pk = float(np.abs(audio).max()) / 32768.0
            _write_wav(out / "pos" / f"pos_{made:03d}_{phrase.replace(' ', '-')}.wav", audio)
            made += 1
            print(f"    saved (peak {pk:.2f})")
        print(f"\n  {made} positive clips -> {out / 'pos'}")

        # --- negatives --------------------------------------------------------
        print(f"\n--- NEGATIVES — talk normally for ~{args.neg_seconds}s, do NOT say 'medo' ---")
        print("Read something aloud, describe your day, whatever — just natural speech")
        print("in this room. This teaches the model what your ordinary talk sounds like.")
        input("Press Enter to start the negative recording...")
        print(f"    recording {args.neg_seconds}s — talk now...")
        neg = _record_seconds(mic, args.neg_seconds)
        step = int(CHUNK_S * SR)
        n = 0
        for start in range(0, len(neg) - step, step // 2):     # 50% overlap = more clips
            _write_wav(out / "neg" / f"neg_{n:04d}.wav", neg[start:start + step])
            n += 1
        print(f"    {n} negative clips -> {out / 'neg'}")
    finally:
        mic.close()

    print("\n" + "=" * 64)
    print(" Capture done. Tell Claude and it will retrain with these clips.")
    print(f"   positives: {made}   negatives: {n}")
    print("=" * 64)


if __name__ == "__main__":
    main()
