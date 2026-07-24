"""Wake-word diagnosis: capture every activation, then report the scores.

The false-activation bug ("MEDO wakes on door slams") can't be tuned blind —
you have to see what a slam actually scores versus a real "hey medo". So with
``wakeword.debug_capture`` on, every activation writes a row (time, score, RMS)
to ``logs/wake_captures/activations.jsonl`` and saves the ~1.5 s buffer that
triggered it as a ``.wav`` next to it. Slam doors and say the wake word for a
day; then label the wavs and ``python -m voice.wakeword --report`` shows the two
score clusters so a threshold can be chosen from data.

Pure and hardware-free: the loop hands ``record`` an audio array, and the
report reads back plain files — so all of it is unit-testable. Uses the stdlib
``wave`` module (no new dependency) for the .wav.
"""

from __future__ import annotations

import json
import logging
import statistics
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_LOG_NAME = "activations.jsonl"
_LABELS_NAME = "labels.csv"      # optional: "wav_name,true|false" the user fills in


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono int16 PCM. ``audio`` is coerced to int16 (frames are already
    int16 PCM; a float array would be silence-quiet, so we assume int16)."""
    pcm = np.asarray(audio).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


class WakeCaptureLog:
    """Records wake activations to disk when enabled; a no-op when not."""

    def __init__(self, directory: str | Path, enabled: bool = False,
                 sample_rate: int = 16000) -> None:
        self.directory = Path(directory)
        self.enabled = bool(enabled)
        self.sample_rate = sample_rate
        self._log = self.directory / _LOG_NAME

    def record(self, score: float, rms: float, audio: np.ndarray,
               when: datetime | None = None) -> Path | None:
        """Save one activation (wav + a json row). None when disabled.

        ``when`` is injectable so tests are deterministic.
        """
        if not self.enabled:
            return None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            when = when or datetime.now()
            stamp = when.strftime("%Y%m%d-%H%M%S-") + f"{when.microsecond // 1000:03d}"
            wav = self.directory / f"wake-{stamp}.wav"
            write_wav(wav, audio, self.sample_rate)
            row = {
                "time": when.isoformat(timespec="milliseconds"),
                "score": round(float(score), 4),
                "rms": round(float(rms), 4),
                "wav": wav.name,
            }
            with self._log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            return wav
        except Exception:      # diagnosis must never crash the voice loop
            logger.warning("wake capture failed", exc_info=True)
            return None

    def rows(self) -> list[dict]:
        if not self._log.exists():
            return []
        out = []
        for line in self._log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


def load_labels(directory: str | Path) -> dict[str, bool]:
    """Read the optional ``labels.csv`` (``wav_name,true|false``) the user fills
    in after listening back. Missing file -> no labels."""
    path = Path(directory) / _LABELS_NAME
    if not path.exists():
        return {}
    labels: dict[str, bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("wav"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            labels[parts[0]] = parts[1].lower() in ("true", "1", "yes", "real", "wake")
    return labels


def _stats(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0}
    s = sorted(scores)
    return {
        "n": len(s), "min": min(s), "max": max(s),
        "median": statistics.median(s),
        "mean": statistics.fmean(s),
    }


def summarize(rows: list[dict], labels: dict[str, bool] | None = None) -> dict:
    """Aggregate captured activations into a score picture.

    Without labels: overall score distribution. With labels: the TRUE (real
    wake) and FALSE (noise) clusters separately, plus a suggested threshold
    that sits above every false score and below the real ones — and the
    false-negative RISK if the clusters overlap.
    """
    labels = labels or {}
    overall = _stats([r["score"] for r in rows])
    result = {"overall": overall, "labelled": bool(labels)}
    if not labels:
        return result

    true_scores = [r["score"] for r in rows if labels.get(r["wav"]) is True]
    false_scores = [r["score"] for r in rows if labels.get(r["wav"]) is False]
    result["true"] = _stats(true_scores)
    result["false"] = _stats(false_scores)
    if true_scores and false_scores:
        hi_false = max(false_scores)
        lo_true = min(true_scores)
        if lo_true > hi_false:
            # clean separation: threshold in the gap (nudged toward the noise)
            result["suggested_threshold"] = round(hi_false + (lo_true - hi_false) * 0.5, 3)
            result["separable"] = True
            result["false_negative_risk"] = "none — the clusters don't overlap"
        else:
            # overlap: pick just above the loud-noise cluster and report the cost
            result["suggested_threshold"] = round(hi_false + 0.001, 3)
            result["separable"] = False
            missed = sum(1 for s in true_scores if s <= hi_false)
            result["false_negative_risk"] = (
                f"{missed}/{len(true_scores)} real wake words would be missed at "
                f"this threshold — clusters overlap; VAD/verifier (S3/S4) needed")
    return result


def format_report(rows: list[dict], labels: dict[str, bool] | None = None) -> str:
    """A human-readable ``--report``."""
    s = summarize(rows, labels)
    o = s["overall"]
    lines = ["", "Wake-word activation report", "=" * 40]
    if not o["n"]:
        lines.append("No captures yet. Set wakeword.debug_capture: true, make "
                     "some noise + say the wake word, then re-run.")
        return "\n".join(lines)
    lines.append(f"{o['n']} activation(s) captured.")
    lines.append(f"  score: min {o['min']:.3f} · median {o['median']:.3f} · "
                 f"max {o['max']:.3f}")
    if not s["labelled"]:
        lines += ["",
                  "Label them to separate real from noise: create labels.csv in",
                  "the captures folder with lines like  wake-YYYY...wav,false",
                  "(false = a noise that shouldn't have woken it), then re-run."]
        return "\n".join(lines)
    t, f = s["true"], s["false"]
    lines.append(f"  REAL wake  ({t.get('n',0)}): "
                 + (f"min {t['min']:.3f} · median {t['median']:.3f}" if t.get('n') else "none labelled"))
    lines.append(f"  NOISE      ({f.get('n',0)}): "
                 + (f"max {f['max']:.3f} · median {f['median']:.3f}" if f.get('n') else "none labelled"))
    if "suggested_threshold" in s:
        lines += ["", f"  suggested threshold: {s['suggested_threshold']}",
                  f"  false-negative risk: {s['false_negative_risk']}"]
    return "\n".join(lines)
