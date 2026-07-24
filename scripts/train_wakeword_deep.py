#!/usr/bin/env python3
"""Deep, two-stage LOCAL training for the "medo" / "hey medo" wake word.

Why two stages (and two Python envs): openWakeWord's feature extractor lives in
the MEDO runtime venv, but torch (~GBs) can't fit on the near-full system drive
here — it lives in a separate venv on D:. So:

  STAGE 1  --stage features   (run with the MEDO venv)
    Generate a DIVERSE positive set for BOTH "medo" and "hey medo" (many
    edge-tts voices + Piper), plus hard negatives (similar words: meadow, memo,
    nemo, metro…) and a big synthetic noise pool. Augment (gain / noise /
    time-stretch / background mixing), extract openWakeWord embeddings, split
    train/val, and write one .npz feature bundle.

  STAGE 2  --stage train      (run with a torch venv)
    Load the .npz, SWEEP hyperparameters, MINE hard negatives from the pool
    (score it, fold the worst false-firers back in), retrain, and pick the head
    with the best held-out recall at a low false-positive rate. Export
    hey_medo.onnx (legacy exporter — loads unchanged in openWakeWord).

The two stages talk only through the .npz, so the envs never have to share deps.

Design notes that matter for it to actually work live:
  * Features are extracted EXACTLY as the runtime does (openWakeWord melspec ->
    embedding -> first 16 frames of 96-dim), so a head trained here drops into
    voice/wakeword.py unchanged.
  * The OLD model was trained with bare "medo" as a NEGATIVE and negatives
    weighted 6x — which is precisely why it ignored a spoken "medo". Here "medo"
    is a positive, and neg-weight is chosen by held-out validation, not fixed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Run directly (python scripts/train_wakeword_deep.py) puts scripts/ on sys.path,
# not the project root — so `from core.config import …` in the features stage
# would ImportError and get swallowed. Put the project root first.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SR = 16000
N_FRAMES = 16          # model input is (16, 96)
CLIP_S = 2.5           # long enough to yield >=16 embedding frames

# Positives split into BARE ("medo") and HEY ("hey medo") so the two can be
# COUNT-BALANCED (the seed samples are all "hey medo" and would otherwise swamp
# the bare form the user actually says). Multiple phonetic spellings give Piper
# a few distinct renderings of the made-up word.
BARE_SPELLINGS = ["medo", "meh doh", "may doh", "medoh", "meddo", "meh dow", "maydoh"]
HEY_SPELLINGS = ["hey medo", "hey meh doh", "hey may doh", "hey medoh", "hey meddo"]
# HARD negatives: real words that sound close to "medo" (teach fine
# discrimination) — the single most important negatives for precision.
NEG_PHRASES = [
    "meadow", "memo", "nemo", "medium", "metro", "motto", "tempo", "meddle",
    "medic", "model", "meta", "menu", "mellow", "mojo", "meadows", "medoc",
    "hey medic", "hey model", "hey there", "hey buddy", "hey mister", "hey now",
    "meadow so", "let me go", "made a", "me too", "the metro", "play media",
]
# DIVERSE speech negatives: a broad spread of ordinary words/phrases so the head
# learns "medo vs OTHER SPEECH", not just "speech vs silence". Without a big
# background-speech corpus (the multi-GB datasets openWakeWord normally uses),
# this synthetic spread is what stops the model firing on any clean utterance.
DIVERSE_NEG = [
    # commands / assistant-y phrases it will hear but must ignore
    "what time is it", "set a timer", "play some music", "turn off the lights",
    "open the browser", "close the window", "take a screenshot", "search the web",
    "what's the weather", "read my email", "call my mom", "send a message",
    "volume up", "next track", "pause the video", "scroll down", "go back",
    "tell me a joke", "how are you", "good morning", "good night", "thank you",
    "yes please", "no thanks", "never mind", "cancel that", "start over",
    # common nouns
    "computer", "keyboard", "monitor", "window", "coffee", "water", "table",
    "phone", "camera", "picture", "music", "movie", "garden", "kitchen",
    "morning", "evening", "weather", "traffic", "message", "calendar", "reminder",
    "tomato", "potato", "banana", "letter", "number", "color", "animal",
    "dog", "cat", "bird", "house", "street", "city", "country", "planet",
    # verbs / adjectives
    "running", "walking", "talking", "reading", "writing", "playing", "working",
    "happy", "hungry", "tired", "ready", "better", "faster", "louder", "quiet",
    # numbers / letters
    "one two three", "four five six", "seven eight nine", "ten eleven twelve",
    "a b c", "x y z", "hello world", "okay then", "all right", "here we go",
    # short sentences
    "i don't know", "let me think", "that sounds good", "can you help me",
    "where are we going", "what should i do", "it's getting late",
    "the sun is out", "i need a break", "let's get started", "see you later",
    "have a good day", "talk to you soon", "on my way", "just a second",
    "hold on a moment", "give me a minute", "not right now", "maybe later",
    "sounds like a plan", "i'll be there", "leave it alone", "keep it simple",
]
# A spread of edge-tts English voices (accents + genders) for diversity.
EDGE_VOICES = [
    "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
    "en-US-ChristopherNeural", "en-US-EricNeural", "en-US-MichelleNeural",
    "en-US-RogerNeural", "en-US-SteffanNeural", "en-US-AnaNeural",
    "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-GB-LibbyNeural",
    "en-GB-MaisieNeural", "en-GB-ThomasNeural",
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    "en-IE-ConnorNeural", "en-IE-EmilyNeural",
    "en-CA-LiamNeural", "en-CA-ClaraNeural",
    "en-NZ-MitchellNeural", "en-NZ-MollyNeural",
    "en-ZA-LukeNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
]


# --- audio helpers (identical framing to scripts/local_train_hey_medo.py) -----

def _read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        n, sr = w.getnframes(), w.getframerate()
        a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if sr != SR and a.size:
        m = int(len(a) * SR / sr)
        a = np.interp(np.linspace(0, 1, m, endpoint=False),
                      np.linspace(0, 1, len(a), endpoint=False), a).astype(np.float32)
    return a


def _time_stretch(a: np.ndarray, factor: float) -> np.ndarray:
    """Resample-based speed change (also shifts pitch — fine for augmentation)."""
    if a.size == 0 or abs(factor - 1.0) < 1e-3:
        return a
    m = max(1, int(len(a) / factor))
    return np.interp(np.linspace(0, 1, m, endpoint=False),
                     np.linspace(0, 1, len(a), endpoint=False), a).astype(np.float32)


def _fit(a: np.ndarray, rng, jitter=True) -> np.ndarray:
    """Place audio near the START of a fixed CLIP_S buffer (the head trains on
    the first 16 embedding frames, so the phrase must sit at the front). A tiny
    random lead-in is allowed so onset timing varies without pushing the phrase
    out of the window."""
    n = int(CLIP_S * SR)
    buf = np.zeros(n, dtype=np.float32)
    a = a[:n]
    start = 0
    if jitter and n > len(a):
        start = int(rng.integers(0, min(int(0.15 * SR), n - len(a) + 1)))
    buf[start:start + len(a)] = a
    return buf


def _augment(a: np.ndarray, rng, noise_bank=None) -> np.ndarray:
    # Wide speed range: resampling shifts pitch AND rate together, so this is
    # our cheapest way to fake many speakers out of Piper's single voice when
    # edge-tts voices aren't reachable (TLS-intercepted network here).
    a = _time_stretch(a, float(rng.uniform(0.80, 1.20)))          # speed/pitch
    a = a * float(rng.uniform(0.55, 1.0))                          # gain
    a = a + rng.normal(0, float(rng.uniform(0.0, 0.02)), a.shape).astype(np.float32)  # hiss
    if noise_bank is not None and len(noise_bank) and rng.random() < 0.5:
        bg = noise_bank[int(rng.integers(0, len(noise_bank)))]
        if len(bg) >= len(a):
            off = int(rng.integers(0, len(bg) - len(a) + 1))
            a = a + bg[off:off + len(a)] * float(rng.uniform(0.05, 0.35))   # room noise
    return np.clip(a, -1.0, 1.0)


def _windows(embeddings) -> np.ndarray:
    """(N, frames, 96) -> (N, 16, 96): take/pad the first 16 frames."""
    out = []
    for e in embeddings:
        e = np.asarray(e)
        if e.shape[0] >= N_FRAMES:
            out.append(e[:N_FRAMES])
        else:
            pad = np.zeros((N_FRAMES - e.shape[0], e.shape[1]), np.float32)
            out.append(np.vstack([e, pad]))
    return np.asarray(out, dtype=np.float32)


def _noise_bank(rng, n: int) -> list[np.ndarray]:
    """Varied synthetic noise: white/pink/brown, tone bursts, transient claps —
    cheap, no network. Doubles as the hard-negative mining pool."""
    bank = []
    ln = int(CLIP_S * SR)
    for i in range(n):
        kind = i % 6
        if kind == 0:                                   # white
            x = rng.normal(0, rng.uniform(0.02, 0.12), ln)
        elif kind == 1:                                 # pink-ish (cumsum LP)
            x = np.cumsum(rng.normal(0, 1, ln)); x = x / (np.abs(x).max() + 1e-6) * rng.uniform(0.1, 0.4)
        elif kind == 2:                                 # brown (double cumsum)
            x = np.cumsum(np.cumsum(rng.normal(0, 1, ln))); x = x / (np.abs(x).max() + 1e-6) * rng.uniform(0.1, 0.4)
        elif kind == 3:                                 # tone burst
            f = rng.uniform(120, 3000); t = np.arange(ln) / SR
            x = np.sin(2 * np.pi * f * t) * rng.uniform(0.05, 0.3)
        elif kind == 4:                                 # transient claps
            x = np.zeros(ln)
            for _ in range(int(rng.integers(1, 6))):
                p = int(rng.integers(0, ln - 400)); x[p:p + 400] += rng.normal(0, 0.5, 400)
        else:                                           # near-silence
            x = rng.normal(0, rng.uniform(0.001, 0.01), ln)
        bank.append(np.clip(x, -1.0, 1.0).astype(np.float32))
    return bank


# ============================ STAGE 1: features ==============================

def _synth_edge(voice: str, text: str):
    from voice.tts import EdgeTTS
    import asyncio
    tts = EdgeTTS(voice)
    wav, sr = asyncio.run(tts.synthesize(text))
    a = wav.astype(np.float32) / 32768.0 if wav.dtype.kind == "i" else wav.astype(np.float32)
    if sr != SR and a.size:
        m = int(len(a) * SR / sr)
        a = np.interp(np.linspace(0, 1, m, endpoint=False),
                      np.linspace(0, 1, len(a), endpoint=False), a).astype(np.float32)
    return a


def _synth_piper(text: str):
    from core.config import load_settings
    from voice.tts import TextToSpeech
    tts = TextToSpeech(load_settings().tts)
    wav, sr = tts.synthesize(text)
    a = wav.astype(np.float32) / 32768.0 if wav.dtype.kind == "i" else wav.astype(np.float32)
    if sr != SR and a.size:
        m = int(len(a) * SR / sr)
        a = np.interp(np.linspace(0, 1, m, endpoint=False),
                      np.linspace(0, 1, len(a), endpoint=False), a).astype(np.float32)
    return a


def _generate_base(voices, spellings, use_edge=True) -> list[np.ndarray]:
    """One clean clip per (voice, spelling); Piper as an always-there fallback."""
    clips = []
    for sp in spellings:
        try:
            a = _synth_piper(sp)
            if a.size:
                clips.append(a)
        except Exception:
            pass
    if use_edge:
        for v in voices:
            for sp in spellings:
                try:
                    a = _synth_edge(v, sp)
                    if a.size:
                        clips.append(a)
                    time.sleep(0.05)          # be gentle on the edge endpoint
                except Exception:
                    continue
    return clips


def stage_features(args) -> None:
    import openwakeword
    from openwakeword.utils import AudioFeatures

    out_dir = Path(args.workdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    n_voices = 2 if args.smoke else len(EDGE_VOICES)
    bare_spell = BARE_SPELLINGS[:1] if args.smoke else BARE_SPELLINGS
    hey_spell = HEY_SPELLINGS[:1] if args.smoke else HEY_SPELLINGS
    neg_voices = EDGE_VOICES[:2] if args.smoke else EDGE_VOICES[:8]
    aug = 2 if args.smoke else args.aug
    pool_n = 60 if args.smoke else args.pool

    noise_bank = _noise_bank(rng, pool_n)      # augmentation bg + mining pool

    def expand(base, times, is_pos):
        buf = []
        for a in base:
            for _ in range(times):
                buf.append(_fit(_augment(a, rng, noise_bank if is_pos else None), rng))
        return buf

    # Positives: seed WAVs (real multi-voice "hey medo") + Piper renderings of
    # both forms. Count-balance BARE vs HEY so the user's "medo" isn't drowned.
    seed_files = sorted(glob.glob(str(ROOT / "models" / "wakeword" / "samples" / "*.wav")))
    hey_base = [_read_wav(f) for f in seed_files] if not args.smoke else []
    hey_base += _generate_base(EDGE_VOICES[:n_voices], hey_spell, use_edge=not args.offline)
    bare_base = _generate_base(EDGE_VOICES[:n_voices], bare_spell, use_edge=not args.offline)
    hey_aug = aug
    bare_aug = max(aug, int(aug * max(1, len(hey_base)) / max(1, len(bare_base))))
    print(f"[features] positives: hey_base={len(hey_base)} (x{hey_aug})  "
          f"bare_base={len(bare_base)} (x{bare_aug})", flush=True)
    pos_audio = expand(hey_base, hey_aug, True) + expand(bare_base, bare_aug, True)

    # Negatives in two tiers: hard near-misses (full aug) and a broad speech
    # spread (many phrases, lighter aug) so the head learns medo-vs-other-speech.
    hard_spell = NEG_PHRASES[:3] if args.smoke else NEG_PHRASES
    div_spell = DIVERSE_NEG[:3] if args.smoke else DIVERSE_NEG
    hard_base = _generate_base(neg_voices, hard_spell, use_edge=not args.offline)
    div_base = _generate_base(neg_voices[:2], div_spell, use_edge=not args.offline)
    div_aug = max(2, aug // 3)
    print(f"[features] negatives: hard_base={len(hard_base)} (x{aug})  "
          f"div_base={len(div_base)} (x{div_aug})", flush=True)
    neg_audio = expand(hard_base, aug, False) + expand(div_base, div_aug, False)

    # Real captured audio (the user's actual voice/mic/room) — the true target
    # distribution TTS can't reproduce. Over-represent real positives so the
    # head anchors on how "medo" REALLY sounds; add real negatives so it learns
    # to reject the user's ordinary speech, not just synthetic speech.
    if args.real_dir:
        rd = Path(args.real_dir)
        rp = [_read_wav(f) for f in sorted(glob.glob(str(rd / "pos" / "*.wav")))]
        rn = [_read_wav(f) for f in sorted(glob.glob(str(rd / "neg" / "*.wav")))]
        print(f"[features] REAL clips: pos={len(rp)} (x{aug * 3})  neg={len(rn)} (x{max(2, aug // 2)})",
              flush=True)
        pos_audio += expand(rp, aug * 3, True)
        neg_audio += expand(rn, max(2, aug // 2), False)

    # noise/silence negatives straight from the bank + explicit silence
    neg_audio += [_fit(n, rng, jitter=False) for n in noise_bank]
    for _ in range(20 if not args.smoke else 4):
        neg_audio.append(np.zeros(int(CLIP_S * SR), np.float32))
    print(f"[features] positives={len(pos_audio)} negatives={len(neg_audio)}; embedding…", flush=True)

    res = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
    F = AudioFeatures(
        melspec_model_path=os.path.join(res, "melspectrogram.onnx"),
        embedding_model_path=os.path.join(res, "embedding_model.onnx"),
    )

    def _i16(clips):
        return (np.clip(np.asarray(clips), -1.0, 1.0) * 32767).astype(np.int16)

    def embed(clips):
        out = []
        for i in range(0, len(clips), 256):
            out.append(_windows(F.embed_clips(_i16(clips[i:i + 256]), batch_size=32)))
            print(f"    embedded {min(i + 256, len(clips))}/{len(clips)}", flush=True)
        return np.concatenate(out) if out else np.zeros((0, N_FRAMES, 96), np.float32)

    Xp = embed(pos_audio)
    Xn = embed(neg_audio)
    # a separate, larger noise pool purely for hard-negative mining in stage 2
    pool_audio = [_fit(n, rng, jitter=False) for n in _noise_bank(rng, pool_n * 3)]
    Xpool = embed(pool_audio)

    # held-out split (never trained on) for honest recall/FP
    def split(X):
        idx = rng.permutation(len(X)); k = int(len(X) * 0.2)
        return X[idx[k:]], X[idx[:k]]

    Xp_tr, Xp_va = split(Xp)
    Xn_tr, Xn_va = split(Xn)
    bundle = out_dir / "bundle.npz"
    np.savez_compressed(bundle, Xp_tr=Xp_tr, Xp_va=Xp_va, Xn_tr=Xn_tr, Xn_va=Xn_va, Xpool=Xpool)
    print(f"[features] wrote {bundle}  pos_tr={len(Xp_tr)} pos_va={len(Xp_va)} "
          f"neg_tr={len(Xn_tr)} neg_va={len(Xn_va)} pool={len(Xpool)}", flush=True)


# ============================ STAGE 2: train =================================

def _build_net(layer_dim, n_blocks):
    import torch.nn as nn

    class FCNBlock(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.fcn_layer, self.relu, self.layer_norm = nn.Linear(d, d), nn.ReLU(), nn.LayerNorm(d)

        def forward(self, x):
            return self.relu(self.layer_norm(self.fcn_layer(x)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.layer1 = nn.Linear(N_FRAMES * 96, layer_dim)
            self.relu1, self.layernorm1 = nn.ReLU(), nn.LayerNorm(layer_dim)
            self.blocks = nn.ModuleList([FCNBlock(layer_dim) for _ in range(n_blocks)])
            self.last_layer = nn.Linear(layer_dim, 1)
            self.last_act = nn.Sigmoid()

        def forward(self, x):
            x = self.relu1(self.layernorm1(self.layer1(self.flatten(x))))
            for b in self.blocks:
                x = b(x)
            return self.last_act(self.last_layer(x))

    return Net()


def _train_head(Xp, Xn, cfg, device, rng, steps):
    import torch
    net = _build_net(cfg["dim"], cfg["blocks"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    Xall = torch.tensor(np.concatenate([Xp, Xn]), dtype=torch.float32, device=device)
    yall = torch.tensor(np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))]),
                        dtype=torch.float32, device=device)
    pos_w, neg_w = 1.0, cfg["neg_w"]
    bs = 256
    net.train()
    for _step in range(steps):
        idx = rng.integers(0, len(Xall), bs).tolist()
        xb, yb = Xall[idx], yall[idx]
        w = torch.where(yb > 0.5, torch.tensor(pos_w, device=device),
                        torch.tensor(neg_w, device=device))
        opt.zero_grad()
        p = net(xb).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy(p, yb, weight=w)
        loss.backward()
        opt.step()
    return net


def _scores(net, X, device):
    import torch
    if len(X) == 0:
        return np.zeros(0, np.float32)
    net.eval()
    with torch.no_grad():
        out = []
        for i in range(0, len(X), 4096):
            xb = torch.tensor(X[i:i + 4096], dtype=torch.float32, device=device)
            out.append(net(xb).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def _evaluate(net, Xp_va, Xn_va, device, thr=0.5):
    pp = _scores(net, Xp_va, device)
    pn = _scores(net, Xn_va, device)
    recall = float((pp > thr).mean()) if len(pp) else 0.0
    fp = float((pn > thr).mean()) if len(pn) else 0.0
    # separation: how far the median positive sits above the 95th-pct negative
    sep = float(np.median(pp) - np.percentile(pn, 95)) if len(pp) and len(pn) else 0.0
    return recall, fp, sep


def stage_train(args) -> None:
    import torch
    bundle = np.load(Path(args.workdir) / "bundle.npz")
    Xp_tr, Xp_va = bundle["Xp_tr"], bundle["Xp_va"]
    Xn_tr, Xn_va = bundle["Xn_tr"], bundle["Xn_va"]
    Xpool = bundle["Xpool"]
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    rng = np.random.default_rng(args.seed)
    steps = 300 if args.smoke else args.steps
    print(f"[train] device={device} pos_tr={len(Xp_tr)} neg_tr={len(Xn_tr)} "
          f"pool={len(Xpool)} steps={steps}", flush=True)

    # sweep grid (small in smoke)
    grid = ([{"dim": 32, "blocks": 1, "lr": 1e-3, "neg_w": 2.0}] if args.smoke else
            [{"dim": d, "blocks": b, "lr": 1e-3, "neg_w": w}
             for d in (48, 64) for b in (1, 2) for w in (1.5, 2.0, 3.0)])

    Xn_cur = Xn_tr
    best = None
    mine_rounds = 1 if args.smoke else args.mine_rounds
    for rnd in range(mine_rounds):
        print(f"[train] === round {rnd + 1}/{mine_rounds}  neg_tr={len(Xn_cur)} ===", flush=True)
        round_best = None
        for cfg in grid:
            net = _train_head(Xp_tr, Xn_cur, cfg, device, rng, steps)
            recall, fp, sep = _evaluate(net, Xp_va, Xn_va, device)
            # objective: maximise recall, then separation, with a hard FP ceiling
            ok = fp <= args.fp_ceiling
            score = (1 if ok else 0, round(recall, 3), round(sep, 3))
            print(f"    cfg dim={cfg['dim']} blk={cfg['blocks']} negw={cfg['neg_w']} "
                  f"-> recall={recall:.3f} fp={fp:.3f} sep={sep:.3f} {'OK' if ok else 'FP!'}",
                  flush=True)
            if round_best is None or score > round_best[0]:
                round_best = (score, cfg, net, recall, fp, sep)
        # hard-negative mining: add the pool clips this model fires hardest on
        _, cfg, net, recall, fp, sep = round_best
        if best is None or round_best[0] > best[0]:
            best = round_best
        if rnd < mine_rounds - 1 and len(Xpool):
            ps = _scores(net, Xpool, device)
            hard = Xpool[np.argsort(-ps)[:max(1, len(Xpool) // 4)]]
            Xn_cur = np.concatenate([Xn_cur, hard])
            print(f"    mined {len(hard)} hard negatives (max pool score {ps.max():.3f})", flush=True)

    score, cfg, net, recall, fp, sep = best
    print(f"[train] BEST cfg={cfg} recall={recall:.3f} fp={fp:.3f} sep={sep:.3f}", flush=True)

    # retrain the winning config on ALL data (train+val) for the shipped model
    net = _train_head(np.concatenate([Xp_tr, Xp_va]),
                      np.concatenate([Xn_cur, Xn_va]), cfg, device, rng, steps)
    out = Path(args.workdir) / "hey_medo.onnx"
    net.eval()
    net_cpu = net.to("cpu")
    torch.onnx.export(
        net_cpu, torch.rand(1, N_FRAMES, 96), str(out),
        input_names=["x"], output_names=["hey_medo"],
        dynamic_axes={"x": {0: "batch"}, "hey_medo": {0: "batch"}},
        opset_version=13, dynamo=False,
    )
    metrics = {"recall": recall, "fp": fp, "separation": sep, "config": cfg,
               "pos_val": int(len(Xp_va)), "neg_val": int(len(Xn_va)),
               "neg_train_final": int(len(Xn_cur))}
    (Path(args.workdir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[train] exported -> {out}", flush=True)
    print(f"[train] metrics -> {json.dumps(metrics)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep two-stage 'medo' wake-word trainer")
    ap.add_argument("--stage", required=True, choices=["features", "train"])
    ap.add_argument("--workdir", default="D:/wakeword_train/features")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--smoke", action="store_true", help="tiny fast end-to-end check")
    ap.add_argument("--offline", action="store_true", help="Piper only, skip edge-tts")
    # features
    ap.add_argument("--aug", type=int, default=8, help="augmentations per base clip")
    ap.add_argument("--pool", type=int, default=400, help="noise-bank / mining pool size")
    ap.add_argument("--real-dir", default=None, dest="real_dir",
                    help="dir with pos/ and neg/ WAVs captured from the real mic")
    # train
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--mine-rounds", type=int, default=3, dest="mine_rounds")
    ap.add_argument("--fp-ceiling", type=float, default=0.02, dest="fp_ceiling")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.stage == "features":
        stage_features(args)
    else:
        stage_train(args)


if __name__ == "__main__":
    main()
