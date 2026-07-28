#!/usr/bin/env python3
"""Best-effort LOCAL 'hey MEDO' training (no GPU, no big datasets).

Honest scope: this trains openWakeWord's small DNN head on top of its shared
embedding model, using the 52 synthetic positives + locally-generated negatives
(near-miss phrases + noise). With no large background corpus it will be weaker
and needs a higher threshold than a Colab-trained model — but it produces a
real, loadable hey_medo.onnx. Tune the threshold afterward with
`python -m voice.wakeword`.
"""
from __future__ import annotations

import glob
import os
import wave
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "models" / "wakeword" / "samples"
OUT = ROOT / "models" / "wakeword" / "hey_medo.onnx"
SR = 16000
N_FRAMES = 16          # model input is (16, 96)
CLIP_S = 2.5           # long enough to yield >=16 embedding frames


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        n, sr = w.getnframes(), w.getframerate()
        a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if sr != SR and a.size:
        m = int(len(a) * SR / sr)
        a = np.interp(np.linspace(0, 1, m, endpoint=False),
                      np.linspace(0, 1, len(a), endpoint=False), a).astype(np.float32)
    return a


def _fit(a: np.ndarray, place_random=True, rng=None) -> np.ndarray:
    """Place/pad audio into a fixed CLIP_S buffer (foreground at random spot)."""
    n = int(CLIP_S * SR)
    buf = np.zeros(n, dtype=np.float32)
    a = a[:n]
    start = 0
    if place_random and rng is not None and n > len(a):
        start = int(rng.integers(0, n - len(a)))
    buf[start:start + len(a)] = a
    return buf


def _augment(a: np.ndarray, rng) -> np.ndarray:
    a = a * float(rng.uniform(0.6, 1.0))                       # gain
    a = a + rng.normal(0, float(rng.uniform(0.0, 0.02)), a.shape).astype(np.float32)  # noise
    return np.clip(a, -1.0, 1.0)


def _windows(embeddings: np.ndarray) -> np.ndarray:
    """(N, frames, 96) -> (N, 16, 96): take/pad the first 16 frames."""
    out = []
    for e in embeddings:
        if e.shape[0] >= N_FRAMES:
            out.append(e[:N_FRAMES])
        else:
            pad = np.zeros((N_FRAMES - e.shape[0], e.shape[1]), np.float32)
            out.append(np.vstack([e, pad]))
    return np.asarray(out, dtype=np.float32)


def _gen_negatives(rng) -> list[np.ndarray]:
    """Local negatives: near-miss TTS phrases + noise + silence (no network reqd
    for the noise part; TTS is best-effort)."""
    clips: list[np.ndarray] = []
    # near-miss + common phrases via Piper (offline) — the strongest negatives.
    # NOTE: bare "medo" / "okay medo" are NOT here — listing them taught the old
    # model to REJECT a spoken "medo". "medo" is a wake word now, not a negative.
    # (For a stronger both-forms model see scripts/train_wakeword_deep.py.)
    phrases = ["hey medic", "hey model", "hey meadow so", "hey there",
               "hey computer", "play media", "the metro", "hey", "meadow",
               "hello", "what time is it", "hey mister", "hey buddy", "hey now"]
    try:
        from core.config import load_settings
        from voice.tts import TextToSpeech
        tts = TextToSpeech(load_settings().tts)
        for p in phrases:
            try:
                wav, sr = tts.synthesize(p)
                a = wav.astype(np.float32) / 32768.0 if wav.dtype.kind == "i" else wav.astype(np.float32)
                clips.append(a)
            except Exception:
                pass
    except Exception:
        pass
    # noise + silence negatives (cheap, many)
    for _ in range(120):
        clips.append(rng.normal(0, rng.uniform(0.005, 0.05), int(CLIP_S * SR)).astype(np.float32))
    for _ in range(20):
        clips.append(np.zeros(int(CLIP_S * SR), np.float32))
    return clips


def _build_net(layer_dim=32, n_blocks=1):
    """openWakeWord's DNN head, reimplemented so we avoid its heavy training
    deps (pronouncing/speechbrain/…). Same architecture => the exported ONNX
    loads in openWakeWord's runtime unchanged."""
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


def main() -> None:
    import openwakeword
    from openwakeword.utils import AudioFeatures

    rng = np.random.default_rng(1234)
    res = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
    F = AudioFeatures(
        melspec_model_path=os.path.join(res, "melspectrogram.onnx"),
        embedding_model_path=os.path.join(res, "embedding_model.onnx"),
    )

    pos_files = sorted(glob.glob(str(SAMPLES / "*.wav")))
    print(f"[train] {len(pos_files)} positive clips; augmenting x8")
    pos_audio = []
    for f in pos_files:
        base = _read_wav(f)
        for _ in range(8):
            # place at START (not random): we train on the first 16 embedding
            # frames, so the phrase MUST sit at the front or a "positive" ends up
            # being silence labelled wake (that made it fire on silence).
            pos_audio.append(_fit(_augment(base, rng), place_random=False))
    # negatives: place TTS/near-miss phrases at the start too (same window),
    # plus the noise/silence clips already fill the buffer.
    neg_audio = [_fit(_augment(c, rng), place_random=False) for c in _gen_negatives(rng)]
    print(f"[train] positives={len(pos_audio)} negatives={len(neg_audio)}; embedding…")

    def _i16(clips):  # openWakeWord's melspec wants 16-bit PCM, not float
        return (np.clip(np.asarray(clips), -1.0, 1.0) * 32767).astype(np.int16)

    Xp = _windows(F.embed_clips(_i16(pos_audio), batch_size=32))
    Xn = _windows(F.embed_clips(_i16(neg_audio), batch_size=32))
    print(f"[train] feature shapes pos={Xp.shape} neg={Xn.shape}")

    import torch

    def _t(a):  # np -> torch WITHOUT the numpy bridge (torch/numpy ABI mismatch)
        a = np.ascontiguousarray(a, dtype=np.float32)
        return torch.frombuffer(bytearray(a.tobytes()), dtype=torch.float32).reshape(a.shape)

    Xall = _t(np.concatenate([Xp, Xn]))
    yall = _t(np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))]))
    # weight negatives heavily — tiny negative set => guard hard against false wakes
    pos_w, neg_w = 1.0, 6.0

    net = _build_net(layer_dim=32, n_blocks=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    steps, bs = 4000, 128
    net.train()
    for step in range(steps):
        idx = rng.integers(0, len(Xall), bs).tolist()  # list, not np array (ABI)
        xb, yb = Xall[idx], yall[idx]
        w = torch.where(yb > 0.5, pos_w, neg_w)
        opt.zero_grad()
        p = net(xb).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy(p, yb, weight=w)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                pp = net(_t(Xp)).squeeze(1)
                pn = net(_t(Xn)).squeeze(1)
            print(f"  step {step:4d} loss {loss.item():.3f} "
                  f"pos_mean {pp.mean():.2f} neg_mean {pn.mean():.2f} "
                  f"pos_recall@0.5 {(pp>0.5).float().mean():.2f} "
                  f"neg_fp@0.5 {(pn>0.5).float().mean():.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    net.eval()
    torch.onnx.export(
        net, torch.rand(1, N_FRAMES, 96), str(OUT),
        input_names=["x"], output_names=["hey_medo"],
        dynamic_axes={"x": {0: "batch"}, "hey_medo": {0: "batch"}},
        opset_version=13,
    )
    print(f"[train] exported -> {OUT}")


if __name__ == "__main__":
    main()
