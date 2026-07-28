# Wake Word Training — "hey MEDO"

How to replace the pretrained **"hey jarvis"** with a custom **"hey MEDO"**
model. The code side is already done: `wakeword.phrase` accepts a model path,
and the framework (onnx/tflite) is picked by file extension.

## Shortcut: the sample generator

MEDO ships a helper that does the tedious part — synthesizing a **diverse**
positive sample set (many edge-tts English voices + Piper) so you don't hand
it to the notebook empty:

```
.venv\Scripts\python -m voice.train_wakeword --generate
# -> models/wakeword/samples/*.wav  (16 kHz mono)
.venv\Scripts\python -m voice.train_wakeword --train    # prints the next steps
```

Upload those samples to the Colab notebook below (or run its generator with
`"hey medo"` as the phrase). Then follow steps 4–7. Diversity matters — a
model trained on one voice overfits, which is why the helper spreads across
a dozen accents/genders.

## How openWakeWord training works (the short version)

You never record thousands of samples yourself. openWakeWord trains on
**synthetic speech**: piper-sample-generator produces a few thousand TTS
clips of the phrase in varied voices/speeds/noise mixes, the trainer adds
negative data (speech that is *not* the phrase), and a small classifier
learns on top of openWakeWord's shared audio embedding. Cost: ~1 hour on a
free Colab T4.

## Steps

1. **Open the official training notebook** (runs end-to-end on free Colab):
   <https://github.com/dscripka/openWakeWord> → `notebooks/automatic_model_training.ipynb`
   (the "automatic model training" notebook — it wraps sample generation,
   augmentation, and training).
2. **Set the target phrase.** Use `"hey medo"` — and because "MEDO" is not an
   English word, also add pronunciation spellings as extra target phrases so
   the TTS generator covers how it's actually said: `"hey meh doh"`,
   `"hey may doh"`. Generate ≥ 2,000 positive samples (default is fine).
3. **Train with the notebook defaults** first (steps, negative weight). Only
   tune if the A/B below shows problems — defaults are good for a first model.
4. **Export/download the trained model** — you get `hey_medo.onnx` (and
   optionally `.tflite`).
5. **Place it in the repo** (git-ignored models dir, next to the Piper voice):

   ```
   models/wakeword/hey_medo.onnx
   ```

6. **Point the config at it** — `config.yaml`:

   ```yaml
   wakeword:
     phrase: "models/wakeword/hey_medo.onnx"   # was: "hey_jarvis"
     threshold: 0.5                            # reset to default for a new model
   ```

7. **A/B the threshold with the live monitor** (the same tool used to debug
   "stuck on STANDING BY"):

   ```
   .venv\Scripts\python -m voice.wakeword
   ```

   - Say "hey MEDO" 10× at normal distance/volume — note the **peak scores**
     (printed continuously, and summarized on Ctrl-C). If peaks sit at
     0.6–0.9, set `threshold` ≈ peak_min − 0.1.
   - **False-positive soak**: leave the monitor running through a podcast /
     normal room noise for 30+ minutes. Every `TRIGGERED` line is a false
     wake — if you get more than ~1/hour, raise the threshold and re-test;
     if raising it starts missing real wakes, train again with more samples
     instead of fighting the threshold.
   - Sanity-check both mics (headset + webcam fallback) — cheap mics score
     lower; the threshold must work on the *worst* mic you actually use.

8. **Update the docs** when it ships: README "Known limitations" (the
   wake-word line dies), [[Roadmap]], and a [[Bug Log]] row if anything
   surprising came up.

## Notes / gotchas

- The main venv runs the **onnx** build (onnxruntime is already a dependency
  of the voice stack). `.tflite` needs `tflite-runtime`, which has no Windows
  wheels — prefer the `.onnx` export on this machine.
- Keep `"hey_jarvis"` working as the committed default until the custom model
  is verified — the config change is one line either way, and anyone cloning
  the repo has no `models/wakeword/` (it's git-ignored, like the Piper voice).
- The bundled-name lookup and the path lookup coexist in
  `voice/wakeword.py::_resolve_model_path` — a bad path logs a warning and
  falls back to the bundled models rather than crashing voice mode.
