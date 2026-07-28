# Custom wake-word models

Drop a trained openWakeWord model here to change MEDO's wake phrase from the
bundled **"hey jarvis"** to your own — e.g. **"hey MEDO"**:

```
models/wakeword/hey_medo.onnx
```

Then point `config.yaml` at it:

```yaml
wakeword:
  phrase: "models/wakeword/hey_medo.onnx"   # .onnx (or .tflite on Linux)
  threshold: 0.5                            # re-tune with: python -m voice.wakeword
```

## Making one

```
.venv\Scripts\python -m voice.train_wakeword --generate   # -> samples/*.wav
.venv\Scripts\python -m voice.train_wakeword --train       # prints training steps
```

The generator synthesizes a diverse "hey medo" positive set (many edge-tts
voices + Piper). The training run itself (~1 h on a free Colab GPU, or locally
if you have the openWakeWord background datasets) is the one manual step — full
runbook in [`docs/Wake Word Training.md`](../../docs/Wake%20Word%20Training.md).

Everything in this folder except this README is git-ignored (models are large
and personal).
