# Roadmap

Ordered by value-for-effort.

1. ~~**Sentence-streaming TTS.**~~ *Shipped 2026-07-04:* `llm/client.py`
   streams both providers (`chat(..., on_delta=...)`), the voice loop speaks
   each completed sentence while the rest generates — first audio ~0.8 s on
   Groq instead of after the full reply. Barge-in (wake word / loud voice /
   HUD button) drops the remaining queued sentences.
2. ~~**Custom "hey MEDO" wake word.**~~ *Shipped 2026-07-24:* `models/wakeword/hey_medo.onnx`, trained on the owner's real captured
   voice folded into a synthetic set. Fires on both "medo" and "hey
   medo" (~95% on genuine tries); `stt_confirm` re-transcribes every
   trigger so the low 0.20 threshold costs an STT pass, not a false wake.
3. ~~**Macedonian voice.**~~ *Shipped 2026-07-04:* Cyrillic replies are spoken
   with `mk-MK-MarijaNeural` via edge-tts (free, online), Piper stays the
   offline fallback. `tts.multilingual` / `tts.mk_voice` in config.
4. **HUD niceties.** Latency sparkline in ROUTING, gesture-name toast over the
   optical feed. *(Provider row with base URL + key, mic picker, and accent
   themes shipped 2026-07-04.)*
5. **Watch app pairing polish.** Setting the phone/PC IP is manual today.
6. **Facts management UI.** List/delete remembered facts from CONFIG.

## Done (2026-07-04, `feature/pointer-themes-dirdots`)

- ~~Wake/interrupt triggers~~ — `POST /wake` breaks STANDING BY without the
  wake phrase; wake-word barge-in stops TTS mid-sentence.
- ~~OS-level scroll gesture~~ — two fingers = real mouse-wheel scroll; also
  zoom (Ctrl+wheel), pinch-drag, volume poses, held-fist exit.
- ~~Provider row in CONFIG~~ — Local/Online toggle, base URL + key (persisted
  to the git-ignored `secrets.local.yaml`).
- Microphone priority list with ~2 s hot-swap + HUD mic picker.
- Orb folder dots (~300 real directories) + file/web search overlay.
