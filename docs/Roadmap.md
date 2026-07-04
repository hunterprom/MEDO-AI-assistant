# Roadmap

Ordered by value-for-effort.

1. **Sentence-streaming TTS.** Stream the LLM reply, synthesize and speak each
   completed sentence immediately (v1 did this in the browser). Needs a
   streaming path in `llm/client.py` (suppress-until-`</think>` like v1's
   `chatStream`). *Barge-in itself shipped 2026-07-04:* say the wake word over
   MEDO's reply (or hit the HUD mic button / `POST /interrupt`) to cut it off
   and be heard.
2. **Custom "hey MEDO" wake word.** Train an openWakeWord model (or switch
   engine) so the assistant answers to its actual name; today it's the
   pretrained "hey jarvis".
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
