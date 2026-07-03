# Roadmap

Ordered by value-for-effort.

1. **Sentence-streaming TTS + barge-in.** Stream the LLM reply, synthesize and
   speak each completed sentence immediately (v1 did this in the browser), and
   let new speech / a fist gesture cancel playback. Needs a streaming path in
   `llm/client.py` (suppress-until-`</think>` like v1's `chatStream`) and a
   stoppable `Speaker`.
2. **Custom "hey MEDO" wake word.** Train an openWakeWord model (or switch
   engine) so the assistant answers to its actual name; today it's the
   pretrained "hey jarvis".
3. **Macedonian voice.** Optional `edge-tts` engine (`mk-MK` neural voices)
   selected automatically when the reply is Cyrillic; falls back to Piper
   offline. (Piper itself has no mk voice to date.)
4. **Wake/interrupt gestures.** v1's open-palm-while-idle = wake,
   fist-while-speaking = barge-in — needs the voice loop to accept external
   triggers (small event-bus extension).
5. **HUD niceties.** Latency sparkline in ROUTING, gesture-name toast over the
   optical feed, provider row (openai base URL + key) in CONFIG.
6. **OS-level scroll gesture.** Two fingers scrolled the chat in v1; map it to
   real mouse-wheel events in pointer mode instead.
7. **Watch app pairing polish.** Setting the phone/PC IP is manual today.
8. **Facts management UI.** List/delete remembered facts from CONFIG.
