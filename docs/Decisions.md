# Decisions

Why the super project is shaped the way it is.

- **v2's Python engine is the base, v1's features ported in** — v2 has the
  better architecture (event bus, one-implementation-two-paths skill contract,
  safety confirmation gate, tests); v1 had the better features (pointer mode,
  think-stripping, facts, desktop catalog). Port features, keep architecture.
- **The Node stack is gone.** The HUD is one static HTML file served by the
  existing aiohttp server. No npm, no Vite, no node_modules — one less
  runtime, several hundred MB saved, zero build step.
- **ctypes instead of pyautogui in the sidecar.** The sidecar must stay
  dependency-light (numpy<2 pin) and move the cursor 15×/s; pyautogui adds a
  default 0.1 s pause per call and a FAILSAFE corner trap. Raw
  `SetCursorPos`/`mouse_event` is 30 lines.
- **`/sys` instead of extending `/status`.** `/status` does a live Ollama
  round-trip per hit (model listing) and its shape is pinned by the watch app
  and tests. Telemetry is a cached dict refreshed by a background task.
- **HUD chat renders only from SSE.** `/ask` responses are ignored on
  purpose — voice, watch, gestures, and typed commands all surface through
  the same event stream, so every client sees every conversation once.
- **`num_ctx: 4096`.** 8192 OOMs the 12 GB GPU with qwen3:30b partially
  offloaded. Learned the hard way in v1.
- **`keep_alive: 30m`.** The single biggest perceived-latency fix: stops
  Ollama unloading the 30B after 5 idle minutes.
- **Never auto-pull qwen3:30b.** 18 GB; D: has ~8 GB free. The launcher warns
  instead. moondream (~1.7 GB) is also only suggested, not pulled.
- **strip_think at the `chat()` choke point** — not in the router — so
  conversation memory, the HUD, TTS, and the confirmation gate all see clean
  text without four call-site fixes. `tool_calls` are never dropped, even
  when content strips to empty.
- **Facts in sqlite with `UNIQUE COLLATE NOCASE`** — dedupe at the schema
  level; short-lived connections per op (no cross-thread sharing).
- **Pointer mode always boots OFF** and turning it on can be refused by
  config — a stray API call must never grab the mouse unasked.
- **Bearer token + localhost exemption on :8710, not TLS.** The companion API
  can open files and (via skills) type/control power, so LAN clients now
  authenticate: a random token generated into git-ignored secrets.local.yaml
  (`remote.token`), checked constant-time on every request, `?token=` fallback
  for clients that can't set headers. Full TLS would mean self-signed-cert
  management on a watch for a LAN-only port — pain without covering any threat
  this doesn't. 127.0.0.1 is exempt so the HUD and vision sidecar keep
  zero-config startup; `remote.auth_enabled: false` restores the old open API
  (documented as unsafe). Access logging is off so `?token=` never lands in
  the console.
- **`run_voice` extracted to `voice/loop.py` as `VoiceLoop`.** A ~335-line
  function in main.py was the one place the modular story collapsed. Pure
  mechanical move: each phase is a named method (standby / listen /
  transcribe / route-streaming / speak / barge-in watcher), tuning constants
  became class attributes, behavior byte-for-byte identical. The audio stack
  imports stay lazy *inside* `run()` — constructing a VoiceLoop touches no
  audio deps, so the wiring is testable on machines without them.
- **Design provenance.** The HUD implements `Medo.dc.html` from the user's
  claude.ai/design handoff zip (the earlier share link had expired; the zip
  in `design/` is the source of truth). `support.js` in the handoff is the
  design tool's runtime, not part of the design — re-implemented in vanilla JS.
- **Deletion protocol.** The two source projects are deleted only after the
  merged app passes tests + live smoke checks, jarvis-web first, the v2
  Downloads folder (the copy source) last.
