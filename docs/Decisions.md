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
- **Design provenance.** The HUD implements `Medo.dc.html` from the user's
  claude.ai/design handoff zip (the earlier share link had expired; the zip
  in `design/` is the source of truth). `support.js` in the handoff is the
  design tool's runtime, not part of the design — re-implemented in vanilla JS.
- **Deletion protocol.** The two source projects are deleted only after the
  merged app passes tests + live smoke checks, jarvis-web first, the v2
  Downloads folder (the copy source) last.
- **Bearer token + localhost exemption on :8710, not TLS.** The companion API
  can type, screenshot, and power off the PC, so "no auth by design" had to
  die. A random token minted into git-ignored `secrets.local.yaml` on first
  serve stops any LAN device from driving the machine; requests from
  127.0.0.1 skip it so the HUD and vision sidecar keep zero-config startup.
  Full TLS was rejected: self-signed certs break the browser HUD and the
  watch's Dart client for no gain on a home LAN — the port still must never
  be forwarded. `?token=` is accepted alongside the header for clients that
  can't set one (EventSource/MJPEG-style embeds); auth fails CLOSED when a
  LAN request arrives before a token exists. `remote.auth_enabled: false`
  restores the old behavior, documented as unsafe.
