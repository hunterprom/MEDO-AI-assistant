# Security — the safe and unsafe sides of MEDO on the desktop

An honest account of what protects you when MEDO runs on your machine, and
what it can do *to* your machine (or expose *about* it) if misused or
misconfigured. MEDO is a desktop agent with real power — it types, clicks,
reads the screen, watches the camera, and can shut the computer down — so it
is worth understanding exactly where the guardrails are and where they aren't.
See also [[Decisions]] (why each choice was made) and [[Bug Log]].

**One-line summary:** MEDO is **local-first and safe by default on a trusted
home LAN**, but it is **not hardened against a hostile local network**, and it
runs **any plugin, MCP server, or cloud provider you point it at** with your
full user privileges. Treat the machine running MEDO as you'd treat one with
remote-desktop enabled: fine behind your own router, never on public Wi-Fi or
a forwarded port.

---

## 1. Threat model — who this is (and isn't) built to resist

**In scope (defended):**
- A curious device on your home LAN poking at the ports.
- An accidental destructive command ("shut down", "delete…") — must be
  confirmed out loud.
- Secrets leaking into git or logs.
- A malformed or oversized request crashing the service.

**Out of scope (NOT defended):**
- A hostile actor already on your LAN who can sniff plaintext traffic.
- Anyone with a login or file access to the machine itself.
- A malicious plugin, MCP server, or model you deliberately install.
- Traffic once you forward a port to the public internet (**don't**).

---

## 2. The safe side — what protects you

### Local-first by default
STT (faster-whisper), the wake word (openWakeWord), the voice (Piper), the
vision model (moondream), gesture recognition, and long-term memory all run
**on your machine**. With the default `provider: ollama`, no conversation, no
audio, and no screen content leaves the computer. This was audited and
corrected once already — v1's "100 % local" claim was false because browser
Web-Speech STT sent audio to Google ([[Bug Log]] #14); v2's is genuine.

### Confirmation gate for destructive actions
Shutdown, restart, and sleep set `needs_confirmation` and route through
`core/safety.py` — MEDO asks "Are you sure?" and only proceeds on a spoken
**yes**. The gate is **bilingual** (English + Macedonian, Cyrillic and
transliterated) and matches the *whole* reply, so an ambiguous answer
("не знам" / "I don't know") cancels rather than guesses ([[Bug Log]] #22).
`safety.confirm_destructive: true` by default. Destructive tools called via
the LLM path hit the same gate.

### Filesystem sandbox
Every file the file-skill touches is checked against
`core.safety.PathWhitelist` — resolved through symlinks, so
`~/Documents/../.ssh` cannot escape the allowed trees
(`~/Documents`, `~/Downloads`, `~/Desktop` by default). The companion API's
`/open` endpoint only opens whitelisted folder shortcuts, never arbitrary
paths.

### No shell injection in app launching
Apps open through a **config-mapped table** (`skills.apps`), not a shell
string built from what you said — so "open chrome; rm -rf …" can't smuggle a
command ([[Bug Log]] #15).

### Pointer mode is opt-in and refusable
Gesture mouse control **always boots OFF**, can be disabled entirely in config
(`vision.pointer.enabled`), and exiting takes a deliberate **held** fist so a
brief misread can't trap you. A stray toggle can never silently grab the
mouse unasked.

### Secrets stay out of git
API keys, the companion-API token, and the picked microphone live in
`secrets.local.yaml`, which is **git-ignored** and never committed, logged, or
echoed back by the API (`/status` reports only *whether* a key is stored).

### Companion API authentication (added in the hardening pass)
The companion API on **port 8710** requires `Authorization: Bearer <token>`
(or `?token=`) from any non-local client. The token is a 32-char random value
minted on first serve into `secrets.local.yaml`. Requests from `127.0.0.1`
(the HUD, the vision sidecar) are exempt so local use stays zero-config. Auth
**fails closed**, uses a constant-time compare, and oversized input is
rejected (`MAX_TEXT_CHARS = 2000`). See [[Decisions]] and [[Bug Log]] #21.

### Watch pairing without handing out the token
"Pair with MEDO" proves physical presence: MEDO flashes a **6-digit code on
the PC screen**, single-use, 2-minute expiry, 5-attempt lockout. UDP discovery
answers only *name + port* — never the token. So an unattended device can, at
worst, make codes flash on your own monitor.

---

## 3. The unsafe side — real risks, ranked

> These are honest limitations, not hidden flaws. Most are safe on a trusted
> LAN and become dangerous only off it, or when you relax a default.

### ⚠️ 3.1 The vision sidecar (port 8731) is UNAUTHENTICATED
This is the biggest gap. `vision/run.py` binds an HTTP server to
**`0.0.0.0:8731`** with `Access-Control-Allow-Origin: *` and **no token
check** on any route. On the LAN, anyone can:
- **watch your webcam** — `GET /video` (live MJPEG) and `/frame.jpg`;
- **toggle gesture mouse control** — `POST /pointer {"on": true}`.

The companion API's M1 auth does **not** cover this port — the sidecar only
*sends* the token to :8710, it doesn't *require* one. **Mitigation:** keep
:8731 behind your firewall, never forward it, and turn the camera/sidecar off
when unused. (A proper fix — bind to `127.0.0.1` or add the same bearer check
— is a good next hardening step; noted for [[Roadmap]].)

### ⚠️ 3.2 No TLS anywhere — LAN traffic is plaintext
Neither :8710 nor :8730 nor :8731 uses HTTPS. On a shared or hostile network,
someone sniffing can read the **bearer token** (on non-local requests), your
transcripts, and the **webcam stream**. This was a deliberate trade — self-
signed certs break the browser HUD and the watch client for no gain on a home
LAN ([[Decisions]]) — but it means the security boundary *is* your router.
**Never forward these ports.**

### ⚠️ 3.3 The companion API is, by design, near-total control
Anyone who is on localhost, or who has the token, can drive `/ask`, which
reaches the full skill set: **type text, take screenshots, read the clipboard,
read the screen (vision), manage windows, change volume, launch apps**, and —
with spoken confirmation — **power the machine off**. That is the product
working as intended, but it means the token is as sensitive as a login. If you
set `remote.auth_enabled: false`, the port becomes **wide open** to the LAN —
the config comments mark this UNSAFE for a reason.

### ⚠️ 3.4 Plugins run arbitrary code with your privileges
Any `plugins/*.py` file is imported at startup and its skills registered
([[Bug Log]] context: this is the extensibility model). A malicious plugin is
**full code execution** as you. Only drop in plugins you have read and trust.

### ⚠️ 3.5 MCP servers run arbitrary commands from config
`mcp.servers` entries spawn whatever `command`/`args` you configure (`npx`,
`uvx`, a binary) or connect to a URL you name. Their tools become callable by
the model. Same rule as plugins: **only connect servers you trust**, and
remember the LLM can invoke their tools.

### ⚠️ 3.6 Cloud egress the moment you leave the local defaults
- Choosing the **GPT / Claude API** provider sends your conversation to that
  vendor.
- The **Claude Code / Codex CLI** agents send prompts to Anthropic / OpenAI.
- **Macedonian replies** use edge-tts, which sends the reply *text* to
  Microsoft's servers (the toggle is `tts.multilingual`).
None of this is hidden, but "local-first" stops being "local-only" the moment
you flip these — decide per your privacy needs.

### ⚠️ 3.7 No general rate limiting
Beyond the input-size cap and the pairing attempt-limit, there's no throttle
on `/ask`, `/type`, etc. A client that holds the token (or any LAN client if
auth is off) can flood the machine with actions.

### ⚠️ 3.8 Always-listening / always-watching while running
In voice mode the microphone is **continuously** sampled for the wake word;
with the sidecar up, the **camera streams** continuously. Both are local, but
the sensors are live whenever MEDO runs — worth knowing before an assistant
sits open on a work machine.

### ⚠️ 3.9 Secrets file is plaintext on disk
`secrets.local.yaml` holds API keys and the token in **cleartext**. It's
git-ignored, but not encrypted — anyone with read access to your user account
(or an unlocked machine, or a backup) gets them. Protect it with normal file
permissions and full-disk encryption (FileVault).

---

## 4. Hardening checklist (what YOU can do)

- [ ] **Never forward ports 8710 / 8730 / 8731** past your router. LAN only.
- [ ] Keep `remote.auth_enabled: true` (the default). Only disable on a
      network you fully trust and physically control.
- [ ] Put the machine on a **trusted private network** — not café/airport
      Wi-Fi. If you must, use a firewall to block inbound 8710/8731.
- [ ] Turn the **camera/vision sidecar off when unused** (it's the one
      unauthenticated surface) — pointer mode also always starts OFF.
- [ ] Only install **plugins and MCP servers you have read and trust.**
- [ ] Decide consciously about **cloud brains** and Macedonian TTS — the
      local Ollama default keeps everything on-device.
- [ ] Enable **FileVault / full-disk encryption**; treat `secrets.local.yaml`
      as a password file. Rotate the token by deleting `remote.token` from it
      and re-pairing.
- [ ] Keep `safety.confirm_destructive: true` and the whitelist tight
      (`safety.whitelist_dirs`) — don't add `~` or `/`.
- [ ] Log out / lock the machine when away — a local user or an open session
      has full access regardless of the network controls above.

---

## 5. Bottom line

MEDO is built to be **safe for its intended use**: a local-first assistant on
your own trusted machine and home network, with confirmation before it does
anything irreversible and no data leaving the box unless you ask it to. Its
power is also its risk — it acts on your desktop with your privileges — so the
security model relies on two things staying true: **the network stays private
(don't forward the ports; mind the unauthenticated camera port)**, and **you
only extend it with code you trust (plugins, MCP servers, cloud providers)**.
Hold those, and MEDO is safe. Break either, and it's as exposed as any tool
that can see your screen and move your mouse.
