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
  `SetCursorPos`/`mouse_event` is 30 lines. The same philosophy carried to
  macOS: `vision/macmouse.py` is raw CoreGraphics via ctypes (CGEvents,
  CFRelease'd — 15 fps would leak otherwise), with osascript only for
  volume/media where the HID media keys would need AppKit. pyautogui exists
  solely as the last-resort `anymouse` backend for other platforms, with its
  pause and corner trap disabled.
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
- **Personality is a decorator layer, not baked into skill strings.** Skills
  return literal, testable replies; `core/persona.py` occasionally appends a
  curated quip (probability `wit_level`) at ONE choke point in the router,
  which by construction skips errors, safety confirmations, and destructive
  actions — the "never joke at a safety prompt" rule lives in one place
  instead of thirty skill files. The LLM path gets the same persona as a 2–3
  sentence manner fragment (num_ctx is 4096; no prompt bloat). Swapping
  `style: professional` changes the whole assistant without touching a skill.
- **MEDO Link: manifest-driven tools + per-capability confirmation, HTTP/WS
  over MQTT.** A device manifest turns into skills through the SAME registry
  path built-ins, plugins, and MCP tools use — zero per-device code in MEDO,
  and the capability description IS the LLM tool description (the device
  author writes for the model). The safety flag lives per capability, not
  per device, because the device author knows which commands move motors —
  flagged ones ride the existing bilingual yes/no gate untouched. HTTP
  polling + optional websocket instead of MQTT: the API already exists,
  polling doubles as the liveness heartbeat, and a broker is one more
  always-on dependency for zero added capability. Offline devices keep
  their tools registered and say so — a tool that vanishes confuses the
  model more than one that reports "offline".
- **Design provenance.** The HUD implements `Medo.dc.html` from the user's
  claude.ai/design handoff zip (the earlier share link had expired; the zip
  in `design/` is the source of truth). `support.js` in the handoff is the
  design tool's runtime, not part of the design — re-implemented in vanilla JS.
- **Deletion protocol.** The two source projects are deleted only after the
  merged app passes tests + live smoke checks, jarvis-web first, the v2
  Downloads folder (the copy source) last.
- **`run_voice` extracted to `voice/loop.py` as a class, not split into free
  functions.** The 335-line function shared seven pieces of state through
  closures (models, barge-in flag, wake event…); as free functions that state
  would have become parameter soup. `VoiceLoop` names each phase
  (`_wait_for_wake`, `_capture`, `_run_turn`, `_speak`,
  `_play_interruptible`) and the phases share state as attributes. Pure
  mechanical move — same log lines, timings, and error handling; heavy
  engines still load in `run()`, so constructing the class stays
  dependency-free for tests. `main.py` drops from ~36 KB to ~21 KB of wiring.
- **Watch pairing = UDP discovery + a 6-digit code on the PC screen, never a
  token handout.** Discovery answers "where is MEDO" (name + port — what a
  LAN scan sees anyway); it must not answer "may I have the token", or M1's
  auth would be theater. The code flashed on the PC console is the trust
  anchor: typing it on the watch proves the person can see this machine's
  screen (same model as Bluetooth/TV pairing). Codes are single-use, expire
  in 2 minutes, and 5 wrong guesses kill the session — and `/pair/start`
  deliberately returns no secret, so an unattended spam of it only flashes
  codes on the owner's screen.
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

## Click snapping: preview the target instead of snapping blind

Pointer mode drives the cursor from the index fingertip, and pinching pulls
that fingertip down as the thumb comes up to meet it. The click therefore lands
tens of pixels below the button being aimed at.

**Decision: snap to the nearest clickable element, but show which one first.**

The obvious implementation — silently press the nearest button — was rejected.
A cursor that sometimes clicks somewhere other than where it is drawn destroys
the one thing a pointer must have, which is that it points. You would stop
trusting it, and an assistant you have to second-guess is worse than one that
misses by 30 px. So the chosen target is outlined on screen and the click waits
`snap_confirm_ms` (default 250 ms) before firing. A wrong pick is visible and
abortable; set the delay to 0 once you trust it.

Snapping is also opt-out at three levels: disabled in config, nothing within
the radius, or no accessibility data available all fall back to clicking the
raw cursor position — the pre-existing behaviour. The feature can only ever
add precision, never take away a click that used to work.

**Decision: the accessibility tree, not pixel template matching.**

Template matching would have to be told what a button looks like, and "what a
button looks like" is different in every application, theme, and DPI setting on
the machine — it would work on the demo and fail on the user's actual tools. UI
Automation is the OS telling us what is actually clickable, including the
control's real bounding box and role, which is exactly the question being
asked. It also degrades honestly: an app that exposes no accessibility data
reports nothing, and we click raw, rather than a matcher confidently finding a
"button" in a texture.

The tree is *probed*, not walked: `ControlFromPoint` at the cursor plus two
rings of eight points. Walking a window's full element tree is far too slow to
sit inside a click, and the probe answers the only question that matters —
what is reachable at these pixels.
## Web fetch: only the user's own links are followed without asking

`web_fetch` reads a page out loud, which means MEDO now pulls text from an
address someone else controls and hands it to the model that decides what to do
next. That is the classic prompt-injection surface, so the rule is about
*provenance*, not content: **a fetched page and an imported document are
UNTRUSTED INPUT, and a URL found inside one is an instruction from a third
party, not a request from the user.** "Also see medo.example/next-steps" sitting
in a PDF is a stranger telling the assistant where to go next; obeying it
silently would make every document the user opens a potential remote control.

**Decision: the trust boundary is where the URL came from, not what it points
at.** `SkillRequest.context` carries `url_source` — `"user"`, `"document"`, or
`"tool"` — set by whatever *put* the link in front of MEDO.

* A **fast-path regex match is trusted by definition**: the pattern matched the
  user's own utterance, so they said the host out loud themselves. There is no
  intermediary to be manipulated.
* The **LLM tool path is not**. By then the URL has passed through a model that
  may have read it out of a document, a previous page, or a tool result, and
  the model cannot reliably tell us which. So a missing or unknown `url_source`
  is treated as untrusted there, and the skill returns `needs_confirmation`
  naming the host ("That link came from a document, not from you. Fetch
  example.com?") — the same spoken yes/no gate the destructive skills use.
* `url_source` is deliberately **not** a tool parameter. If the model could
  declare it, an injected page would simply instruct the model to claim the
  link came from the user, and the check would be theatre. Provenance is
  reported by the code that owns the source, never by the component the
  attacker is talking to.

The confirmation is worth the friction because the failure it prevents is
silent: the user hears a summary and never learns that MEDO also visited a URL
of the attacker's choosing, with the user's IP and network position.

**Decision: non-http(s) schemes are refused outright, and NOT offered as a
confirmation.** `file:///`, `javascript:` and `data:` are not web pages —
there is nothing to read, only a local file to exfiltrate or a payload to
smuggle past the checks above. A confirmation is for a request that is
legitimate but consequential; asking "shall I open file:///etc/passwd?" would
frame an always-wrong action as a user preference, and the only outcomes are a
mistaken yes or a question that should never have been asked. The skill says it
reads http and https only and stops there.

---

## Import: pictures become documents, not a second retrieval path

`import_file` takes one file the user names out loud and makes it answerable
later. Documents were easy — they go through the same
chunk → embed → store pipeline `reindex()` already uses, via a new
`DocumentIndex.index_file()`, and `search_documents` finds them with no new
code. Pictures were the real question: an image has no text to chunk, so RAG
has nothing to work with.

**Decision: the vision model's description IS the document.** On import, the
picture is described once, and that description — plus the original path — is
written beside the image as a Markdown sidecar and indexed like any other file.
"What was in that diagram I imported" is then an ordinary documents query.

The rejected alternative was a separate image store with its own lookup ("find
the picture that…"). It would have meant a second retrieval path, a second
ranking implementation, and a user-visible seam: asking about a diagram would
work differently from asking about a PDF, for no reason the user could see. The
cost of the chosen design is that the description is written once, at import,
so a question the description didn't anticipate can't be answered from the
index — which is why the describe prompt asks for part numbers, labels and
layout rather than a one-line caption.

**Decision: the whitelist is checked on the SOURCE, and imports are copied,
never executed.** Importing is the one operation that reads a file MEDO was not
previously pointed at, so it is exactly where `PathWhitelist` has to apply. The
copy keeps its original suffix and is only ever read, so nothing in this path
can turn a document into a program.

---

## Trigger phrases and forced language are config, not code

Two things a user should never need Python for: what words they say, and which
language they say them in.

**Decision: `skills.triggers` appends to the same `patterns` list the built-ins
use.** A configured phrase becomes one more compiled pattern on that skill
instance — there is no parallel matching path, and nothing downstream (router,
registry, tests) can tell a config phrase from a code one. Phrases are matched
**literally**: they are escaped before compiling, so a user typing `what's up?`
gets what they typed instead of a regex error, and a config file cannot inject
a catastrophically backtracking pattern into the hot path. Phrases under three
characters are rejected because `"go"` would shadow every skill registered
after its owner. A custom phrase can only *add wording* — it cannot create a
skill, change what one does, or lift a confirmation gate.

**Decision: `stt.language_mode` folds into `stt.language` rather than becoming
a second field.** `language` already meant "None detects, a code forces", but
that was invisible to anyone reading the file. `language_mode` is the spelling
that says "auto" out loud; a validator resolves it into `language`, which stays
the only field the transcriber reads. Forcing a language also fixes the reply
language and the voice for free, because `transcribe_with_language` returns the
forced code and that is what picks both. A code MEDO has no voice for falls
back to auto-detect with a warning — forcing a decode nobody can be answered in
is worse than guessing.

---

## Naming a specialist has to resolve before the fast path claims the turn

Two bugs with one shape. `ask_specialist`'s patterns must capture a *free-form*
name — you address an expert by title, not by a fixed keyword — so
`"what does this page say about batteries"` captured "this page" as the expert.
The fast path stopped there and answered "I don't have that specialist",
and `web_fetch`, registered later, never saw the request. Separately, the
Macedonian pattern (`прашај го …`) had always matched and always resolved to
nobody, because `find_specialist` only knew English keys and titles — so every
Macedonian specialist request dead-ended in the same reply.

**Decision: resolution happens at match time.** `AskSpecialistSkill.match()`
returns `None` when the captured name is not a council member, so the router
carries on to the skills after it. A greedy pattern that can capture anything
must prove the capture is real before it claims the turn — the same fix
`OpenDiscoveredAppSkill` needed for `"open <anything>"`.

**Decision: native titles are data (`Specialist.aliases`), and the definite
article is stripped per word.** Macedonian glues the article onto the noun and,
in a phrase, onto the *adjective* — "машински инженер" is spoken
"машинскиот инженер". Stripping `от/та/то/те/ов/ва` from each word of 5+
characters keeps the alias table to base forms instead of every inflection,
and leaves short function words ("за", "на") alone. Config-defined specialists
take `aliases` too.

---

## Lion mode is a defensive-security PROFILE, not a bypass

Lion mode began as a "stop asking me" switch: on the router it skipped the
confirmation gate and overrode the PC-control switch. That was replaced.

**Decision: Lion mode changes PRESENTATION and which skills are SURFACED, and
changes nothing about safety.** Turning it on reskins the HUD deep red, shows a
LION indicator, and surfaces a group of read-only, local-machine, advisory
security skills (a listening-port audit, a firewall audit, a process explainer,
a file-permission explainer, an update/hygiene check). The confirmation gate,
the path whitelist, and the PC-control switch behave **identically** whether it
is on or off — asserted directly in the router tests, which now run the same
destructive request with the profile on and off and require the same gated
outcome. The flag moved from `safety.lion_mode` to `mode.lion` precisely
because it is no longer a safety control.

**Why "surface more, restrict nothing less" is the right design for a portfolio
piece.** Defensive security — auditing your own machine, explaining what's
listening, advising on hardening — is a real, hireable specialty, and it
demonstrates well: it is concrete, useful, and safe to show. An "unrestricted
mode" demonstrates the opposite. It is a liability: a single mis-heard word
reaching a system with every guard down is exactly the failure a reviewer would
(rightly) flag, and it teaches the user to click through the prompts that exist
to protect them. So the profile adds capability (more skills, a distinct look)
without ever subtracting a safeguard.

**Decision: the defensive skills are read-only, local-only, and refuse offense
in character.** Every skill inspects and advises; none kills a process, opens a
port, or changes a permission — those remain normal actuation that goes through
the normal confirmation gate, so the profile adds no fast path around it. A
"kill this process" request is declined by the explainer (it's an action, not a
description). The hard limits — this machine only; no scanning/probing other
hosts; no malware, exploits, credential-cracking, or bypass techniques; no
disabling MEDO's own guards — live both in the skills' guard code
(`is_offensive`) and in the system prompt handed to the model on every
advice-generating call (`LION_SYSTEM`), so the framing survives even when an LLM
writes the prose. Asked for any of the above, MEDO refuses and says lion mode is
defensive-only.

---

## HUD theme + effects are cosmetic layers that never touch behaviour

The C2 theme pack (glass, minimal, maximal, deep-red skins; scanlines/glitch/
grid effects) is CSS-only. **Decision: an effect layer is always
`pointer-events:none` and always sits behind the transcript**, so no theme can
block a click or dim the chat text — the one hard constraint on decoration. The
deep-red skin recolours the whole scene with a single `mix-blend-mode:color`
layer, which rotates hue while preserving luminance, so text stays exactly as
legible as it was rather than being re-typeset in a new palette. Theme (base
skin), effects (overlays), and the existing accent-hue swatches are three
independent axes that compose freely. Deep red is also Lion mode's identity:
arming Lion applies it without overwriting the user's hand-picked theme, and
disarming restores it.

---

## Detection is constrained to a user-chosen pair of languages (max two active)

MEDO listens for at most two languages at a time (default English + Macedonian),
chosen in `languages.active`. **Decision: per-utterance language detection is
constrained to those two — never open-ended across the ~90 languages Whisper
knows.** This is an accuracy feature, not a preference: open-ended language ID
mis-detects constantly (spoken Macedonian heard as Bulgarian/Serbian, Spanish as
Portuguese), and each mis-detection decodes with the wrong tokenizer and comes
out garbled. Scoring only the two active languages (via faster-whisper's
`detect_language` probabilities) removes the third language from the running, so
the choice is strictly en-vs-mk by actual probability — with a snap fallback
that pins any out-of-pair auto-detect to the primary language.

**Decision: two is a hard cap, validated on load.** `languages.active` with
three or more entries fails fast; `primary` must be one of the active languages;
unknown codes are dropped. `detection: fixed` forces the primary language with no
detection at all (fastest, most accurate for a single speaker); `auto_pair`
detects between the two. The legacy `stt.allowed_languages` / `language_mode`
fields survive only as back-compat inputs when no `languages:` block is present.

**Decision: everything language-dependent reads the active set — nothing hardcodes
en/mk.** The yes/no confirmation gate loads word banks per language from
`lang/confirm_words/<code>.yaml` (adding a language is a word file, not code), for
the active languages only; persona quips and the TTS voice follow the utterance's
language and degrade to the primary when a bank or voice is missing. The wake word
is a separate trained model — changing the active languages does not change the
phrase MEDO wakes to.

**Adding a language:** it needs STT support (faster-whisper handles it), a neural
voice (an entry in `core/languages.py`), and a yes/no word file
(`lang/confirm_words/<code>.yaml`, copied from `_template.yaml`). Add the code to
`languages.available`, then to `languages.active` (replacing one of the two) to
use it. The HUD CONFIG screen has a picker that enforces the two-language cap and
persists the choice (a restart reloads the STT model).

**Intertec note:** MEDO constrains language detection to a user-chosen pair, which
raises STT reliability and keeps the bilingual UX coherent — detection never
guesses across all languages, and every language-dependent surface (STT, yes/no
words, persona, voice) is driven by that one two-item set rather than scattered
per-feature assumptions.

---

## MCP servers: curated, disabled-by-default, secrets out of git

MEDO speaks MCP (`core/mcp.py`), so any app that serves MCP becomes callable
tools. `config.yaml`'s `mcp.servers` ships a curated set — **GitHub**,
**Google Maps**, **Google Drive**, **sequential-thinking**, **filesystem**,
**memory**, **Brave Search**, and a **Fusion 360** placeholder.

**Decision: every server ships `enabled: false`.** Enabling one spawns an npx
process that DOWNLOADS on first run — with the disk near full and most servers
needing a secret, silently spawning eight on startup is wrong. The user flips on
what they want. Only Node/npx servers are listed: this machine has npx but not
uvx or Docker (so GitHub's newer Docker-only official server isn't used — the
reference npx server is).

**Decision: MCP secrets live in the git-ignored overrides, never in
`config.yaml`.** `apply_local_secrets` injects a `mcp_env:` block from
`secrets.local.yaml` into the environment before servers spawn, so a GitHub PAT
or Google/Brave key is inherited by the child process without ever touching a
tracked file.

**Fusion 360 — the honest bit:** there is no official (Autodesk or Anthropic)
Fusion 360 MCP server. Autodesk exposes no MCP; community bridges run an add-in
INSIDE Fusion that serves a local endpoint an MCP server proxies. It's not a
drop-in — the entry is a disabled `url:` placeholder pointing at where such a
bridge would listen, with the setup left to the user (add-in + bridge). Google
Drive similarly needs a one-time OAuth consent before it works.

Verified: the pipeline connects to a real server (`sequential-thinking` loaded
its tool) — the mechanism works; each server just needs enabling + its secret.
