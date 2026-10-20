# Troubleshooting

Everything on this page is user-fixable — no code changes needed. If your
problem isn't here, `docs/runbook.md` covers operational procedures and
`docs/bug-log.md` tracks known defects.

Split out of `README-detailed.md`, which had grown to the point where the
answer you needed was buried in the middle of a feature tour.

---

**Q: MEDO is stuck on STANDING BY / seems deaf.**
A: Run the live tester: `.venv\Scripts\python -m voice.wakeword`. It lists input
devices and prints mic level + wake score every frame. Level stuck near 0 →
wrong/muted microphone: fix `audio.input_device` or pick a mic in the HUD CONFIG
tab. Score peaking just under the threshold → lower `wakeword.threshold`. Or skip
the phrase entirely: the HUD **WAKE** button starts a turn without it.

**Q: Replies say "I can't reach my language model".**
A: The selected brain isn't reachable. Ollama: install/start it and pull a model
(`ollama pull llama3.2:3b`). GPT/Claude API: enter the key in the HUD CONFIG tab
and check the base URL. Claude Code / Codex: log in once on this machine. Fast-
path commands (time, volume, apps…) work regardless.

**Q: A council question / "second opinion" says the council can't meet.**
A: The council role-plays experts on the model that's loaded, so it needs a
reachable brain (or, for a CLI-agent brain, a local `tool_brain_model` on Ollama
so several specialists don't each spawn a slow process). Start Ollama or pick a
provider in the HUD.

**Q: "Type …" works in English but does nothing with Macedonian text.**
A: Cyrillic goes through a clipboard paste (Cmd+V on macOS). If it still fails,
the focused app may block programmatic paste; your clipboard is restored either
way.

**Q: The watch says "missing or invalid token".**
A: Re-pair from the watch (**Pair with MEDO** → type the 6-digit code MEDO shows
on the PC), or copy `remote.token` from the PC's `secrets.local.yaml`.

**Q: The camera feed is black / the sidecar exits immediately.**
A: On **macOS** this is almost always the **Camera permission** — grant it to the
app that launches the sidecar (Terminal / `run.command`) in System Settings →
Privacy & Security → **Camera**. Make sure no other app is holding the webcam.
The sidecar also needs its own venv (`run.command` builds `.venv-vision` on first
launch). Pointer mode additionally needs the **Accessibility** permission on
macOS.

**Q: A non-English reply isn't spoken (but shows in the HUD).**
A: The neural voices (edge-tts) need internet. Offline, English falls back to the
local Piper voice; a non-English reply is shown but not spoken rather than read
aloud with English phonemes. Text in the HUD stays correct either way.

**Q: A remote HUD (phone on the LAN) can't drive MEDO.**
A: By design — the API token is served to loopback only and cross-origin browser
requests are refused. Reach a remote HUD over an authenticated tunnel, or add its
origin to `remote.allowed_origins` on a trusted network.
