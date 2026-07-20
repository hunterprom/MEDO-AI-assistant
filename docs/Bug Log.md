# Bug Log

Bugs found and fixed while building the super project (2026-07-03), plus the
honest list of what still isn't perfect. See [[Roadmap]] for planned work.

## Fixed during the merge

| # | Bug | Fix |
|---|-----|-----|
| 1 | qwen3:30b's `<think>` reasoning leaked into TTS, conversation memory, and the yes/no confirmation gate (v2 had no stripping at all) | `strip_think()` at the single `chat()` choke point in `llm/client.py`; unterminated-`<think>` truncation handled; tool_calls preserved |
| 2 | 30B unloaded after Ollama's 5-minute default → 15–30 s reload before every delayed reply | `keep_alive: 30m` in every chat payload |
| 3 | `default_model: null` triggered an interactive startup picker — broke one-click launch and headless serving | ships `qwen3:30b`; graceful fallback to first installed model |
| 4 | STT hardcoded `language: "en"` — Macedonian input impossible (v1 regression) | `stt.language: null` = per-utterance auto-detect; nullable in config schema |
| 5 | English `initial_prompt` biased Macedonian decodes once auto-detect was on | prompt nulled by default, documented |
| 6 | LLM returning empty content (truncated mid-think) was answered with the misleading "can't reach my model" | honest `EMPTY_REPLY` fallback |
| 7 | v1's cursor mapping mirrored X unconditionally — double-mirror when the camera already flips (`flip: true`) | mirror is flip-conditional in `vision/pointer.to_screen` |
| 8 | Discrete gestures (point_up→volume up, pinch→lock!) would fire while steering the cursor | discrete map suspended in pointer mode |
| 9 | `pyautogui` in the sidecar would add a 0.1 s pause per cursor move (~10× too slow for 15 fps) and a FAILSAFE corner trap | ctypes `SetCursorPos`/`mouse_event` instead; zero new sidecar deps |
| 10 | `pyautogui.write()` silently drops Cyrillic — "type" was ASCII-only | non-ASCII goes through clipboard paste (pyperclip), clipboard restored after |
| 11 | Polling `/status` for HUD stats would hammer Ollama every 2 s (it lists models per hit) | new cached `GET /sys` with a background psutil/nvidia-smi collector |
| 12 | HUD replies rendered twice when typed (fetch response + SSE `routed`) | chat renders only from SSE |
| 13 | v1 `.env` was dead config — the launcher ran `npm run server`, which never loaded it (num_ctx 8192 vs effective 4096) | one config source (`config.yaml`); num_ctx pinned 4096 (8192 OOMs this GPU) |
| 14 | v1 "100% local" claim was false: browser Web-Speech STT sends audio to Google | whisper is genuinely local |
| 15 | v1 `open_app` built a shell string from user text (injection-shaped) | config-mapped app launch table (v2 design) kept |
| 16 | "forget it" would have deleted every fact containing "it" | vague-pronoun guard in ForgetFactSkill |
| 17 | "what do you remember" could be *stored as a fact* | recall/forget register before remember; remember is start-anchored |
| 18 | Windows volume broken with newer pycaw (`AudioDevice` has no `.Activate`) | 3-way endpoint fallback (session fix, carried over) |
| 19 | "close browser" killed a nonexistent `browser.exe` | image name derived from the launch command (session fix, carried over) |
| 20 | Missing Piper voice crashed `--voice` and took the HUD+API down with it | launcher downloads the voice; TTS optional; servers survive voice-stack failure (session fixes, carried over) |
| 21 | Confirmation gate was English-only — the bilingual assistant could not confirm/cancel destructive actions in Macedonian ("да"/"не" fell through as unknown) | yes/no sets extended with Cyrillic + Whisper's Latin transliterations; replies normalized (lowercase, punctuation stripped, whitespace collapsed) but still matched whole — "не знам" stays ambiguous, never a cancel |
| 22 | Macedonian STT only understood trivial phrases ("Како си?" worked, complex sentences decoded as garbage): Whisper `small` is too weak for mk, auto-detect misheard mk as Bulgarian/Serbian (wrong tokenizer → garbled decode), CUDA math DLLs were missing (CPU int8), and the 1.0 s silence gate cut sentences at mid-thought pauses | `large-v3-turbo` on GPU — pip `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` wheels + DLL-dir registration in stt.py; en/mk detection clamp (one forced re-decode on misdetection); silence gate 1.4 s. A/B on a synthesized complex mk sentence: small → "� micronquax 석고", turbo → the full sentence (decode 1.5 s, warm load 3.1 s) |

## Known limitations (open, by design or deferred)

- Macedonian replies are **spoken with an English Piper voice** (no mk voice
  exists for Piper). Text in the HUD is correct. → [[Roadmap]]
- Wake word is pretrained **"hey jarvis"**, not "MEDO". → [[Roadmap]]
- Brightness: external monitors don't expose WMI brightness — graceful error.
- moondream↔qwen3 VRAM contention on 12 GB: one reload after vision calls.
- Companion API is deliberately unauthenticated — **LAN only, never forward
  port 8710.**
- v1's two-finger chat scroll and open-palm-wake/fist-barge-in gestures are
  not carried over yet (need voice-loop integration). → [[Roadmap]]
