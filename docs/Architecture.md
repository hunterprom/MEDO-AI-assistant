# Architecture

Merged system: **v2's Python engine is the base**; v1's unique features
(gesture mouse control, thinking-model handling, bilingual prompt, desktop
tool catalog, long-term facts, screen vision) were ported into it. The whole
Node stack from v1 is gone — the HUD is one static HTML file served by the
Python app. See [[Decisions]] for the why.

## Pipeline

```
mic ──► openWakeWord ──► record until silence ──► faster-whisper (auto mk/en)
                                                        │
             HUD :8730 (SSE observer) ◄── EventBus ◄────┤
                                                        ▼
  watch app ──► companion API :8710 /ask ──────► Intent Router
  vision sidecar gestures ──► POST /ask ────────►   │
                                                    ├─ FAST: regex skill (<1 ms)
                                                    └─ LLM: Ollama tool loop
                                                        (strip_think, ≤4 rounds,
                                                         confirmation gate)
                                                        │
                                              Piper TTS ──► speakers
```

## Processes & venvs

| Process | Venv | Why separate |
|---|---|---|
| `main.py --voice --hud --serve` | `.venv` | voice stack, aiohttp servers |
| `vision/run.py` sidecar | `.venv-vision` | MediaPipe pins numpy<2 |
| `ollama serve` | — | needs `OLLAMA_MODELS=D:\OllamaModels`, `GGML_CUDA_NO_PINNED=1` |

The sidecar is dependency-light (stdlib + cv2 + mediapipe + yaml) and never
imports the main app. It talks to MEDO only through `POST :8710/ask`, and the
main app talks back only through `:8731/pointer` + `/frame.jpg`.

## Skills (28 registered, 24 exposed as LLM tools)

datetime · timers · notes · **recall/forget/remember facts** · **window
actions** · apps · **see_camera** · **see_screen** · files · volume (pycaw
Core Audio) · media · **type_text** · **press_keys** · **clipboard** ·
**brightness** · system_info · screenshot · **pointer_control** · power ·
weather · news · web search — bold = new in the merge. One implementation
serves both routing paths (regex `patterns` + `tool_schema`).

## Pointer mode (v1's crown jewel, rebuilt)

Camera thread in the sidecar: index-tip landmark → sensitivity remap
(recentered ×2.5) → EMA smoothing → `SetCursorPos` via ctypes (no pyautogui in
the sidecar — no 0.1 s pause, no FAILSAFE trap). Pinch/three-finger click with
600 ms debounce + 3-frame pose hold. Fist (stabilized) exits. The discrete
gesture→utterance map is suspended while pointer mode is on. Toggles: voice
("pointer on/off"), HUD switch, `POST :8731/pointer {"on": bool}` — always
boots OFF.

## HUD contract (frozen)

`ui/hud.py` injects a JSON config token into `ui/web/index.html` and streams
`{hello|state|transcript|routed}` over SSE `/events`. The page POSTs commands
to `:8710/ask` and polls `:8710/sys` (2 s cached CPU/RAM/GPU collector).
Chat bubbles render **only** from SSE so replies never appear twice.
Design source: `design/Medo.dc.html` (claude.ai/design handoff).
