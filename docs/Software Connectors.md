# Software Connectors — control local apps by command

MEDO can control other applications on the PC by command ("play/pause",
"minimize", "new tab", "close Notepad"). This is **direct local control — NOT
MCP.** No MCP servers, no protocol, no network daemon. It is "MEDO Link, but for
local software instead of hardware devices": a connector declares an app + its
actions, and each action becomes a routable skill through the *same* registration
path MEDO Link uses. Off by default (`software.enabled: false`) — flipping it on
is purely additive.

## The control ladder (robust → brittle, degrades gracefully)

Every action prefers mechanisms in this order; if none is available it says so —
it never fakes success:

1. **Native local API / CLI** — the app's own control surface (a media CLI, an
   OBS websocket, `code` for VS Code). Survives UI changes. The CLI adapter runs
   only a concrete argv list, never a raw shell, and **always** through the
   policy engine's `run_command` capability. The API adapter is localhost-only
   and gated by `network`.
2. **App-specific hotkeys** sent to the app's FOCUSED window — the
   *focus target → act → restore focus* flow (`Mechanisms.send_app_hotkey`).
3. **UI automation over the accessibility tree** (`software/accessibility.py`) —
   the brittle fallback, reusing the M7 pointer-snap approach (`vision/snap.py`).

## Security (non-negotiable)

Every action routes through the central **policy engine** (`security/policy.py`):

- Each action declares its **capabilities** (`control_input`, `run_command`,
  `network`); the router gates the skill on them exactly like any other.
- **Confirmation** for high-impact/irreversible actions (closing an app with
  unsaved work, running a CLI command).
- **Sending or posting anything ALWAYS confirms first** — MEDO never sends on
  your behalf without a yes. (See `TemplateConnector.send_message`.)
- **Trust boundary:** software control is honored ONLY from your own voice/text.
  An instruction that arrives from untrusted content — a document, a web page,
  another app's output (`context["untrusted"]` / `provenance == "untrusted"`) —
  is refused, never acted on.

## Adding an app = a new connector file (no core edit)

Copy `software/connectors/template.py`, fill in `app_id`, `display_name`,
`detect()`, and `actions()`, then add the class to
`software/connectors/__init__.py`'s `CONNECTOR_CLASSES`. Each `Action` carries:
its trigger `phrases` (EN + MK), `mechanisms` preference, declared `capabilities`,
`requires_confirmation`, and a `run(params)` that calls the shared mechanisms.

## Reference connectors shipped

- **media** — play/pause/next/previous/volume via global media keys (unverified
  by nature — reported honestly).
- **window** — focus / minimize / close a window; `close` confirms first.
- **browser** — new tab / next tab via app hotkeys (the focus→act→restore flow).

`OBS`-style local-API connectors are supported by the API adapter but shipped as
opt-in templates — verify the app's websocket/port from its live docs before
wiring (don't hardcode an invented interface).

---

*Intertec note:* MEDO controls local software through a connector abstraction with
a robustness ladder (native API/CLI → app hotkeys → accessibility-tree
automation), every action gated by the central policy engine and the
user-instruction trust boundary.
