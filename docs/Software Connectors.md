# Software Connectors — control local apps by command

MEDO can control other applications on the PC by command ("play/pause",
"minimize", "new tab", "close Notepad"). This is **direct local control — NOT
MCP.** No MCP servers, no protocol, no network daemon. It is "MEDO Link, but for
local software instead of hardware devices": a connector declares an app + its
actions, and each action becomes a routable skill through the *same* registration
path MEDO Link uses. Off by default (`software.enabled: false`) — flipping it on
is purely additive.

## Turn it on — the HUD panel

You don't have to edit `config.yaml`. Open the HUD (`localhost:8730`) → **CONFIG**:

- **App control** — a top-level toggle, next to *PC control* and *Developer mode*.
  It's the master switch: on = MEDO may drive your local apps by voice/text.
- **Software control · local apps** — a panel (under the CONNECT/connectors
  section) with, per connector (**media**, **window**, **browser**):
  - a **live availability** dot — ● installed · ✕ not installed · ○ shown once the
    feature is on;
  - the **kinds of commands** it understands (e.g. media → `play pause · next
    track · volume up …`), so you can see what to say;
  - a **per-connector toggle** to enable just that one.

Changes **apply on the next restart** (connectors register at startup, like MCP
servers), and are **remembered per machine** — persisted to the git-ignored
overrides (`secrets.local.yaml`), never `config.yaml`.

Under the hood the panel is two companion-API endpoints: `GET /software`
(master + per-connector state, live detect, and each connector's actions) and
`POST /control/software` (`{"on": bool}` for the master switch and/or
`{"app_id": ..., "app_on": bool}` per connector). The trust boundary and policy
gating below are unchanged whether you toggle from the HUD or `config.yaml`.

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

## Learning an app's UI (know where to go)

Beyond the built-in connectors, MEDO can **learn an app it doesn't have a
connector for** — "learn how to use CapCut" (or the HUD's *Teach MEDO an app*
box). It scans the app and remembers where its controls are, so later you can
ask "where is the effects search in CapCut" (locate) or "open the effects panel
in CapCut" (navigate). Three skills: `learn_app`, `locate_in_app` (read-only),
`navigate_in_app` (actuation, policy-gated). Maps live per-install under
`data/software_maps/` (git-ignored). Code: `software/knowledge.py` (the map +
fuzzy search), `software/ui_scan.py` (the scan).

- **How it scans (safely):** it reads the accessibility tree (UI Automation) and
  **reveals menus then Escapes back out** — it never activates a normal control.
- **Vision backup (UIA + vision):** custom/Electron apps like CapCut expose
  little to UIA. When the tree comes back thin, MEDO looks at a screenshot of the
  window with the local vision model (qwen2.5-vl) and adds the controls it sees —
  a label, a coarse area, and the model's 0-1000 centre point
  (`software/vision_probe.py`). Backup only, `software.vision_scan: true` by
  default; set it false for UIA-only maps.
- **Clicking a vision-seen control:** `navigate_in_app` tries UIA by name first;
  if that can't reach it (no UIA name), it clicks where the model saw it,
  **re-resolved against the live window** — so a moved window still works as long
  as its layout hasn't changed. Honest limits: a 3B model's labels and points are
  approximate — reliable for *locate*, best-effort for *click*.
- **Assistant-first:** the system prompt tells the brain to PREFER these tools
  (and the connectors) over describing steps.

---

*Intertec note:* MEDO controls local software through a connector abstraction with
a robustness ladder (native API/CLI → app hotkeys → accessibility-tree
automation), every action gated by the central policy engine and the
user-instruction trust boundary.
