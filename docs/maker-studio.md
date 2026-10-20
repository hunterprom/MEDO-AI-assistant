# Maker Studio — generate functional engineering artifacts

MEDO can create **real, functional** engineering artifacts from a spoken/typed
description: electrical schematics, printable 3D models, and (planned) buildable
code. The principle is what makes the output *functional* rather than a useless
picture:

> **Generate code, not artifacts.** For each domain the brain writes code in a
> real, deterministic library (a schematic DSL, code-CAD, source). MEDO then
> **executes that code locally in a sandbox** to produce the genuine artifact — a
> valid schematic + image, a real STL/STEP mesh — and iterates by editing and
> re-running the code. Not "draw a picture of a screw."

Off by default: `studio.enabled: false`. It runs generated (untrusted) code on
your machine — sandboxed — so turning it on opts into that.

## The pipeline (`studio/engine.py`)

```
intent → BRIEF   (ask a clarifying question if a critical spec is missing)
       → CODE    (the brain writes real library code — the source of truth)
       → EXECUTE in a sandbox → artifact
              └─ error? feed the traceback back and retry (bounded: studio.max_retries)
       → CHECK   (printability / sanity → advisories; a 'fail' means NOT shipped as "ready")
       → VERSION + present  (code + artifact saved as vN; iterate = vN+1, history kept)
```

Everything shared lives in the engine; each **domain** (`studio/domains/`)
supplies only the library-specific `system_prompt` / `user_prompt` / `execute` /
`check`. Projects land one-per-creation under `studio.projects_dir`
(`~/MEDO-studio` by default), each a folder with the code, the artifacts, and
numbered versions.

## The domains

- **S2 — Schematics** (`studio/domains/schematic.py`): the brain writes
  **SchemDraw** (Python schematic DSL) → **SVG + PNG**. It's given component/pin
  guidance (ESP32/Arduino GPIO, power rails, LEDs need a series resistor,
  pull-ups). A light **advisory** check flags an LED with no resistor, a missing
  ground, or no power rail — reported, never silently "fixed". (KiCad netlist
  export is an optional follow-on; KiCad CLI is a separate system install.)
  - Say: *"design a circuit for an ESP32 blinking an LED"*, *"draw a schematic for
    a 3.3V regulator"*, *"sketch a circuit for a motor driver"* — a **drafting
    verb** (design / draw / sketch / generate) plus the noun.
  - Wiring *questions* go to `circuit_help` instead, which explains the
    connections and writes the Arduino sketch: *"how do I wire up an LED"*,
    *"wiring diagram for an ultrasonic sensor"*, *"wire up a temperature sensor
    to an Arduino"*. That skill registers first and would otherwise claim the
    drafting phrases too, so it stands aside for an explicit request to **draw**
    one — but only while `studio.enabled`, since with the Studio off an
    explanation beats a phrase that routes nowhere.

- **S3 — 3D models** (`studio/domains/model3d.py`): the brain writes **CadQuery**
  (Python code-CAD) → **STL + STEP**, parametric (key dimensions are named
  variables, so *"make the wall 3 mm"* edits and re-runs). A real
  **printability check** (`studio/domains/mesh.py`, pure-Python) reports the
  triangle count, the bounding box, and whether the mesh is **watertight** — a
  non-manifold result is a `fail` advisory, so it is **never** called
  print-ready.
  - Say: *"model a bracket for a motor"*, *"3D-print a phone stand"*, *"design an
    enclosure for an ESP32"*.

- **S4 — Code building** (`studio/domains/code.py`): the brain writes a
  self-contained Python script (with its own `assert` self-tests), and MEDO
  **runs it in the sandbox to verify it works** — the run *is* the check (exit 0
  = its self-tests passed). A script with no self-tests is flagged (`warn`).
  Generated code only ever runs in the sandbox with declared capabilities, is
  never auto-installed into MEDO, and any out-of-sandbox action needs
  confirmation.
  - Say: *"write me a script that renames files by date"*, *"build a tool to
    convert CSV to JSON"*.

## S5 — projects, HUD, listing

Every creation is a project folder under `studio.projects_dir`. Ask *"show my
studio projects"* or *"open my last 3D model / schematic / script"*
(`studio_projects` skill), or see them in the HUD **CONFIG → Maker Studio ·
Projects** panel (recent projects, domain icon, version count). The companion API
exposes `GET /studio`.

## Security / sandbox (the hard dependency)

Generated code is UNTRUSTED — the model wrote it. `studio/sandbox.py` runs it:

- **gated first** by the central policy engine (`security/policy.py`,
  `RUN_COMMAND` + `WRITE_FILES`, `Actor.MODEL` / `UNTRUSTED` provenance);
- in a **scrubbed environment** (secrets stripped, `HOME`/`TEMP` redirected into
  the project so a `~`- or temp-write can't escape);
- **cwd-isolated** to the project's version dir, with a **hard timeout** that
  kills the whole process tree;
- optional OS-isolation wrapper (`studio.sandbox_cmd`).

Honest limit: without `sandbox_cmd` this is process + env + cwd isolation —
strong for the filesystem and secrets, but it does **not** hard-block network.

## Dependencies (optional)

SchemDraw and CadQuery are **optional** (`requirements-studio.txt`); each skill
degrades with a spoken *"install X to enable this"* when its library is absent —
nothing breaks at startup. CadQuery is large (pulls OpenCascade).

```
.venv\Scripts\pip install -r requirements-studio.txt
```

## Timing — a build is not a spoken turn

Making something real takes **minutes**, nearly all of it the brain writing the
library code. Measured here: a parametric phone stand was 195 s of code writing
alone, ~510 s from the spoken sentence to a finished STL.

Two consequences, both deliberate:

- **The build runs in the background.** MEDO answers *"I'll model that now and
  tell you when it's ready"* immediately and speaks the result when it lands, so
  the assistant stays answerable while it works. A second request during a build
  is declined rather than queued. (A skill constructed without an announcer —
  the tests, the CLI — still builds synchronously.)
- **Code generation gets `studio.generate_timeout_s` (600 s), not the
  conversation timeout.** `llm.cli_timeout_s` is 180 s because nobody waits
  three minutes for a *spoken* answer; that ceiling killed the 195 s phone stand
  just short of finishing, four times over, and reported it as a failure.

A clarifying question is still asked **in the turn that prompted it** — the
brief is computed before anything slow starts, because a question arriving ten
minutes later is no use.

## Platform notes

- **CadQuery's OpenCascade bindings can segfault while unloading** (Windows,
  cadquery 2.8.0 + OCP 7.9.3: exit `0xC0000005`, reproducible on a bare
  `import cadquery`). The script has already finished and both exports have
  landed. `Model3DDomain.execute` therefore judges success on a **complete,
  parseable STL** rather than the exit code, logs the crash, and keeps the
  model. A timeout or a policy refusal is never salvaged this way — those
  killed the script rather than followed it — and a truncated export fails the
  STL reader, so it isn't "any file will do".
- **Generated code runs headless** (`MPLBACKEND=Agg`, set in the sandbox
  environment). schemdraw's matplotlib backend ends a drawing with
  `fig.show()` + `plt.show()`, which starts the MPL event loop and blocks
  forever in a subprocess with no display — every schematic used to run until
  the sandbox timeout killed it.
- **A CLI-agent brain is not asked for a spoken answer here.** `_flatten_for_cli`
  normally ends the prompt with *"Reply with only your spoken answer, as plain
  text"*; the Studio passes `for_speech=False`, because that trailer is the last
  line the agent reads and it overrode *"Output ONLY the Python code"* — a
  request for a phone stand came back as an English paragraph describing one,
  which was then written to `model.py` and executed.

## Honest limits

- **Quality scales with the brain.** A small local model produces simpler designs
  than a strong one. Set a stronger model for hard generation via `studio.model`
  — but note it names a model of the **active provider**, not another provider:
  the Brain builds its client from `settings.llm`, so with `llm.provider:
  claude-code` a `studio.model` of `qwen3:30b` would be passed to the Claude CLI,
  not to Ollama. To generate locally, switch the brain. Never expect
  frontier-level CAD/EDA from a 3B local model.
- MEDO **assists**; it does not replace an engineer's review. Complex or
  safety-critical designs need checking. The advisories are sanity checks, not a
  sign-off.
- Config: `studio.enabled`, `projects_dir`, `model`, `max_retries`,
  `exec_timeout_s`, `sandbox_cmd`, `keep_versions`.

---

*Intertec note:* MEDO generates functional engineering artifacts by having the
LLM write real library code (SchemDraw / CadQuery / source), executing it in a
security-sandboxed, self-correcting loop, with parametric iteration and
printability/sanity checks — not opaque image generation.
