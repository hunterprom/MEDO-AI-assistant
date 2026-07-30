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
    a 3.3V regulator"*, *"wire up a temperature sensor to an Arduino"*.

- **S3 — 3D models** (`studio/domains/model3d.py`): the brain writes **CadQuery**
  (Python code-CAD) → **STL + STEP**, parametric (key dimensions are named
  variables, so *"make the wall 3 mm"* edits and re-runs). A real
  **printability check** (`studio/domains/mesh.py`, pure-Python) reports the
  triangle count, the bounding box, and whether the mesh is **watertight** — a
  non-manifold result is a `fail` advisory, so it is **never** called
  print-ready.
  - Say: *"model a bracket for a motor"*, *"3D-print a phone stand"*, *"design an
    enclosure for an ESP32"*.

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

## Honest limits

- **Quality scales with the brain.** A small local model produces simpler designs
  than a strong one. Set a stronger model for hard generation via `studio.model`
  (a cloud brain honours the privacy gate). Never expect frontier-level CAD/EDA
  from a 3B local model.
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
