"""S2 — electrical schematics: the brain writes SchemDraw (a Python schematic
DSL), MEDO runs it into a real SVG + PNG, then runs light sanity advisories.
Iterate by editing the code ("add a pull-up on GPIO4"). Degrades with an honest
"install schemdraw" message when the library isn't present.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List

from studio.domain import Advisory, Brief, ExecOutcome

_SYSTEM = (
    "You are an electronics engineer. Write a COMPLETE SchemDraw (Python) script "
    "that draws the requested schematic and saves it. Rules:\n"
    "- `import schemdraw` and `import schemdraw.elements as elm`.\n"
    "- Build the drawing, then save BOTH files into the current directory:\n"
    "    with schemdraw.Drawing() as d:\n"
    "        # ... add and connect elements ...\n"
    "        d.save('schematic.svg')\n"
    "        d.save('schematic.png')\n"
    "- Label parts and values (R1 220Ω, C1 100nF, U1). Use correct pin roles for "
    "microcontrollers (ESP32/Arduino GPIO, 3V3/5V/VCC, GND).\n"
    "- Include power and ground. Put a current-limiting resistor in series with "
    "any LED. Add pull-up/pull-down resistors where a pin needs one.\n"
    "- Output ONLY the Python code — no prose, no markdown fences."
)


class SchematicDomain:
    name = "schematic"
    filename = "schematic.py"

    def build_brief(self, description: str, answers: dict) -> Brief:
        d = (description or "").strip()
        if len(d) < 3:
            return Brief(description=d, question="What circuit should I draw?")
        return Brief(description=d)

    def system_prompt(self) -> str:
        return _SYSTEM

    def user_prompt(self, brief: Brief, prior_code: str, error: str) -> str:
        if prior_code.strip():
            prompt = (f"Here is the current SchemDraw script:\n\n{prior_code}\n\n"
                      f"Apply this change and return the FULL updated script: "
                      f"{brief.description}")
        else:
            prompt = f"Draw this schematic with SchemDraw: {brief.description}"
        if error:
            prompt += (f"\n\nThe previous script FAILED — fix it and return the "
                       f"full corrected script. Error:\n{error}")
        return prompt

    async def execute(self, code: str, workdir: Path, sandbox) -> ExecOutcome:
        if importlib.util.find_spec("schemdraw") is None:
            return ExecOutcome(ok=False, stderr=(
                "the 'schemdraw' library isn't installed on this machine — run "
                "'pip install schemdraw' to enable schematic design"))
        workdir = Path(workdir)
        result = await sandbox.run_python(workdir / self.filename, workdir)
        arts = [f for f in ("schematic.svg", "schematic.png")
                if (workdir / f).exists()]
        if result.ok and any(a.endswith(".svg") for a in arts):
            return ExecOutcome(ok=True, artifacts=arts, stdout=result.stdout,
                               stderr=result.stderr, preview="schematic.svg")
        err = (result.stderr or ("the script ran but drew no schematic image"
               if result.ok else "the script failed to run"))
        return ExecOutcome(ok=False, artifacts=arts, stdout=result.stdout,
                           stderr=err)

    def check(self, outcome: ExecOutcome, workdir: Path) -> List[Advisory]:
        """Advisory sanity from the generated code (no netlist parse) — reported,
        never silently 'fixed'."""
        try:
            code = (Path(workdir) / self.filename).read_text(
                encoding="utf-8").lower()
        except Exception:
            return []
        adv: List[Advisory] = []
        if "led" in code and "resistor" not in code:
            adv.append(Advisory("warn",
                "I see an LED but no resistor — check there's a current-limiting "
                "resistor so it isn't over-driven"))
        if not any(g in code for g in ("ground", "gnd", ".ground")):
            adv.append(Advisory("warn",
                "no ground reference is obvious — verify the return path"))
        if not any(p in code for p in ("vcc", "3v3", "3.3", "5v", "vdd", "vin",
                                       "power", "battery", "supply")):
            adv.append(Advisory("warn", "no power rail is obvious — check the "
                                        "supply connection"))
        return adv
