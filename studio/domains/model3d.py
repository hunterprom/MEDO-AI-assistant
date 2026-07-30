"""S3 — functional 3D models: the brain writes CadQuery, MEDO runs it into a real
STL + STEP, then checks printability. Parametric (dimensions are named variables
the brain exposes, so "make the wall 3 mm" edits + re-runs). Degrades with an
honest "install cadquery" message when the library isn't present.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List

from studio.domain import Advisory, Brief, ExecOutcome
from studio.domains.mesh import inspect_stl

_SYSTEM = (
    "You are a mechanical CAD engineer. Write a COMPLETE CadQuery (Python) script "
    "that builds the requested part parametrically and exports it. Rules:\n"
    "- `import cadquery as cq`.\n"
    "- Put the key dimensions as named variables (millimetres) at the TOP so they "
    "can be tuned, and build from them.\n"
    "- Build ONE watertight solid in a variable named `result` (a cq.Workplane).\n"
    "- Export EXACTLY these two files into the current directory:\n"
    "    cq.exporters.export(result, 'model.stl')\n"
    "    cq.exporters.export(result, 'model.step')\n"
    "- Prefer manufacturable, printable geometry: a flat base, no needle-thin "
    "walls, fillets/chamfers where sensible.\n"
    "- Output ONLY the Python code — no prose, no markdown fences."
)


class Model3DDomain:
    name = "model3d"
    filename = "model.py"

    def build_brief(self, description: str, answers: dict) -> Brief:
        d = (description or "").strip()
        if len(d) < 3:
            return Brief(description=d, question="What should I model?")
        return Brief(description=d)

    def system_prompt(self) -> str:
        return _SYSTEM

    def user_prompt(self, brief: Brief, prior_code: str, error: str) -> str:
        if prior_code.strip():
            prompt = (f"Here is the current CadQuery script:\n\n{prior_code}\n\n"
                      f"Apply this change and return the FULL updated script: "
                      f"{brief.description}")
        else:
            prompt = f"Model this part with CadQuery: {brief.description}"
        if error:
            prompt += (f"\n\nThe previous script FAILED — fix it and return the "
                       f"full corrected script. Error:\n{error}")
        return prompt

    async def execute(self, code: str, workdir: Path, sandbox) -> ExecOutcome:
        if importlib.util.find_spec("cadquery") is None:
            return ExecOutcome(ok=False, stderr=(
                "the 'cadquery' library isn't installed on this machine — run "
                "'pip install cadquery' to enable 3D modelling"))
        workdir = Path(workdir)
        result = await sandbox.run_python(workdir / self.filename, workdir)
        arts = [f for f in ("model.stl", "model.step") if (workdir / f).exists()]
        if result.ok and arts:
            return ExecOutcome(ok=True, artifacts=arts, stdout=result.stdout,
                               stderr=result.stderr,
                               preview=("model.stl" if "model.stl" in arts
                                        else arts[0]))
        err = (result.stderr or ("the script ran but exported no STL/STEP"
               if result.ok else "the script failed to run"))
        return ExecOutcome(ok=False, artifacts=arts, stdout=result.stdout,
                           stderr=err)

    def check(self, outcome: ExecOutcome, workdir: Path) -> List[Advisory]:
        stl = Path(workdir) / "model.stl"
        if not stl.exists():
            return []
        info = inspect_stl(stl)
        if info is None:
            return [Advisory("warn", "I couldn't inspect the STL to verify it.")]
        dx, dy, dz = info["bbox"]
        adv: List[Advisory] = []
        if not info["watertight"]:
            adv.append(Advisory("fail",
                f"the mesh isn't watertight ({info['open_edges']} open edges), so "
                f"it isn't print-ready as-is"))
        else:
            adv.append(Advisory("info",
                f"watertight · {info['triangles']} triangles · "
                f"{dx:.1f}×{dy:.1f}×{dz:.1f} mm"))
        if max(dx, dy, dz) > 250:
            adv.append(Advisory("warn",
                f"it's {max(dx, dy, dz):.0f} mm on its longest side — bigger than a "
                f"typical print bed; you may need to scale or split it"))
        if 0 < min(dx, dy, dz) < 0.4 and info["watertight"]:
            adv.append(Advisory("warn",
                "it has a sub-0.4 mm dimension — below most nozzle widths"))
        return adv
