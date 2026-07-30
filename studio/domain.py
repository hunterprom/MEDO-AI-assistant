"""The Domain interface each Maker Studio skill implements (schematic / 3D / code).

Keeps the engine domain-agnostic. A domain knows FOUR things; the shared
orchestration (generate → sandbox-execute → self-correct → check → version) all
lives in ``studio.engine``:

1. ``build_brief`` — turn a description (+ any answered questions) into a
   structured Brief; set ``Brief.question`` to ask for a critical missing spec
   instead of guessing.
2. ``system_prompt`` / ``user_prompt`` — steer the brain to write code in the
   target library (the user prompt also carries prior code + the last error, so
   the self-correction loop and parametric iteration reuse the same call).
3. ``execute`` — run the generated code (via the sandbox) into a real artifact.
4. ``check`` — sanity/printability/tests → advisories, reported honestly (a
   ``fail`` advisory means the engine will NOT claim the result is "ready").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable


@dataclass
class Brief:
    """A structured build request. ``question`` non-empty => the engine returns
    it to be asked, and generation waits for the answer."""

    description: str
    requirements: dict = field(default_factory=dict)
    question: str = ""


@dataclass
class Advisory:
    """A reported check result. ``level``: 'fail' (blocks a 'ready' claim) |
    'warn' (surface it) | 'info'."""

    level: str
    message: str


@dataclass
class ExecOutcome:
    """What executing the generated code produced."""

    ok: bool
    artifacts: List[str] = field(default_factory=list)   # files produced (relative)
    stdout: str = ""
    stderr: str = ""
    preview: Optional[str] = None                         # a HUD preview asset (rel)


@runtime_checkable
class Domain(Protocol):
    #: Stable domain id, e.g. "schematic" | "model3d" | "code".
    name: str
    #: The code file the brain writes, e.g. "design.py".
    filename: str

    def build_brief(self, description: str, answers: dict) -> Brief: ...

    def system_prompt(self) -> str: ...

    def user_prompt(self, brief: Brief, prior_code: str, error: str) -> str: ...

    async def execute(self, code: str, workdir: Path, sandbox) -> ExecOutcome: ...

    def check(self, outcome: ExecOutcome, workdir: Path) -> List[Advisory]: ...
