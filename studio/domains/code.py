"""S4 — code building: the brain writes a self-contained script, MEDO RUNS it in
the sandbox to verify it works (its own asserts/self-tests must pass), and checks
it. Iterate by editing ("add a --csv flag"). Uses the same engine loop, so a
failing run feeds the traceback back and self-corrects.

Scope + safety: self-contained scripts only; they run ONLY in the sandbox with
declared capabilities, never network/file/system access, and are never
auto-installed into MEDO (that's the separate plugin flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from studio.domain import Advisory, Brief, ExecOutcome

_SYSTEM = (
    "You are a senior Python engineer. Write a COMPLETE, self-contained Python "
    "script that does exactly what's asked. Rules:\n"
    "- Standard library only unless truly necessary.\n"
    "- Include a few `assert`-based self-tests (or `test_*` functions you call at "
    "the end) that verify the CORE behaviour, plus an `if __name__ == \"__main__\":`"
    " demo.\n"
    "- The script MUST run to completion with exit code 0 when executed (all "
    "asserts pass). Do NOT read the network, read/write files outside the current "
    "directory, or install anything.\n"
    "- Output ONLY the Python code — no prose, no markdown fences."
)


class CodeDomain:
    name = "code"
    filename = "main.py"

    def build_brief(self, description: str, answers: dict) -> Brief:
        d = (description or "").strip()
        if len(d) < 3:
            return Brief(description=d, question="What should the script do?")
        return Brief(description=d)

    def system_prompt(self) -> str:
        return _SYSTEM

    def user_prompt(self, brief: Brief, prior_code: str, error: str) -> str:
        if prior_code.strip():
            prompt = (f"Here is the current script:\n\n{prior_code}\n\nApply this "
                      f"change and return the FULL updated script: {brief.description}")
        else:
            prompt = f"Write a Python script that: {brief.description}"
        if error:
            prompt += (f"\n\nThe previous script FAILED when run — fix it and "
                       f"return the full corrected script. Error:\n{error}")
        return prompt

    async def execute(self, code: str, workdir: Path, sandbox) -> ExecOutcome:
        # Running it IS the verification: its own asserts must pass (exit 0).
        workdir = Path(workdir)
        result = await sandbox.run_python(workdir / self.filename, workdir)
        if result.ok:
            return ExecOutcome(ok=True, artifacts=[self.filename],
                               stdout=result.stdout, stderr=result.stderr,
                               preview=self.filename)
        return ExecOutcome(ok=False, artifacts=[self.filename],
                           stdout=result.stdout,
                           stderr=result.stderr or "the script failed to run")

    def check(self, outcome: ExecOutcome, workdir: Path) -> List[Advisory]:
        try:
            code = (Path(workdir) / self.filename).read_text(
                encoding="utf-8").lower()
        except Exception:
            return []
        if "assert" in code or "def test" in code or "unittest" in code:
            return [Advisory("info", "it runs and its self-tests pass")]
        return [Advisory("warn", "the script runs, but it has no self-tests — I "
                                 "can only confirm it executes, not that it's "
                                 "correct")]
