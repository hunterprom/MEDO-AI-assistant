"""S4 — keep an approved plan as a named routine ("save this as demo prep").

A saved Cowork task IS a routine: an ordered list of skills with arguments,
triggered by a phrase. So this writes a `routines/*.yaml` and hands it to
:mod:`core.composer` rather than growing a second, parallel store of chains —
one loader, one validator, one set of rules.

Composer's rules are the reason this can refuse. A routine may not contain a
step that needs a spoken confirmation (a prompt in the middle of a chain has
nobody to answer it when the chain fires on a timer). A Cowork plan CAN contain
one, because a person is standing there approving it. So a plan with a
confirming step is a perfectly good plan and an invalid routine, and the honest
outcome is to say which step and why — not to save it with the confirmation
quietly dropped.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from cowork.plan import Plan

logger = logging.getLogger(__name__)

#: A routine file name: lower-case, dashes, nothing that can escape the folder.
_SLUG = re.compile(r"[^a-z0-9]+")
#: Words that trail a save request but are not part of the name, with whatever
#: punctuation came with them. Applied repeatedly — "demo prep routine, please"
#: has two of them, and one pass left the comma behind in the file name.
_TRAILING = re.compile(r"[\s,.]*\b(?:routine|рутина|please|те\s+молам)\b[\s,.]*$",
                       re.IGNORECASE)


def slug(name: str) -> str:
    """The file name for a routine called ``name``.

    Macedonian is transliterated rather than stripped: "подготовка" becomes
    ``podgotovka.yaml``. Dropping the non-latin characters instead would leave
    an empty slug, and an empty slug is a file called ``.yaml`` — a saved
    routine the user can never find again.
    """
    from core import mk

    return _SLUG.sub("-", mk.to_latin(name or "").strip().lower()).strip("-")


def clean_name(name: str) -> str:
    """"as 'demo prep' routine, please" -> "demo prep"."""
    text = " ".join((name or "").split())
    for _ in range(3):                       # bounded: each pass only shortens
        stripped = _TRAILING.sub("", text).strip().strip("\"'‘’").strip()
        if stripped == text:
            break
        text = stripped
    return " ".join(text.split())


def plan_to_routine(plan: Plan, name: str) -> dict[str, Any]:
    """The routine YAML for an approved plan. Only steps MEDO can actually do."""
    steps: list[dict[str, Any]] = []
    for step in plan.steps:
        if not step.possible:
            continue                       # an impossible step is not a step
        entry: dict[str, Any] = {"skill": step.capability}
        if step.args:
            entry["args"] = {str(k): v for k, v in step.args.items()}
        steps.append(entry)
    return {
        "name": name,
        "description": f"Saved from a task: {plan.task}",
        "trigger": {"phrase": [name]},
        "steps": steps,
    }


def save_plan_as_routine(plan: Plan, name: str, registry: Any,
                         directory: str | Path) -> tuple[str, str]:
    """Write and register the routine. Returns ``(name, error)`` — one or the other.

    Registering it live matters: without it MEDO would say "say it any time"
    about a phrase that only works after the next restart, which is the kind of
    small lie that makes a user stop trusting the big statements.
    """
    from core.composer import RoutineError, parse_routine, register_routines

    name = clean_name(name)
    if not name:
        return "", "I need a name for it"
    if not slug(name):
        # Nothing left to build a file name out of (punctuation, emoji, a
        # script with no transliteration). Refusing beats writing ".yaml".
        return "", "I can't make a file name out of that — try another name"
    data = plan_to_routine(plan, name)
    if not data["steps"]:
        return "", "there's nothing in that plan I can actually do"

    try:
        routine = parse_routine(data, source=f"{slug(name)}.yaml")
    except RoutineError as exc:
        return "", str(exc)

    path = Path(directory) / f"{slug(name)}.yaml"
    if path.exists():
        return "", f"there's already a routine file called {path.name}"

    registered, refused, warnings = register_routines(registry, [routine])
    if not registered:
        # Composer's own words — a step that needs confirming, or a name clash.
        return "", (refused[0] if refused else "it couldn't be registered")

    import yaml

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8")
    except OSError as exc:
        # It IS registered for this session; say so rather than claim a save
        # that did not happen.
        logger.warning("cowork: could not write %s: %s", path, exc)
        return name, f"I couldn't write the file ({exc}) — it works until restart"
    for warning in warnings:
        logger.info("cowork: %s", warning)
    return name, ""
