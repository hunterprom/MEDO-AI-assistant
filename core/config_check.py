"""Catching a bad config.yaml at startup, and saying what is wrong with it.

Pydantic already rejects a value of the wrong TYPE. What it can't catch is a
value of the right type that makes no sense — ``max_members: 0`` (a council of
nobody), ``semantic_threshold: 95`` (someone thought it was a percentage),
``port: 70000``, a TLS certificate without its key. Those load cleanly and then
fail hours later as a puzzling silence: no expert answers, nothing routes
semantically, the API never binds.

Two rules shape this module.

**Report everything at once.** Fixing a config file one error per restart is
miserable, and a voice assistant's restart costs a model reload. Every problem
found is collected and printed together.

**Say the value, the key, and the fix.** "validation error" is useless at 2am.
Each problem names the setting as it appears in ``config.yaml``, what it is set
to, and what would be acceptable.

The distinction between an error and a warning is whether MEDO can *honestly*
run: an unusable value is fatal, a merely surprising one is a warning that lets
startup continue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """config.yaml holds values MEDO cannot run with."""


@dataclass(frozen=True)
class Problem:
    """One thing wrong with the configuration."""

    key: str                 # dotted path as it appears in config.yaml
    value: object
    message: str             # what is acceptable, and why
    fatal: bool = True

    def __str__(self) -> str:
        return f"  {self.key} = {self.value!r}\n      {self.message}"


def _in_range(problems: list[Problem], key: str, value, low, high, *,
              why: str = "", fatal: bool = True) -> None:
    if value is None:
        return
    if not (low <= value <= high):
        problems.append(Problem(
            key, value,
            f"must be between {low} and {high}." + (f" {why}" if why else ""),
            fatal))


def check_settings(settings) -> list[Problem]:
    """Every problem with ``settings``, in the order a person would fix them."""
    problems: list[Problem] = []

    # --- the router's three tiers -------------------------------------------
    router = settings.router
    _in_range(problems, "router.semantic_threshold", router.semantic_threshold,
              0.0, 1.0,
              why="It is a cosine similarity, not a percentage — 0.6 is a "
                  "sensible match bar.")
    _in_range(problems, "router.semantic_margin", router.semantic_margin, 0.0, 1.0,
              why="How far the best match must beat the runner-up before the "
                  "tier commits.")

    # --- the council ---------------------------------------------------------
    council = settings.council
    if council.max_members < 1:
        problems.append(Problem(
            "council.max_members", council.max_members,
            "must be at least 1 — a council of nobody answers nothing. "
            "Set council.enabled: false to switch the feature off instead."))
    elif council.max_members > 8:
        problems.append(Problem(
            "council.max_members", council.max_members,
            "is very high: every member is a separate model round-trip, so this "
            "makes one question take that many times as long. 3 is the default.",
            fatal=False))
    _in_range(problems, "council.semantic_threshold", council.semantic_threshold,
              0.0, 1.0,
              why="It is a cosine similarity; 0.45 is the default bar for "
                  "'this question is in that expert's field'.")
    if council.semantic_tie_band < 0:
        problems.append(Problem(
            "council.semantic_tie_band", council.semantic_tie_band,
            "must not be negative — it is the width of the band inside which "
            "two similarity scores count as equal."))
    for key in council.disabled:
        if key not in council.extra:
            from core.council import COUNCIL

            if key not in {m.key for m in COUNCIL}:
                problems.append(Problem(
                    "council.disabled", key,
                    "names a specialist that does not exist, so it disables "
                    "nothing. Check the spelling against core/council.py.",
                    fatal=False))
    for key, spec in (council.extra or {}).items():
        if not isinstance(spec, dict) or not spec.get("prompt"):
            problems.append(Problem(
                f"council.extra.{key}", spec,
                "needs at least a 'prompt' — the paragraph telling the model "
                "who this specialist is. Without it the entry is skipped.",
                fatal=False))

    # --- memory --------------------------------------------------------------
    memory = settings.memory
    if memory.max_turns < 1:
        problems.append(Problem(
            "memory.max_turns", memory.max_turns,
            "must be at least 1, or MEDO forgets what you just said mid-turn."))
    if memory.max_facts < 0:
        problems.append(Problem(
            "memory.max_facts", memory.max_facts,
            "cannot be negative. Use 0 to inject no remembered facts."))
    if memory.max_turns > 50:
        problems.append(Problem(
            "memory.max_turns", memory.max_turns,
            "is more conversation than the local brain's context window holds "
            "(qwen3:30b runs at num_ctx 4096 here) — the oldest turns will be "
            "silently dropped by the model, not by MEDO.", fatal=False))

    # --- the companion API ---------------------------------------------------
    remote = settings.remote
    _in_range(problems, "remote.port", remote.port, 1, 65535,
              why="A TCP port. Below 1024 also needs administrator rights.")
    if remote.rate_limit_per_minute < 0:
        problems.append(Problem(
            "remote.rate_limit_per_minute", remote.rate_limit_per_minute,
            "cannot be negative. Use 0 to disable throttling entirely."))
    if remote.rate_limit_burst < 0:
        problems.append(Problem(
            "remote.rate_limit_burst", remote.rate_limit_burst,
            "cannot be negative. Use 0 to derive the burst from the rate."))
    if bool(remote.tls_certfile) != bool(remote.tls_keyfile):
        problems.append(Problem(
            "remote.tls_certfile / remote.tls_keyfile",
            (remote.tls_certfile, remote.tls_keyfile),
            "must be set together: a certificate is unusable without its "
            "private key. Leave BOTH empty to have MEDO mint a self-signed "
            "pair on first run."))
    for field in ("tls_certfile", "tls_keyfile"):
        path = getattr(remote, field)
        if path and not Path(path).expanduser().exists():
            problems.append(Problem(
                f"remote.{field}", path,
                "points at a file that does not exist. MEDO will fall back to "
                "plain HTTP and say so in the log.", fatal=False))
    if remote.enabled and remote.host not in ("127.0.0.1", "localhost", "::1") \
            and not remote.auth_enabled:
        problems.append(Problem(
            "remote.auth_enabled", remote.auth_enabled,
            "is off while the API listens on the network "
            f"({remote.host}) — anything on the LAN can then drive MEDO. "
            "Turn auth back on, or bind to 127.0.0.1."))

    # --- the agents that act -------------------------------------------------
    if settings.browser.max_steps < 1:
        problems.append(Problem(
            "browser.max_steps", settings.browser.max_steps,
            "must be at least 1, or a web task can never do anything. Use "
            "browser.enabled: false to switch it off."))
    if settings.vision.agent_max_steps < 1:
        problems.append(Problem(
            "vision.agent_max_steps", settings.vision.agent_max_steps,
            "must be at least 1. Use vision.agent_enabled: false to switch the "
            "screen agent off."))

    # --- self-dev ------------------------------------------------------------
    self_dev = settings.self_dev
    if self_dev.max_diff_bytes < 0 or self_dev.max_files_changed < 0:
        problems.append(Problem(
            "self_dev.max_diff_bytes / max_files_changed",
            (self_dev.max_diff_bytes, self_dev.max_files_changed),
            "cannot be negative. Use 0 for 'no limit' (not recommended: an "
            "unreviewable diff makes the approval step meaningless)."))
    if self_dev.enabled and self_dev.auto_apply:
        problems.append(Problem(
            "self_dev.auto_apply", True,
            "lets MEDO merge its own code changes with no one reading the "
            "diff. The approval is still recorded, but you are making it in "
            "advance for every future change.", fatal=False))
    if self_dev.enabled and not self_dev.denylist:
        problems.append(Problem(
            "self_dev.denylist", self_dev.denylist,
            "is empty while self-dev is on, so MEDO may rewrite its own safety "
            "code, its config, and the companion API. Keep the default list."))

    # --- safety --------------------------------------------------------------
    if not settings.safety.confirm_destructive:
        problems.append(Problem(
            "safety.confirm_destructive", False,
            "means destructive actions run IMMEDIATELY, with no 'are you "
            "sure?'. The PC-control switch and the policy engine still apply.",
            fatal=False))
    if getattr(settings.security, "yolo", False):
        problems.append(Problem(
            "security.yolo", True,
            "turns the WHOLE security layer off. It is a development flag and "
            "must never be committed as on.", fatal=False))

    # --- cowork --------------------------------------------------------------
    cowork = getattr(settings, "cowork", None)
    if cowork is not None and cowork.enabled:
        if cowork.max_steps < 1:
            problems.append(Problem(
                "cowork.max_steps", cowork.max_steps,
                "must be at least 1 — at 0 every task is refused as too big. "
                "Set cowork.enabled: false to switch the feature off instead."))
        elif cowork.max_steps > 12:
            problems.append(Problem(
                "cowork.max_steps", cowork.max_steps,
                "is very high. A plan this long is read out loud before you "
                "approve it, and a local brain rarely gets that many steps "
                "right — 6 is the default for a reason.", fatal=False))
        if cowork.allow_unattended:
            problems.append(Problem(
                "cowork.allow_unattended", True,
                "lets a plan run with nobody there to approve it. The approval "
                "is Cowork's whole safety model; on a schedule the only thing "
                "that could answer a prompt is the schedule itself.",
                fatal=False))

    # --- logging -------------------------------------------------------------
    if settings.logging.correction_window_s < 0:
        problems.append(Problem(
            "logging.correction_window_s", settings.logging.correction_window_s,
            "cannot be negative. Use 0 to stop attributing corrections to the "
            "previous route."))

    return problems


def format_problems(problems: list[Problem]) -> str:
    """The message a person reads when startup refuses."""
    fatal = [p for p in problems if p.fatal]
    warnings = [p for p in problems if not p.fatal]
    lines: list[str] = []
    if fatal:
        lines.append(f"config.yaml has {len(fatal)} setting(s) MEDO can't run with:")
        lines += [str(p) for p in fatal]
    if warnings:
        lines.append(f"{len(warnings)} setting(s) worth a look:")
        lines += [str(p) for p in warnings]
    return "\n".join(lines)


def validate_settings(settings, *, strict: bool = True) -> list[Problem]:
    """Check ``settings``; raise :class:`ConfigError` on anything fatal.

    Warnings are logged, never raised — MEDO starting up and telling you that
    ``security.yolo`` is on is more useful than MEDO refusing to start.
    """
    problems = check_settings(settings)
    warnings = [p for p in problems if not p.fatal]
    if warnings:
        logger.warning("configuration warnings:\n%s", format_problems(warnings))
    fatal = [p for p in problems if p.fatal]
    if fatal and strict:
        raise ConfigError(format_problems(fatal))
    return problems


def _main() -> int:  # pragma: no cover - operator utility
    """``python -m core.config_check`` — validate config.yaml and report."""
    from core.config import load_settings

    try:
        settings = load_settings(validate=False)
    except Exception as exc:
        print(f"config.yaml could not be loaded at all:\n{exc}")
        return 2
    problems = check_settings(settings)
    if not problems:
        print("config.yaml looks good.")
        return 0
    print(format_problems(problems))
    return 1 if any(p.fatal for p in problems) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
