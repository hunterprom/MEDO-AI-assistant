"""MEDO Link registry: validate manifests, mint skills, dispatch commands (M9).

A device POSTs its manifest to ``/link/register`` (token-authed); every
capability is hot-registered as BOTH an LLM tool and optional fast-path
patterns by wrapping it in a :class:`DeviceCapabilitySkill` — the exact same
``Skill`` contract plugins use, so one code path serves built-ins, plugins,
MCP tools, and devices. Capabilities flagged ``requires_confirmation`` ride
MEDO's existing spoken yes/no gate (bilingual since M2) with zero extra code:
the wrapper skill simply follows the same protocol destructive skills do.

Dispatch: websocket devices get commands pushed; http_poll devices fetch
``GET /link/commands/{device_id}`` (which doubles as the liveness heartbeat)
and answer ``POST /link/result``. A device that hasn't polled or connected
recently is *offline* — its tools stay registered and invocations return a
spoken "offline" line. A dispatched command that gets no result within the
timeout returns "the device didn't respond".

Manifests persist in the shared sqlite DB and are re-registered on startup,
so a reboot doesn't forget the fleet. Validation is a small stdlib checker
mirroring ``link/manifest-schema.json`` (the schema file is the formal
reference; no jsonschema dependency).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

_DEVICE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")
_CAP_NAME = re.compile(r"^[a-z0-9][a-z0-9_]{0,31}$")
#: Heuristic for a catastrophic-backtracking regex: a quantified group whose
#: body holds a quantifier, optional, or alternation — (a+)+, (a*)*, (.+)*,
#: (a?)+, (a|a)+. Not exhaustive, but it rejects the classic ReDoS shapes a
#: device manifest could smuggle in.
_REDOS = re.compile(r"\([^()]*[+*?|][^()]*\)[+*]")
TRANSPORTS = ("http_poll", "websocket")

#: A device is offline when it hasn't polled/connected for this long.
OFFLINE_AFTER_S = 20.0
#: How long a dispatched command waits for the device's result.
COMMAND_TIMEOUT_S = 10.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS link_devices (
    device_id TEXT PRIMARY KEY,
    manifest TEXT NOT NULL,
    registered_at REAL NOT NULL
);
"""


def validate_manifest(data: Any) -> list[str]:
    """Errors in a manifest dict; empty list = valid. Pure and unit-tested.

    Mirrors ``link/manifest-schema.json`` — update both together.
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["manifest must be a JSON object"]
    device_id = data.get("device_id")
    if not isinstance(device_id, str) or not _DEVICE_ID.match(device_id):
        errors.append("device_id must match ^[a-z0-9][a-z0-9_-]{1,31}$")
    name = data.get("name")
    if not isinstance(name, str) or not (1 <= len(name) <= 60):
        errors.append("name must be a 1-60 char string")
    if data.get("transport", "http_poll") not in TRANSPORTS:
        errors.append("transport must be http_poll or websocket")
    caps = data.get("capabilities")
    if not isinstance(caps, list) or not (1 <= len(caps) <= 32):
        errors.append("capabilities must be a list of 1-32 entries")
        caps = []
    seen_names: set[str] = set()
    for i, cap in enumerate(caps):
        where = f"capabilities[{i}]"
        if not isinstance(cap, dict):
            errors.append(f"{where} must be an object")
            continue
        cname = cap.get("name")
        if not isinstance(cname, str) or not _CAP_NAME.match(cname):
            errors.append(f"{where}.name must match ^[a-z0-9][a-z0-9_]{{0,31}}$")
        elif cname in seen_names:
            # Duplicate names -> two skills with the same id; the second
            # register() raises ValueError mid-install, leaving inconsistent
            # state. Reject up front.
            errors.append(f"{where}.name {cname!r} is a duplicate in this manifest")
        elif isinstance(cname, str):
            seen_names.add(cname)
        desc = cap.get("description")
        if not isinstance(desc, str) or not (1 <= len(desc) <= 300):
            errors.append(f"{where}.description must be a 1-300 char string")
        params = cap.get("params", {})
        if not isinstance(params, dict):
            errors.append(f"{where}.params must be an object")
        pats = cap.get("fast_patterns", []) or []
        if not isinstance(pats, list):
            errors.append(f"{where}.fast_patterns must be a list")
            pats = []
        if len(pats) > 8:                            # schema maxItems: 8
            errors.append(f"{where}.fast_patterns allows at most 8 patterns")
        for pat in pats:
            if not isinstance(pat, str):
                errors.append(f"{where}.fast_patterns entries must be strings")
                continue
            if len(pat) > 120:                       # a fast-path regex is short
                errors.append(f"{where}.fast_patterns entry is too long (max 120 chars)")
                continue
            if _REDOS.search(pat):
                # A nested quantifier ((a+)+) can hang the router for minutes on
                # one utterance — a trivial DoS from one manifest field.
                errors.append(
                    f"{where}.fast_patterns {pat!r} has a nested quantifier "
                    f"(catastrophic-backtracking risk) — rewrite it")
                continue
            try:
                re.compile(pat, re.IGNORECASE)
            except (re.error, TypeError) as exc:
                errors.append(f"{where}.fast_patterns {pat!r} is not a regex: {exc}")
        if not isinstance(cap.get("requires_confirmation", False), bool):
            errors.append(f"{where}.requires_confirmation must be a boolean")
    return errors


class _DeviceState:
    """Runtime side of one device: liveness, command queue, pending results."""

    def __init__(self) -> None:
        self.last_seen = 0.0                      # monotonic; 0 = never
        self.queue: list[dict] = []               # commands awaiting a poll
        self.futures: dict[str, asyncio.Future] = {}
        self.ws = None                            # live WebSocketResponse or None

    def online(self) -> bool:
        if self.ws is not None:
            return True
        return (time.monotonic() - self.last_seen) < OFFLINE_AFTER_S \
            and self.last_seen > 0.0


class DeviceCapabilitySkill(Skill):
    """One device capability speaking the normal Skill contract."""

    def __init__(self, link: "LinkRegistry", device_id: str, device_name: str,
                 cap: dict) -> None:
        self._link = link
        self._device_id = device_id
        self._device_name = device_name
        self._cap = cap
        self.name = f"{device_id}_{cap['name']}"
        self.description = f"[{device_name}] {cap['description']}"
        self.patterns = [re.compile(p, re.IGNORECASE)
                         for p in cap.get("fast_patterns", []) or []]
        self.requires_confirmation = bool(cap.get("requires_confirmation", False))

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": dict(self._cap.get("params", {}) or {}),
                },
            },
        }

    async def execute(self, request: SkillRequest) -> SkillResult:
        # Same protocol as built-in destructive skills: ask once, act on yes.
        if self.requires_confirmation and not request.context.get("confirmed"):
            return SkillResult(
                f"This will make {self._device_name} {self._cap['name'].replace('_', ' ')}. "
                f"Are you sure?",
                needs_confirmation=True,
            )
        return await self._link.dispatch(
            self._device_id, self._cap["name"], request.args or {}
        )


class LinkRegistry:
    """Validates, persists, and hot-registers device manifests; routes commands."""

    def __init__(self, skills: SkillRegistry, db_path: str | Path,
                 command_timeout_s: float = COMMAND_TIMEOUT_S) -> None:
        self._skills = skills
        self._db_path = str(db_path)
        self._timeout = command_timeout_s
        self._manifests: dict[str, dict] = {}
        self._state: dict[str, _DeviceState] = {}

    # -- persistence ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.executescript(_SCHEMA)
        return conn

    def load_persisted(self) -> int:
        """Re-register every stored manifest (startup). Returns device count."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT manifest FROM link_devices ORDER BY registered_at").fetchall()
        count = 0
        for (raw,) in rows:
            try:
                self._install(json.loads(raw))
                count += 1
            except Exception:  # one bad row must not block startup
                logger.exception("could not restore a persisted device manifest")
        return count

    # -- registration ---------------------------------------------------------

    def register(self, manifest: dict) -> list[str]:
        """Validate + persist + hot-register. Returns validation errors ([] = ok)."""
        errors = validate_manifest(manifest)
        if errors:
            return errors
        self._install(manifest)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO link_devices (device_id, manifest, registered_at)"
                " VALUES (?, ?, ?)",
                (manifest["device_id"], json.dumps(manifest), time.time()),
            )
        logger.info("link device %r registered (%d capabilities)",
                    manifest["device_id"], len(manifest["capabilities"]))
        return []

    def _install(self, manifest: dict) -> None:
        device_id = manifest["device_id"]
        # Re-registration replaces: drop this device's previous skills first.
        # Match on the OWNING device, not a name prefix — else device "robo_dog"
        # would clobber device "robo"'s "robo_dog_sit" skill (prefix collision).
        for skill in list(self._skills.all()):
            if isinstance(skill, DeviceCapabilitySkill) \
                    and skill._device_id == device_id:
                self._skills.unregister(skill.name)
        self._manifests[device_id] = manifest
        self._state.setdefault(device_id, _DeviceState())
        for cap in manifest["capabilities"]:
            self._skills.register(
                DeviceCapabilitySkill(self, device_id, manifest["name"], cap))

    # -- liveness + transport -------------------------------------------------

    def known(self, device_id: str) -> bool:
        return device_id in self._manifests

    def touch(self, device_id: str) -> None:
        """A poll/result arrived — the device is alive."""
        state = self._state.get(device_id)
        if state is not None:
            state.last_seen = time.monotonic()

    def attach_ws(self, device_id: str, ws) -> None:
        self._state[device_id].ws = ws
        self.touch(device_id)

    def detach_ws(self, device_id: str) -> None:
        state = self._state.get(device_id)
        if state is not None:
            state.ws = None

    def drain_commands(self, device_id: str) -> list[dict]:
        """Queued commands for a polling device (marks it alive)."""
        self.touch(device_id)
        state = self._state[device_id]
        commands, state.queue = state.queue, []
        return commands

    def resolve(self, device_id: str, command_id: str, ok: bool,
                message: str = "") -> bool:
        """A device posted a command result; wake the waiting dispatch."""
        self.touch(device_id)
        state = self._state.get(device_id)
        future = state.futures.pop(command_id, None) if state else None
        if future is None or future.done():
            return False
        future.set_result({"ok": bool(ok), "message": str(message or "")})
        return True

    def devices(self) -> list[dict]:
        """HUD list: name, online, capability count."""
        out = []
        for device_id, manifest in self._manifests.items():
            state = self._state[device_id]
            out.append({
                "device_id": device_id,
                "name": manifest["name"],
                "description": manifest.get("description", ""),
                "transport": manifest.get("transport", "http_poll"),
                "online": state.online(),
                "capabilities": len(manifest["capabilities"]),
            })
        return out

    # -- dispatch -------------------------------------------------------------

    async def dispatch(self, device_id: str, capability: str,
                       params: dict) -> SkillResult:
        manifest = self._manifests.get(device_id)
        if manifest is None:
            return SkillResult("I don't know that device.", success=False)
        name = manifest["name"]
        state = self._state[device_id]
        if not state.online():
            return SkillResult(f"{name} is offline.", success=False)

        command = {"id": secrets.token_hex(8), "capability": capability,
                   "params": params or {}}
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        state.futures[command["id"]] = future
        if state.ws is not None:
            try:
                await state.ws.send_json({"type": "command", **command})
            except Exception:
                state.futures.pop(command["id"], None)
                self.detach_ws(device_id)
                return SkillResult(f"{name} dropped the connection.", success=False)
        else:
            state.queue.append(command)
        try:
            result = await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError:
            state.futures.pop(command["id"], None)
            return SkillResult(f"{name} didn't respond.", success=False)
        if not result["ok"]:
            return SkillResult(
                result["message"] or f"{name} reported a failure.", success=False)
        return SkillResult(result["message"] or "Done.",
                           data={"device": device_id, "capability": capability})
