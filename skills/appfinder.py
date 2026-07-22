"""Find installed applications — and offer to install the ones that are missing.

The gap: ``skills/apps.py`` can only launch what someone typed into
``config.yaml``. Ask MEDO to open Obsidian and it answers "I don't have
obsidian configured" while Obsidian sits in the Start Menu. The machine
already knows what is installed; nothing was asking it.

Three skills over one discovery pass:

* :class:`LocateAppSkill` — "do I have Obsidian?", "where is Blender installed?"
* :class:`OpenDiscoveredAppSkill` — "open Obsidian" for anything found, so the
  config table becomes a place for *overrides* rather than a prerequisite.
  Registered after the configured launcher, so it only ever picks up what fell
  through.
* :class:`InstallAppSkill` — "install Obsidian" via the system package manager.

Discovery reads what the OS already publishes — Start Menu shortcuts on
Windows, ``.app`` bundles on macOS, ``.desktop`` entries on Linux. No registry
spelunking and no filesystem crawl: those are slow, noisy, and full of
uninstaller stubs.

Installing is the one action here that changes the machine, so it is gated
twice: the package is *resolved and named back to you* before anything runs,
and only then does a confirmed install proceed. The package manager is invoked
as an argv list with the package id as a single argument — never a shell
string — so a spoken app name cannot become a command.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import mk
from core.platform import IS_MACOS, IS_WINDOWS, open_path
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

#: Discovery is a directory walk; cache it briefly so three questions in a row
#: don't re-scan 170 shortcuts each time. Short enough that installing
#: something and immediately opening it still works.
_CACHE_TTL_S = 60.0

#: Shortcut names that are never the app you asked for.
_NOISE = re.compile(
    r"\b(uninstall|remove|readme|documentation|help|manual|release notes|"
    r"website|homepage|support|licence|license|changelog|repair|modify)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FoundApp:
    """An installed application the OS told us about."""

    name: str            # "Obsidian"
    path: Path           # the shortcut / bundle / desktop entry to launch
    source: str          # where it was found, for the spoken answer


def _windows_roots() -> list[Path]:
    import os

    return [
        Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
        Path(os.environ.get("ProgramData", "")) / "Microsoft/Windows/Start Menu/Programs",
    ]


def _scan() -> list[FoundApp]:
    """Every application the OS publishes, uncached."""
    found: list[FoundApp] = []
    if IS_WINDOWS:
        for root in _windows_roots():
            if not root.exists():
                continue
            for lnk in root.rglob("*.lnk"):
                if _NOISE.search(lnk.stem):
                    continue
                found.append(FoundApp(lnk.stem, lnk, "Start Menu"))
    elif IS_MACOS:
        for root in (Path("/Applications"), Path.home() / "Applications"):
            if root.exists():
                found += [FoundApp(app.stem, app, "Applications")
                          for app in root.glob("*.app")]
    else:
        for root in (Path("/usr/share/applications"),
                     Path.home() / ".local/share/applications"):
            if root.exists():
                found += [FoundApp(entry.stem, entry, "desktop entries")
                          for entry in root.glob("*.desktop")]
    # Same app in both the user and machine Start Menu: keep one.
    unique: dict[str, FoundApp] = {}
    for app in found:
        unique.setdefault(app.name.lower(), app)
    return sorted(unique.values(), key=lambda a: a.name.lower())


_cache: tuple[float, list[FoundApp]] = (0.0, [])


def installed_apps(force: bool = False) -> list[FoundApp]:
    """Installed applications, cached for :data:`_CACHE_TTL_S`."""
    global _cache
    stamp, apps = _cache
    if force or not apps or (time.monotonic() - stamp) > _CACHE_TTL_S:
        apps = _scan()
        _cache = (time.monotonic(), apps)
    return apps


#: Spoken abbreviations that no substring match can bridge. Deliberately short:
#: this is for names people say differently from how they are written, not a
#: place to re-list every application.
_ALIASES = {
    "vs code": "visual studio code",
    "vscode": "visual studio code",
    "vs": "visual studio",
    "ps": "powershell",
    "cmd": "command prompt",
    "word": "microsoft word",
    "excel": "microsoft excel",
    "teams": "microsoft teams",
    "premiere": "adobe premiere",
    "photoshop": "adobe photoshop",
    "acrobat": "adobe acrobat",
    "хром": "google chrome",
    "ворд": "microsoft word",
}


def score_name(candidate: str, wanted: str) -> float:
    """How well an installed app's name answers a spoken one. Pure, 0 = no match."""
    candidate, wanted = candidate.strip().lower(), wanted.strip().lower()
    if not candidate or not wanted:
        return 0.0
    if candidate == wanted:
        return 1.0
    if candidate.startswith(wanted):
        return 0.9
    if wanted in candidate:
        # Prefer the shortest container: "Obsidian" beats "Obsidian Sandbox".
        return 0.7 + 0.2 * (len(wanted) / len(candidate))
    words = {w for w in re.findall(r"\w+", candidate)}
    return 0.5 if wanted in words else 0.0


def find_app(name: str, apps: list[FoundApp] | None = None) -> FoundApp | None:
    """The installed app that best matches a spoken name, or None."""
    name = name.strip().strip(" .,?!\"'")
    if not name:
        return None
    name = _ALIASES.get(name.lower(), name)
    best, best_score = None, 0.0
    for app in (installed_apps() if apps is None else apps):
        score = score_name(app.name, name)
        if score > best_score:
            best, best_score = app, score
    return best


# --- installing ----------------------------------------------------------------


def package_manager() -> tuple[str, list[str]] | None:
    """(name, argv prefix) of the package manager here, or None if there isn't one."""
    import shutil

    if IS_WINDOWS and shutil.which("winget"):
        return "winget", ["winget"]
    if IS_MACOS and shutil.which("brew"):
        return "Homebrew", ["brew"]
    if shutil.which("flatpak"):
        return "Flatpak", ["flatpak"]
    if shutil.which("apt-get"):
        return "apt", ["sudo", "apt-get"]
    return None


def parse_winget_search(output: str) -> list[tuple[str, str]]:
    """winget's table output -> [(name, id)]. Pure, so it is testable offline.

    winget prints a fixed-width table whose columns shift with terminal width,
    so the id is taken as the token that looks like one (``Publisher.Product``)
    rather than by slicing at a column offset.
    """
    results: list[tuple[str, str]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("-") or line.lower().startswith("name "):
            continue
        match = re.search(r"\s([\w][\w+.-]*\.[\w][\w+.-]*)(?:\s|$)", line)
        if not match:
            continue
        package_id = match.group(1)
        name = line[: match.start()].strip()
        if name and "." in package_id:
            results.append((name, package_id))
    return results


def search_packages(query: str, timeout_s: float = 25.0) -> list[tuple[str, str]]:
    """Ask the package manager what it has. Empty list on any failure."""
    manager = package_manager()
    if manager is None or manager[0] != "winget":
        return []                      # only winget's output is parsed for now
    try:
        proc = subprocess.run(
            [*manager[1], "search", "--source", "winget", "--accept-source-agreements",
             "-q", query],
            capture_output=True, text=True, timeout=timeout_s, check=False,
            encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("package search failed", exc_info=True)
        return []
    return parse_winget_search(proc.stdout or "")


def install_command(package_id: str) -> list[str] | None:
    """argv for installing ``package_id``, or None when there's no manager.

    An argv list, never a shell string: a spoken application name reaches this
    function, and a spoken name must never be able to become a command.
    """
    manager = package_manager()
    if manager is None:
        return None
    name, argv = manager
    if name == "winget":
        return [*argv, "install", "--id", package_id, "--exact",
                "--accept-package-agreements", "--accept-source-agreements"]
    if name == "Homebrew":
        return [*argv, "install", "--cask", package_id]
    if name == "Flatpak":
        return [*argv, "install", "-y", package_id]
    return [*argv, "install", "-y", package_id]


# --- skills --------------------------------------------------------------------


class LocateAppSkill(Skill):
    """"do I have Obsidian?" — answer from what the OS has published."""

    name = "locate_app"
    controls_pc = False          # it only looks; nothing is launched or changed
    description = (
        "Check whether an application is installed on this computer and say "
        "where it was found. Use for 'do I have X', 'is X installed', "
        "'where is X'."
    )

    patterns = [
        re.compile(r"\b(?:do\s+i\s+have|have\s+i\s+got)\s+(?P<app>[\w .+-]+?)"
                   r"(?:\s+installed)?\s*[?.!]*$", re.IGNORECASE),
        re.compile(r"\bis\s+(?P<app2>[\w .+-]+?)\s+installed\b", re.IGNORECASE),
        re.compile(r"\bwhere\s+is\s+(?P<app3>[\w .+-]+?)\s+installed\b", re.IGNORECASE),
        re.compile(r"\b(?:find|locate)\s+(?:the\s+)?(?:app|application|program)\s+"
                   r"(?P<app4>[\w .+-]+)", re.IGNORECASE),
        re.compile(r"\b(?:what|which)\s+(?:apps|applications|programs)\s+"
                   r"(?:do\s+i\s+have|are\s+installed)\b", re.IGNORECASE),
        # MK
        re.compile(r"\bдали\s+(?:го\s+)?имам\s+(?P<appm>[\w .+-]+?)\s*[?.!]*$",
                   re.IGNORECASE),
        re.compile(r"\bкаде\s+е\s+инсталиран(?:а|о)?\s+(?P<appm2>[\w .+-]+)",
                   re.IGNORECASE),
    ]

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        name = (request.args.get("app") or gd.get("app") or gd.get("app2")
                or gd.get("app3") or gd.get("app4") or gd.get("appm")
                or gd.get("appm2") or "").strip()

        apps = await asyncio.to_thread(installed_apps)
        if not name:
            # "what apps do I have" — a count plus a sample; reading 172 names
            # aloud is not an answer.
            sample = ", ".join(a.name for a in apps[:8])
            return SkillResult(
                f"Гледам {len(apps)} апликации, меѓу нив: {sample}." if speak_mk
                else f"I can see {len(apps)} applications, including: {sample}.",
                data={"count": len(apps)})

        app = find_app(name, apps)
        if app is None:
            manager = package_manager()
            hint = (f" Кажи „инсталирај {name}“ ако сакаш да ја инсталирам."
                    if speak_mk else
                    f" Say 'install {name}' and I'll fetch it.") if manager else ""
            return SkillResult(
                (f"Не гледам {name} инсталирано.{hint}" if speak_mk
                 else f"I don't see {name} installed.{hint}"), success=False,
                data={"installed": False, "query": name})
        return SkillResult(
            f"Да, {app.name} е инсталирано ({app.source})." if speak_mk
            else f"Yes — {app.name} is installed, from the {app.source}.",
            data={"installed": True, "name": app.name, "path": str(app.path)})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string",
                                "description": "Application name. Omit to list."},
                    },
                    "required": [],
                },
            },
        }


class OpenDiscoveredAppSkill(Skill):
    """"open Obsidian" for anything installed, not just what's in config.

    Registered AFTER the configured launcher and the website opener, so it only
    sees what they didn't want. That ordering is the whole safety story: this
    pattern is broad by necessity ("open <anything>"), and it earns that by
    being last and by refusing anything it cannot actually find.
    """

    name = "open_installed_app"
    controls_pc = True
    description = (
        "Launch an installed application by name, discovered from the system "
        "rather than the config file."
    )

    patterns = [
        re.compile(r"\b(?:open|launch|start|run)\s+(?:the\s+|my\s+)?"
                   r"(?P<app>[\w .+-]{2,40}?)(?:\s+(?:app|application|program))?\s*[.!]*$",
                   re.IGNORECASE),
        re.compile(rf"\b(?:{mk.OPEN}){mk.CLITICS}\s+(?P<appm>[\w .+-]{{2,40}}?)\s*[.!]*$",
                   re.IGNORECASE),
    ]

    def match(self, text: str):
        """Claim the utterance only when the name resolves to a real app.

        The pattern above has to be broad — an app can be called anything — so
        on its own it swallows any sentence containing "run" or "start":
        "start over please" became the app "over please", and "…to run it to
        find me an interesting video" became an app name in full. Matching is
        therefore gated on discovery: if nothing by that name is installed,
        this skill never claimed the sentence and the router carries on to the
        LLM, which is the right home for those.
        """
        found = super().match(text)
        if found is None:
            return None
        groups = found.groupdict()
        name = (groups.get("app") or groups.get("appm") or "").strip()
        return found if name and find_app(name) is not None else None

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        name = (request.args.get("app") or gd.get("app") or gd.get("appm") or "").strip()
        if not name:
            return SkillResult("Која апликација?" if speak_mk else "Which app?",
                               success=False)
        app = await asyncio.to_thread(find_app, name)
        if app is None:
            manager = package_manager()
            hint = (" Сакаш ли да ја инсталирам?" if speak_mk
                    else " Want me to install it?") if manager else ""
            return SkillResult(
                (f"Не најдов апликација „{name}“.{hint}" if speak_mk
                 else f"I couldn't find an app called {name}.{hint}"),
                success=False, data={"query": name})
        try:
            await asyncio.to_thread(open_path, app.path)
        except Exception as exc:
            logger.warning("could not launch %s: %s", app.path, exc)
            return SkillResult(
                f"Не успеав да го отворам {app.name}." if speak_mk
                else f"I couldn't launch {app.name}.", success=False)
        return SkillResult(f"Отворам {app.name}." if speak_mk
                           else f"Opening {app.name}.",
                           data={"name": app.name, "path": str(app.path)})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {"app": {"type": "string"}},
                    "required": ["app"],
                },
            },
        }


class InstallAppSkill(Skill):
    """"install Obsidian" — resolve the package, name it back, then install."""

    name = "install_app"
    controls_pc = True
    description = (
        "Install an application using the system package manager (winget, "
        "Homebrew, Flatpak, apt). Use only when the user asks to install "
        "something."
    )

    patterns = [
        re.compile(r"\binstall\s+(?:the\s+)?(?:app\s+|application\s+|program\s+)?"
                   r"(?P<app>[\w .+-]{2,40}?)\s*(?:\s+(?:for\s+me|please))?\s*[.!]*$",
                   re.IGNORECASE),
        re.compile(r"\b(?:download\s+and\s+install|get\s+me)\s+(?P<app2>[\w .+-]{2,40})",
                   re.IGNORECASE),
        re.compile(r"\bинсталирај\s+(?:ми\s+)?(?:ја\s+|го\s+)?(?P<appm>[\w .+-]{2,40})",
                   re.IGNORECASE),
    ]

    #: How long to let an install run before giving up on reporting its result.
    INSTALL_TIMEOUT_S = 600.0

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        name = (request.args.get("app") or gd.get("app") or gd.get("app2")
                or gd.get("appm") or "").strip()
        if not name:
            return SkillResult("Што да инсталирам?" if speak_mk
                               else "What should I install?", success=False)

        manager = package_manager()
        if manager is None:
            return SkillResult(
                "Нема менаџер на пакети на овој систем." if speak_mk
                else "There's no package manager on this system, so I can't "
                     "install anything.", success=False)

        # Already here? Installing over it is a slow way to do nothing.
        existing = await asyncio.to_thread(find_app, name)
        if existing is not None and not request.context.get("confirmed"):
            return SkillResult(
                f"{existing.name} е веќе инсталирано." if speak_mk
                else f"{existing.name} is already installed.",
                data={"already": True, "name": existing.name})

        package_id = request.args.get("package_id") or ""
        if not package_id:
            matches = await asyncio.to_thread(search_packages, name)
            if not matches:
                return SkillResult(
                    f"{manager[0]} не најде пакет „{name}“." if speak_mk
                    else f"{manager[0]} couldn't find a package called {name}.",
                    success=False)
            package_name, package_id = matches[0]
        else:
            package_name = name

        if not request.context.get("confirmed"):
            # Name the resolved package before doing anything: "install obsidian"
            # matching some unrelated package is exactly the failure worth
            # catching, and only the user can catch it.
            return SkillResult(
                f"Најдов {package_name} ({package_id}) преку {manager[0]}. "
                f"Да го инсталирам?" if speak_mk else
                f"I found {package_name} ({package_id}) on {manager[0]}. "
                f"Shall I install it?",
                needs_confirmation=True,
                data={"package_id": package_id, "name": package_name})

        argv = install_command(package_id)
        if argv is None:
            return SkillResult("No package manager available.", success=False)
        try:
            proc = await asyncio.to_thread(
                subprocess.run, argv, capture_output=True, text=True,
                timeout=self.INSTALL_TIMEOUT_S, check=False,
            )
        except subprocess.TimeoutExpired:
            return SkillResult(
                f"{package_name} сè уште се инсталира — трае подолго од очекувано."
                if speak_mk else
                f"{package_name} is still installing — it's taking longer than "
                f"I'll wait around for. It should finish on its own.",
                data={"package_id": package_id, "timeout": True})
        except OSError as exc:
            logger.warning("install failed to launch: %s", exc)
            return SkillResult(
                f"Не успеав да го стартувам {manager[0]}." if speak_mk
                else f"I couldn't run {manager[0]}.", success=False)

        if proc.returncode != 0:
            detail = (proc.stdout or proc.stderr or "").strip().splitlines()
            why = detail[-1][:120] if detail else f"exit code {proc.returncode}"
            return SkillResult(
                f"Инсталацијата на {package_name} не успеа: {why}" if speak_mk
                else f"Installing {package_name} failed: {why}",
                success=False, data={"package_id": package_id})
        installed_apps(force=True)     # the new app should be launchable now
        return SkillResult(
            f"{package_name} е инсталирано." if speak_mk
            else f"{package_name} is installed.",
            data={"package_id": package_id, "name": package_name})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string",
                                "description": "Application to install."},
                    },
                    "required": ["app"],
                },
            },
        }
