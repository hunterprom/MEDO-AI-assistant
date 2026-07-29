"""Lion mode's defensive-security skills — read-only, local, advisory.

Every skill here obeys the same five rules, and they are the reason this file
is safe to ship:

1. **This machine only.** Nothing scans, probes, or reaches another host.
2. **Read-only by default.** These inspect and explain; they never kill a
   process, open a port, or change a permission. Any such change is a normal
   actuation skill and goes through the normal confirmation gate — lion mode
   does not add a fast path around it.
3. **User-initiated.** They are surfaced only while lion mode is on, and they
   answer a spoken request.
4. **Advisory framing.** The output is "here is what I see, you decide."
5. **Defensive-only.** Asked to scan someone else, write an exploit, crack a
   credential, or disable MEDO's own guards, they refuse in character.

The hard limits in :data:`LION_SYSTEM` are handed to the model on every
advice-generating call, so the framing survives even when an LLM writes the
prose.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Callable

from core import mk
from core.config import Settings
from core.safety import PathWhitelist
from skills.base import Skill, SkillRequest, SkillResult

# --- the guardrails, as text the model also sees -----------------------------

LION_SYSTEM = (
    "You are MEDO's defensive-security advisor. You operate ONLY on the user's "
    "own local machine, and only in a read-only, advisory capacity: you "
    "explain what is there and suggest hardening steps in plain language. "
    "Hard limits you never cross: you do not scan, probe, or access any other "
    "machine; you do not write malware, exploits, credential-cracking, or "
    "techniques for bypassing security controls; you do not help disable "
    "MEDO's own safety layers. If asked for any of those, refuse briefly and "
    "say lion mode is defensive-only. Answer in plain spoken prose, be "
    "concrete, and remind the user that acting on anything is their decision."
)

REFUSAL = (
    "Lion mode is defensive only. I audit and advise on this machine — I don't "
    "scan, probe, or touch other systems, and I won't write exploits, crack "
    "credentials, or disable MEDO's own safeguards. Tell me what you want to "
    "harden and I'll take a look."
)
REFUSAL_MK = (
    "Лав режимот е само одбранбен. Проверувам и советувам за овој компјутер — "
    "не скенирам туѓи системи и не пишувам напади, не кршам лозинки и не ги "
    "гаснам безбедносните заштити на MEDO. Кажи ми што сакаш да зацврстиме."
)

OFF_REPLY = ("Lion mode is off — say 'lion mode on' and I'll bring out the "
             "security tools.")
OFF_REPLY_MK = ("Лав мод е исклучен — кажи „лав мод вклучи“ и ќе ги извадам "
                "безбедносните алатки.")

# offensive intent: another host, an attack tool, or "turn my guards off"
# NB: the pattern is bounded by \b at both ends, so a prefix alternative must
# consume the WHOLE word ("keylog\w*", not "keylogg") or the trailing \b fails
# mid-word ("keylogg" + "er" has no boundary between them).
_OFFENSIVE = re.compile(
    r"\b(exploit\w*|malware|keylog\w*|ransomware|rootkit\w*|backdoor\w*|payload\w*|"
    r"metasploit|reverse\s+shell\w*|brute[\s-]?force|crack\w*|"
    r"password\s+cracker\w*|hashcat|john\s+the\s+ripper|ddos|botnet\w*|"
    r"privilege\s+escalat\w*|bypass\s+(?:the\s+)?(?:whitelist|safety|"
    r"confirmation|gate)|disable\s+(?:the\s+)?(?:whitelist|safety|"
    r"confirmation|firewall\s+then\s+attack))\b",
    re.IGNORECASE,
)
# a remote target named alongside a scan/probe verb
_SCAN_VERB = re.compile(r"\b(scan|probe|enumerate|nmap|attack|penetrat\w*|pentest)\b",
                        re.IGNORECASE)
_HOSTISH = re.compile(
    r"\b(\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2})?"          # IPv4 / CIDR
    r"|(?:[a-z0-9-]+\.)+[a-z]{2,})\b", re.IGNORECASE)   # a dotted hostname
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "this", "my", "local"}
# When the scan is framed as LOCAL, a domain mentioned in passing ("scan my
# computer for connections to facebook.com") is a subject, not a target — that
# is a legitimate local audit, not an out-of-bounds remote scan.
_LOCAL_SCOPE = re.compile(
    r"\b(?:my|this)\s+(?:pc|computer|machine|laptop|system|ports?|network|box)\b"
    r"|\blocalhost\b|\bthis\s+device\b|\bmy\s+own\b", re.IGNORECASE)
# Suffixes _HOSTISH mistakes for a TLD ("report.pdf" -> host "report.pdf").
_FILE_SUFFIX = re.compile(
    r"\.(?:pdf|txt|md|docx?|xlsx?|pptx?|png|jpe?g|gif|csv|zip|log|json|ya?ml|exe|dll)$",
    re.IGNORECASE)


def _names_remote_target(text: str) -> bool:
    """A scan verb pointed at a host that isn't clearly THIS machine."""
    if not _SCAN_VERB.search(text):
        return False
    if _LOCAL_SCOPE.search(text):          # "scan MY computer …" — local audit
        return False
    for m in _HOSTISH.finditer(text):
        token = m.group(1).lower()
        host = token.split("/")[0]
        if host in _LOCAL_HOSTS or _FILE_SUFFIX.search(host):
            continue                       # a local marker, or a filename
        try:                                   # loopback IPs are local, allow
            if ipaddress.ip_address(host).is_loopback:
                continue
        except ValueError:
            pass
        return True
    return False


#: Defensive verbs — the request LOOKS FOR / removes a threat rather than making
#: one. ("scan my pc for malware", "check this machine for keyloggers".)
_DEFENSIVE_VERB = re.compile(
    r"\b(?:scan|check|audit|detect|find|search|look|review|inspect|remove|clean|"
    r"protect|harden|secure|defend|провери|скенира\w*|заштит\w*)\b", re.IGNORECASE)
#: Offensive actions — PRODUCING, installing, or USING the offensive thing. These
#: stay refused even when the target is this machine (local scope can't launder
#: "write a keylogger" into an allowed request).
_OFFENSIVE_ACTION = re.compile(
    r"\b(?:write|create|build|make|generate|develop|code|install|deploy|plant|"
    r"run|use|launch|execute|craft|напиши|создад\w*|направи|инсталира\w*)\b",
    re.IGNORECASE)
#: A break between the producing verb and the offensive keyword means the verb
#: governs something ELSE, not the keyword: a clause boundary ("scan for malware,
#: then write a report"), or an object/oblique marker showing the keyword is not
#: the verb's DIRECT object ("generate a report ON a keylogger", "a summary OF
#: malware", "make SURE no rootkit"). Plain modifiers (a/simple/two/new) are NOT
#: breaks — "write a simple keylogger" still reads as producing it. Datives like
#: "for me" are deliberately NOT breaks, so "install for me a keylogger" stays
#: refused.
_SPAN_BREAK = re.compile(
    r"[,;.]|\b(?:and|then|or|but|so|after|before|while|because|"
    r"on|of|about|against|behind|sure|that|which)\b", re.IGNORECASE)
#: A defensive noun the offensive word can MODIFY ("malware scan", "keylogger
#: detector", "rootkit removal") — then it's the audit's subject, not the object
#: being made.
_DEFENSIVE_HEAD = re.compile(
    r"\s+(?:scan\w*|check\w*|audit\w*|detect\w*|remov\w*|cleanup|sweep|"
    r"protection|defen\w*)\b", re.IGNORECASE)


def _offensive_action_produces(text: str) -> bool:
    """True only when an offensive-action verb actually GOVERNS the offensive
    keyword — "write a keylogger", "install a rootkit" — as opposed to a generic
    verb that merely co-occurs ("scan for malware and make sure it's clean") or
    the keyword used as a modifier of a defensive noun ("run a malware scan")."""
    for m in _OFFENSIVE.finditer(text):
        if _DEFENSIVE_HEAD.match(text[m.end():]):
            continue                       # "malware scan" — the audit's subject
        for a in _OFFENSIVE_ACTION.finditer(text[:m.start()]):
            span = text[a.end():m.start()]
            # The verb governs the keyword as its DIRECT object when nothing
            # breaks the span and they're close — "write a simple keylogger" is
            # producing it; "generate a report on a keylogger" / "make sure no
            # rootkit" are not.
            if not _SPAN_BREAK.search(span) and len(span) <= 40:
                return True                # verb produces the keyword
    return False


def is_offensive(text: str) -> bool:
    """True when the request is out of bounds for a defensive-only profile.

    An offensive KEYWORD (malware, keylogger, rootkit…) is allowed only when it
    is clearly the SUBJECT of a defensive, local audit — "scan MY computer FOR
    malware" — and NOT paired with a verb that would produce, install, or run
    it. Scanning a remote host is always out of bounds.
    """
    if _names_remote_target(text):
        return True
    if _OFFENSIVE.search(text):
        defensive_local = (_LOCAL_SCOPE.search(text) and _DEFENSIVE_VERB.search(text)
                           and not _offensive_action_produces(text))
        return not defensive_local
    return False


# --- common ports, so a port audit actually explains itself ------------------

COMMON_PORTS: dict[int, str] = {
    22: "SSH (remote shell)", 23: "Telnet (INSECURE — avoid)",
    25: "SMTP mail", 53: "DNS", 80: "HTTP web", 110: "POP3 mail",
    135: "Windows RPC", 139: "NetBIOS", 143: "IMAP mail", 443: "HTTPS web",
    445: "SMB file sharing", 465: "SMTPS mail", 587: "SMTP mail",
    631: "IPP printing", 993: "IMAPS mail", 995: "POP3S mail",
    1433: "MS SQL Server", 1900: "SSDP/UPnP discovery", 3306: "MySQL",
    3389: "RDP remote desktop", 5353: "mDNS/Bonjour", 5432: "PostgreSQL",
    5900: "VNC remote desktop", 6379: "Redis", 8080: "HTTP (alt)",
    8443: "HTTPS (alt)", 11434: "Ollama (local LLM)", 27017: "MongoDB",
    # MEDO's own services
    8710: "MEDO companion API", 8730: "MEDO HUD", 8731: "MEDO vision sidecar",
}
# Ports that are unremarkable to find open even when bound to all interfaces.
_EXPECTED_PUBLIC = {53, 5353, 1900, 137, 138, 139, 445, 631,
                    8710, 8730, 8731, 11434}


class _LionSkill(Skill):
    """Base: surfaced only while lion mode is on, and refuses offensive asks."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def match(self, text: str):
        # Surfaced only in lion mode. Out of lion mode these phrases fall
        # through to the normal router (LLM), so nothing is hidden — the tools
        # are just not on the fast path unless the profile is active.
        if not self._settings.mode.lion:
            return None
        return super().match(text)

    def _preamble(self, request: SkillRequest) -> SkillResult | None:
        """Shared gate: off -> say so; offensive -> refuse. None -> proceed."""
        speak_mk = mk.is_cyrillic(request.text)
        if not self._settings.mode.lion:
            return SkillResult(OFF_REPLY_MK if speak_mk else OFF_REPLY,
                               success=False, data={"lion": False})
        if is_offensive(request.text):
            return SkillResult(REFUSAL_MK if speak_mk else REFUSAL,
                               success=False, data={"refused": "offensive"})
        return None


# --- 1. listening-port audit --------------------------------------------------

def _default_list_ports() -> list[dict]:
    """This machine's listening sockets: [{port, proto, addr, pid, process}]."""
    import psutil

    out = []
    for c in psutil.net_connections(kind="inet"):
        if c.status != psutil.CONN_LISTEN or not c.laddr:
            continue
        name = ""
        if c.pid:
            try:
                name = psutil.Process(c.pid).name()
            except Exception:                  # noqa: BLE001 - process may be gone
                name = ""
        out.append({
            "port": c.laddr.port,
            "proto": "tcp" if c.type == 1 else "udp",
            "addr": c.laddr.ip,
            "pid": c.pid, "process": name,
        })
    return out


def summarize_ports(ports: list[dict]) -> dict:
    """Group ports into known/unexpected, for both speech and the HUD."""
    known, unexpected = [], []
    for p in sorted(ports, key=lambda x: x["port"]):
        label = COMMON_PORTS.get(p["port"])
        public = p["addr"] not in ("127.0.0.1", "::1", "localhost")
        row = {**p, "label": label or "unknown service",
               "public": public}
        # "Investigate" = reachable from the network AND neither a known-benign
        # public service nor obviously local. Advisory only.
        if public and p["port"] not in _EXPECTED_PUBLIC and label is None:
            unexpected.append(row)
        else:
            known.append(row)
    return {"known": known, "unexpected": unexpected, "total": len(ports)}


class SecurityCheckSkill(_LionSkill):
    name = "security_check"
    controls_pc = False
    # Off-lion a semantic hit gives the honest "say lion mode on" reply (its
    # _preamble), never a fabricated port list — which is the whole point.
    routing_phrases = [
        "is anything suspicious listening on my computer",
        "what programs are accepting network connections",
        "are there any open ports I should worry about",
        "how secure is my computer",
        "check whether my machine is exposed to the network",
        "audit my network exposure",
    ]
    description = (
        "Lion mode: list the listening network ports on THIS machine, name the "
        "owning process, explain common ones, and flag anything unexpected for "
        "the user to investigate. Read-only; reports only."
    )
    patterns = [
        re.compile(r"\bsecurity\s+check\b", re.IGNORECASE),
        re.compile(r"\b(?:check|audit|show|list)\s+(?:my\s+|the\s+|open\s+)?"
                   r"(?:listening\s+)?ports\b", re.IGNORECASE),
        re.compile(r"\bwhat(?:'?s| is)\s+listening\b", re.IGNORECASE),
        re.compile(r"\b(?:port\s+scan|scan\s+(?:my\s+|this\s+)?"
                   r"(?:machine|computer|ports))\b", re.IGNORECASE),
        re.compile(r"\bбезбедносна\s+проверка\b", re.IGNORECASE),
        re.compile(r"\b(?:провери|прикажи)\s+(?:ги\s+)?портови", re.IGNORECASE),
    ]

    def __init__(self, settings: Settings, list_ports: Callable | None = None) -> None:
        super().__init__(settings)
        self._list_ports = list_ports or _default_list_ports

    async def execute(self, request: SkillRequest) -> SkillResult:
        blocked = self._preamble(request)
        if blocked is not None:
            return blocked
        speak_mk = mk.is_cyrillic(request.text)
        import asyncio
        try:
            ports = await asyncio.to_thread(self._list_ports)
        except Exception:
            return SkillResult(
                "Не можев да ги прочитам портовите." if speak_mk else
                "I couldn't read the listening ports on this machine.",
                success=False)
        s = summarize_ports(ports)
        if speak_mk:
            head = f"Слушаат {s['total']} порти на овој компјутер."
            if s["unexpected"]:
                names = ", ".join(str(u["port"]) for u in s["unexpected"][:5])
                tail = (f" Провери ги овие — не се вообичаени и се достапни од "
                        f"мрежата: {names}. Јас само пријавувам; ти одлучуваш.")
            else:
                tail = " Ништо необично достапно однадвор."
        else:
            head = f"{s['total']} ports are listening on this machine."
            if s["unexpected"]:
                names = ", ".join(
                    f"{u['port']} ({u['process'] or 'unknown'})"
                    for u in s["unexpected"][:5])
                tail = (f" Worth a look — these are network-reachable and not a "
                        f"service I recognise: {names}. I'm only reporting; you "
                        f"decide whether any of it should be there.")
            else:
                tail = " Nothing unusual is reachable from the network."
        return SkillResult(head + tail, data={"ports": s})

    def tool_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": {}, "required": []}}}


# --- 2. explain a process -----------------------------------------------------

def _default_describe_pid(query: str) -> dict | None:
    """Look up a process by pid or name on THIS machine (read-only)."""
    import psutil

    query = query.strip()
    procs = []
    if query.isdigit():
        try:
            procs = [psutil.Process(int(query))]
        except Exception:
            return None
    else:
        q = query.lower()
        for p in psutil.process_iter(["name"]):
            if q in (p.info["name"] or "").lower():
                procs.append(p)
                if len(procs) >= 3:
                    break
    if not procs:
        return None
    p = procs[0]
    try:
        with p.oneshot():
            return {"pid": p.pid, "name": p.name(),
                    "exe": (p.exe() if _safe(p.exe) else ""),
                    "username": (p.username() if _safe(p.username) else "")}
    except Exception:
        return {"pid": p.pid, "name": query}


def _safe(fn) -> bool:
    try:
        fn()
        return True
    except Exception:
        return False


class ExplainProcessSkill(_LionSkill):
    name = "explain_process"
    controls_pc = False
    description = (
        "Lion mode: given a PID or process name on THIS machine, describe what "
        "it is and whether it's commonly legitimate. Read-only — it never kills "
        "or changes a process."
    )
    patterns = [
        re.compile(r"\bexplain\s+(?:this\s+|the\s+)?process\s+(?P<q>[\w.\- ]+)$", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+(?:the\s+)?process\s+(?P<q2>[\w.\- ]+)$", re.IGNORECASE),
        re.compile(r"\bобјасни\s+(?:го\s+)?процес(?:от)?\s+(?P<qm>[\w.\- ]+)$", re.IGNORECASE),
    ]

    def __init__(self, settings: Settings, describe_pid: Callable | None = None,
                 explain=None) -> None:
        super().__init__(settings)
        self._describe = describe_pid or _default_describe_pid
        self._explain = explain          # async(system, user) -> str; optional

    async def execute(self, request: SkillRequest) -> SkillResult:
        blocked = self._preamble(request)
        if blocked is not None:
            return blocked
        speak_mk = mk.is_cyrillic(request.text)
        gd = request.match.groupdict() if request.match else {}
        query = (request.args.get("process") or gd.get("q") or gd.get("q2")
                 or gd.get("qm") or "").strip()
        # A kill request is an ACTION; this read-only skill won't do it, and
        # says so plainly rather than pretending.
        if re.search(r"\b(kill|terminate|end|stop)\b", request.text, re.IGNORECASE):
            return SkillResult(
                "Јас само објаснувам процеси. Ако сакаш да згасне, тоа е "
                "дејство и ќе прашам пред тоа." if speak_mk else
                "I only explain processes — I won't kill one. Stopping a "
                "process is an action, and that still goes through the normal "
                "confirmation, lion mode or not.", success=False,
                data={"refused": "action"})
        import asyncio
        info = await asyncio.to_thread(self._describe, query)
        if info is None:
            return SkillResult(
                f"Не најдов процес „{query}“." if speak_mk else
                f"I couldn't find a process matching {query!r} on this machine.",
                success=False)
        if self._explain is not None:
            user = (f"Describe this local process for the user and say whether "
                    f"it is commonly legitimate: {info}")
            try:
                text = (await self._explain(LION_SYSTEM, user) or "").strip()
                if text:
                    return SkillResult(text, data={"process": info})
            except Exception:
                pass
        # No model: still give the facts, framed advisory.
        base = (f"Процес {info['name']} (PID {info['pid']})."
                if speak_mk else
                f"That's {info['name']} (PID {info['pid']})"
                + (f", run by {info['username']}" if info.get("username") else "")
                + (f", from {info['exe']}" if info.get("exe") else "")
                + ". If you don't recognise it, look up the executable name "
                  "before acting — I'm only describing it.")
        return SkillResult(base, data={"process": info})

    def tool_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": {
                "process": {"type": "string",
                            "description": "PID or process name on this machine."}},
                "required": ["process"]}}}


# --- 3. firewall audit --------------------------------------------------------

def _default_read_firewall() -> str:
    """Read (never change) the local firewall state. Windows: netsh."""
    import subprocess
    import sys

    if sys.platform == "win32":
        cp = subprocess.run(
            ["netsh", "advfirewall", "show", "allprofiles", "state"],
            capture_output=True, text=True, timeout=15)
        return cp.stdout or cp.stderr
    # Best-effort elsewhere; read-only status commands.
    for cmd in (["ufw", "status"], ["firewall-cmd", "--state"]):
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if cp.returncode == 0:
                return cp.stdout
        except FileNotFoundError:
            continue
    return ""


class FirewallAuditSkill(_LionSkill):
    name = "firewall_audit"
    controls_pc = False
    routing_phrases = [
        "is my firewall on", "is my firewall actually protecting me",
        "are all my firewall profiles enabled", "how's my firewall looking",
        "is my computer's firewall active", "review my firewall settings",
    ]
    description = (
        "Lion mode: read (never modify) the local firewall profile and "
        "summarize it, with hardening suggestions as advice. Read-only."
    )
    patterns = [
        re.compile(r"\bfirewall\s+(?:audit|check|status|review)\b", re.IGNORECASE),
        re.compile(r"\b(?:audit|check|review)\s+(?:my\s+|the\s+)?firewall\b", re.IGNORECASE),
        re.compile(r"\b(?:провери|провери\s+го)\s+(?:заштитниот\s+)?ѕид", re.IGNORECASE),
    ]

    def __init__(self, settings: Settings, read_firewall: Callable | None = None,
                 explain=None) -> None:
        super().__init__(settings)
        self._read = read_firewall or _default_read_firewall
        self._explain = explain

    async def execute(self, request: SkillRequest) -> SkillResult:
        blocked = self._preamble(request)
        if blocked is not None:
            return blocked
        speak_mk = mk.is_cyrillic(request.text)
        import asyncio
        try:
            raw = await asyncio.to_thread(self._read)
        except Exception:
            raw = ""
        if not raw.strip():
            return SkillResult(
                "Не можев да ја прочитам состојбата на заштитниот ѕид." if speak_mk
                else "I couldn't read the firewall state on this machine.",
                success=False)
        states = re.findall(r"([A-Za-z ]+Profile)\s+Settings.*?State\s+(ON|OFF)",
                            raw, re.IGNORECASE | re.DOTALL)
        off = [name.strip() for name, st in states if st.upper() == "OFF"]
        if self._explain is not None:
            try:
                text = (await self._explain(
                    LION_SYSTEM,
                    f"Summarize this local firewall state and give hardening "
                    f"advice:\n{raw[:1500]}") or "").strip()
                if text:
                    return SkillResult(text, data={"off_profiles": off})
            except Exception:
                pass
        if off:
            msg = ("Заштитниот ѕид е ИСКЛУЧЕН за: " + ", ".join(off) +
                   ". Препорачувам да го вклучиш." if speak_mk else
                   f"Your firewall is OFF for: {', '.join(off)}. I'd turn it "
                   "back on for those profiles — that's the single biggest "
                   "hardening step here. Your call.")
        else:
            msg = ("Заштитниот ѕид е вклучен за сите профили." if speak_mk else
                   "Your firewall is on for every profile — good. Keep inbound "
                   "rules to the minimum you actually use.")
        return SkillResult(msg, data={"off_profiles": off})


# --- 4. update / hygiene check ------------------------------------------------

def _default_list_updates() -> list[dict]:
    """Apps with an update available (Windows: winget). Read-only."""
    import subprocess
    import sys

    if sys.platform != "win32":
        return []
    cp = subprocess.run(["winget", "upgrade"], capture_output=True, text=True,
                        timeout=60)
    out = []
    for line in (cp.stdout or "").splitlines():
        # winget columns: Name  Id  Version  Available  Source
        m = re.match(r"^(.{1,40}?)\s{2,}\S+\s{2,}(\S+)\s{2,}(\S+)\s{2,}\S+$", line)
        if m and m.group(2) != m.group(3) and "Version" not in line:
            out.append({"name": m.group(1).strip(),
                        "current": m.group(2), "available": m.group(3)})
    return out


HYGIENE = [
    "use a password manager and a unique password per site",
    "turn on two-factor authentication where it's offered",
    "keep the OS and browser set to auto-update",
    "review which apps launch at startup",
    "lock the screen when you step away",
]


class UpdateCheckSkill(_LionSkill):
    name = "security_updates"
    controls_pc = False
    routing_phrases = [
        "do any of my apps need updating", "is my software out of date",
        "what should I update to stay secure", "am I keeping my system patched",
        "which programs have updates waiting",
        "give me a security hygiene checklist",
    ]
    description = (
        "Lion mode: read-only report of installed apps with updates available, "
        "plus a short password-hygiene checklist. Advisory."
    )
    patterns = [
        re.compile(r"\b(?:check\s+for\s+updates|update\s+check|what\s+needs\s+updating)\b", re.IGNORECASE),
        re.compile(r"\bpassword\s+hygiene\b", re.IGNORECASE),
        re.compile(r"\b(?:security\s+)?hygiene\s+check\b", re.IGNORECASE),
        re.compile(r"\b(?:провери\s+)?надградби\b", re.IGNORECASE),
    ]

    def __init__(self, settings: Settings, list_updates: Callable | None = None) -> None:
        super().__init__(settings)
        self._list_updates = list_updates or _default_list_updates

    async def execute(self, request: SkillRequest) -> SkillResult:
        blocked = self._preamble(request)
        if blocked is not None:
            return blocked
        speak_mk = mk.is_cyrillic(request.text)
        import asyncio
        try:
            updates = await asyncio.to_thread(self._list_updates)
        except Exception:
            updates = []
        n = len(updates)
        tips = "; ".join(HYGIENE[:3])
        if speak_mk:
            head = (f"{n} апликации имаат достапна надградба." if n else
                    "Сите апликации изгледаат ажурни (каде што системот покажува).")
            return SkillResult(head + " За хигиена: " + tips + ".",
                               data={"updates": updates})
        head = (f"{n} app{'s' if n != 1 else ''} "
                f"{'have' if n != 1 else 'has'} an update available"
                + (": " + ", ".join(u["name"] for u in updates[:5]) if n else
                   " (where the OS reports it)")
                + ".") if True else ""
        return SkillResult(
            head + f" On hygiene: {tips}. These are suggestions — updating is "
            "your call.", data={"updates": updates})


# --- 5. explain a file's permissions -----------------------------------------

def _default_stat_path(path: str) -> dict | None:
    import os
    import stat as statmod
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {"path": path, "mode": statmod.filemode(st.st_mode),
            "octal": oct(st.st_mode & 0o777), "size": st.st_size}


class ExplainPermissionsSkill(_LionSkill):
    name = "explain_permissions"
    controls_pc = False
    description = (
        "Lion mode: describe a file's permissions on THIS machine and why they "
        "might matter. Advisory; read-only, and the path must be inside the "
        "allowed folders."
    )
    patterns = [
        re.compile(r"\bexplain\s+(?:the\s+)?permissions?\s+(?:of|for|on)\s+(?P<p>.+)$", re.IGNORECASE),
        re.compile(r"\bwho\s+can\s+(?:read|access)\s+(?P<p2>.+)$", re.IGNORECASE),
        re.compile(r"\bобјасни\s+(?:ги\s+)?дозволите\s+(?:на|за)\s+(?P<pm>.+)$", re.IGNORECASE),
    ]

    def __init__(self, settings: Settings, whitelist: PathWhitelist,
                 stat_path: Callable | None = None) -> None:
        super().__init__(settings)
        self._whitelist = whitelist
        self._stat = stat_path or _default_stat_path

    async def execute(self, request: SkillRequest) -> SkillResult:
        blocked = self._preamble(request)
        if blocked is not None:
            return blocked
        speak_mk = mk.is_cyrillic(request.text)
        gd = request.match.groupdict() if request.match else {}
        raw = (request.args.get("path") or gd.get("p") or gd.get("p2")
               or gd.get("pm") or "").strip().strip("\"'` ")
        from core.config import expand_path
        try:
            path = expand_path(raw)
        except (OSError, RuntimeError, ValueError):
            path = None
        if path is None or not path.exists():
            return SkillResult(
                f"Не најдов „{raw}“." if speak_mk else
                f"I couldn't find {raw!r}.", success=False)
        # Same boundary as every other file-reading skill.
        if not self._whitelist.is_allowed(path):
            return SkillResult(
                "Таа патека е надвор од дозволените папки." if speak_mk else
                "That path is outside the folders I'm allowed to read.",
                success=False, data={"reason": "outside"})
        import asyncio
        info = await asyncio.to_thread(self._stat, str(path))
        if info is None:
            return SkillResult(
                "Не можев да ги прочитам дозволите." if speak_mk else
                "I couldn't read that file's permissions.", success=False)
        if speak_mk:
            msg = (f"{path.name}: дозволи {info['mode']} ({info['octal']}). "
                   "Ако друг може да пишува тука, тоа е ризик — ти одлучуваш.")
        else:
            world = info["octal"][-1]
            note = (" Note: the last digit means other users on this machine "
                    "have access too — worth tightening if it's sensitive."
                    if world not in ("0",) else
                    " Only your account has broad access here, which is fine.")
            msg = (f"{path.name} has permissions {info['mode']} "
                   f"({info['octal']}).{note} Advisory only — changing it is "
                   "your call.")
        return SkillResult(msg, data={"permissions": info})

    def tool_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": ["path"]}}}


#: Registered as a group so main.py can surface them together.
def build_security_skills(settings: Settings, whitelist: PathWhitelist,
                          explain=None) -> list[Skill]:
    return [
        SecurityCheckSkill(settings),
        ExplainProcessSkill(settings, explain=explain),
        FirewallAuditSkill(settings, explain=explain),
        UpdateCheckSkill(settings),
        ExplainPermissionsSkill(settings, whitelist),
    ]
