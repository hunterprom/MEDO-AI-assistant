"""The agent cluster graph — capability spheres and their skill-stars.

The HUD's cluster view draws each capability DOMAIN as a sphere and each thing
it can actually do as a STAR inside it. The whole point is that it stays honest:
the graph is derived from the live :class:`~skills.base.SkillRegistry` at page
load, so a skill you add tomorrow shows up as a star tomorrow, in its domain,
with no picture to hand-edit.

Two kinds of star:

* Ordinary domains (Vision, Web, System…) — one star per registered skill.
* The **Council** — its stars are the specialists themselves (Electrical,
  Robotics, Physics, Law…), because "an agent per major" is exactly what that
  sphere is meant to show. Those come from :data:`core.council.COUNCIL`, so
  adding a specialist in ``council.extra`` adds a star too.

A skill that isn't mapped to a domain still appears — under ``plugins`` if it
came from a drop-in/plugin module, else ``other`` — so a new capability can
never silently vanish from the map. That fallback is the safety net that keeps
"data-driven" a promise rather than a hope.

Pure and side-effect free: it reads the registry, it does not touch it. The
HUD lights a sphere when one of its stars fires by looking the fired skill up
in :func:`skill_domain_index`.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- domains -----------------------------------------------------------------

#: Display order and label for every sphere. A domain with no stars at runtime
#: is dropped, so this can list more than a given install actually has.
DOMAIN_ORDER: tuple[tuple[str, str], ...] = (
    ("vision", "Vision"),
    ("experts", "Council"),
    ("web", "Web"),
    ("knowledge", "Knowledge"),
    ("memory", "Memory"),
    ("system", "System"),
    ("apps", "Apps"),
    ("agenda", "Agenda"),
    ("environment", "Environment"),
    ("security", "Security"),
    ("modes", "Modes"),
    ("plugins", "Plugins"),
    ("other", "Other"),
)

#: skill name -> domain key. The single source of truth for where a star lands
#: and which sphere lights when that skill fires. Unmapped skills fall back by
#: module (see :func:`_fallback_domain`), so this does not have to be complete
#: to stay correct — only to stay tidy.
_DOMAIN: dict[str, str] = {
    # vision
    "see_camera": "vision", "see_screen": "vision", "what_is_this": "vision",
    "see_bench": "vision", "pointer_control": "vision",
    # council: the skills route here; the visible stars are the specialists.
    "ask_specialist": "experts", "convene_council": "experts",
    "circuit_help": "experts", "list_council": "experts",
    # web
    "web_search": "web", "web_fetch": "web", "open_website": "web",
    "site_search": "web", "play_media": "web", "browser_control": "web",
    "browser_task": "web",
    # knowledge / files
    "files": "knowledge", "search_documents": "knowledge", "edit_file": "knowledge",
    "open_in_editor": "knowledge", "write_in_app": "knowledge",
    "import_file": "knowledge", "notes": "knowledge", "resistor_colors": "knowledge",
    # memory
    "remember_fact": "memory", "recall_facts": "memory", "forget_fact": "memory",
    # system / control
    "power": "system", "volume": "system", "brightness": "system",
    "screenshot": "system", "system_info": "system", "type_text": "system",
    "press_keys": "system", "window_action": "system", "clipboard": "system",
    "media": "system", "operate_screen": "system", "repeat": "system",
    # apps
    "apps": "apps", "locate_app": "apps", "install_app": "apps",
    "open_installed_app": "apps", "obsidian_note": "apps",
    "obsidian_open": "apps", "obsidian_search": "apps",
    # agenda
    "datetime": "agenda", "timers": "agenda", "briefing": "agenda",
    # environment
    "weather": "environment", "news": "environment",
    # security (Lion mode's defensive-audit skills)
    "security_check": "security", "explain_process": "security",
    "firewall_audit": "security", "security_updates": "security",
    "explain_permissions": "security",
    # modes
    "session_mode": "modes", "dictation": "modes", "lion_mode": "modes",
}

#: Nicer star captions than the raw skill name. Missing entries are prettified
#: from the name ("open_in_editor" -> "Open In Editor").
_STAR_LABELS: dict[str, str] = {
    "see_camera": "Camera", "see_screen": "Screen Read", "what_is_this": "Cursor Look",
    "see_bench": "Bench", "pointer_control": "Pointer",
    "web_search": "Web Search", "web_fetch": "Read Page", "open_website": "Open Site",
    "site_search": "Site Search", "play_media": "Play Media", "browser_control": "Browser",
    "browser_task": "Browser Agent",
    "files": "Files", "search_documents": "Documents", "edit_file": "Edit File",
    "open_in_editor": "Open In Editor", "write_in_app": "Write In App",
    "import_file": "Import", "notes": "Notes", "resistor_colors": "Resistor",
    "remember_fact": "Remember", "recall_facts": "Recall", "forget_fact": "Forget",
    "power": "Power", "volume": "Volume", "brightness": "Brightness",
    "screenshot": "Screenshot", "system_info": "System Info", "type_text": "Type",
    "press_keys": "Keys", "window_action": "Windows", "clipboard": "Clipboard",
    "media": "Media Keys", "operate_screen": "Screen Agent", "repeat": "Repeat",
    "apps": "Launch", "locate_app": "Locate", "install_app": "Install",
    "open_installed_app": "Open App", "obsidian_note": "Obsidian Note",
    "obsidian_open": "Obsidian Open", "obsidian_search": "Obsidian Search",
    "datetime": "Time", "timers": "Timers", "briefing": "Briefing",
    "weather": "Weather", "news": "News",
    "security_check": "Port Audit", "explain_process": "Process",
    "firewall_audit": "Firewall", "security_updates": "Updates",
    "explain_permissions": "Permissions",
    "session_mode": "Session Mode", "dictation": "Dictation", "lion_mode": "Lion Mode",
}


@dataclass(frozen=True)
class Star:
    """One capability inside a sphere. ``skill`` is None for a council member,
    which is an agent rather than a directly-fired skill."""

    label: str
    skill: str | None = None
    controls_pc: bool = False


@dataclass(frozen=True)
class Sphere:
    key: str
    label: str
    stars: tuple[Star, ...]


def prettify(name: str) -> str:
    """"open_in_editor" -> "Open In Editor" — the fallback star caption."""
    return " ".join(w.capitalize() for w in str(name).replace("_", " ").split())


def _fallback_domain(skill) -> str:
    """Where an unmapped skill goes: a plugin module -> ``plugins``, else
    ``other``. Never drops the skill; the map only exists to keep new
    capabilities visible without a code edit here."""
    module = type(skill).__module__
    if not module.startswith("skills.") and not module.startswith("core."):
        return "plugins"
    return "other"


def _domain_for(skill) -> str:
    return _DOMAIN.get(getattr(skill, "name", ""), None) or _fallback_domain(skill)


def skill_domain_index(registry) -> dict[str, str]:
    """skill name -> domain key, for EVERY registered skill.

    This is the lighting map: when a routed event names a skill, the HUD looks
    it up here to know which sphere to flare. It covers council skills too
    (``ask_specialist`` -> ``experts``) even though their visible stars are the
    specialists, so firing the council still lights the council sphere.
    """
    return {s.name: _domain_for(s) for s in registry.all()}


def build_spheres(registry, council=None) -> list[Sphere]:
    """The ordered, non-empty spheres with their stars.

    ``council`` is the specialist roster (``core.council`` list); when omitted
    the built-in :data:`~core.council.COUNCIL` is used. Pass the *enabled*
    roster so a specialist switched off in config doesn't show as a star.
    """
    # Group skills by domain, preserving registry (registration) order so the
    # map reads the same way the router resolves.
    grouped: dict[str, list[Star]] = {key: [] for key, _ in DOMAIN_ORDER}
    for skill in registry.all():
        domain = _domain_for(skill)
        if domain == "experts":
            # The council's stars are its members, added below — the three
            # council *skills* are plumbing, not stars.
            continue
        grouped.setdefault(domain, []).append(Star(
            label=_STAR_LABELS.get(skill.name) or prettify(skill.name),
            skill=skill.name,
            controls_pc=bool(getattr(skill, "controls_pc", False)),
        ))

    if council is None:
        from core.council import COUNCIL as council
    # The discipline (key) reads cleaner as a star than the spoken title:
    # "Robotics", "Physics", "Law" — matching how the user thinks of the
    # majors — rather than "Roboticist", "Physicist", "Lawyer".
    grouped["experts"] = [
        Star(label=prettify(member.key)) for member in council
    ]

    labels = dict(DOMAIN_ORDER)
    spheres: list[Sphere] = []
    # DOMAIN_ORDER first (stable, curated), then any fallback domain key that
    # somehow isn't in the order table — defensive, should not happen.
    seen = set()
    for key, label in DOMAIN_ORDER:
        stars = grouped.get(key) or []
        if stars:
            spheres.append(Sphere(key, label, tuple(stars)))
        seen.add(key)
    for key, stars in grouped.items():
        if key not in seen and stars:
            spheres.append(Sphere(key, labels.get(key, prettify(key)), tuple(stars)))
    return spheres


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")


def capabilities_feed(registry, council=None) -> dict:
    """The live capability map served at ``GET /capabilities`` (M21 S1).

    Same grouping as :func:`agent_graph`, reshaped into what the orb needs to
    place and light nodes: agents (domains) each with their ``skills[]``, every
    skill carrying ``id``/``label``/``status``/``description``. ``status`` is a
    sane default (``idle``) in this snapshot; live status arrives over SSE
    (S3). ``skillIndex`` maps every registered skill id -> its agent id, so a
    ``routed`` event can find the node to flare.

    Council specialists are ``kind: "agent"`` leaf nodes (they aren't directly
    routable skills); everything else is ``kind: "skill"`` with a real id that
    matches the ``routed`` event's ``skill``.
    """
    spheres = build_spheres(registry, council)
    by_name = {s.name: s for s in registry.all()}
    agents = []
    for sphere in spheres:
        skills = []
        for star in sphere.stars:
            skill_obj = by_name.get(star.skill) if star.skill else None
            skills.append({
                "id": star.skill or f"{sphere.key}:{_slug(star.label)}",
                "label": star.label,
                "kind": "skill" if star.skill else "agent",
                "status": "idle",
                "controlsPc": star.controls_pc,
                "description": (skill_obj.description if skill_obj else ""),
            })
        agents.append({"id": sphere.key, "label": sphere.label, "skills": skills})
    return {"agents": agents, "skillIndex": skill_domain_index(registry)}


def agent_graph(registry, council=None) -> dict:
    """The JSON-able graph the HUD embeds: ``{spheres, skillDomain}``."""
    spheres = build_spheres(registry, council)
    return {
        "spheres": [
            {
                "key": s.key,
                "label": s.label,
                "stars": [
                    {"label": st.label, "skill": st.skill,
                     "controlsPc": st.controls_pc}
                    for st in s.stars
                ],
            }
            for s in spheres
        ],
        "skillDomain": skill_domain_index(registry),
    }
