"""The specialist council — one expert persona per major.

The idea MEDO's owner asked for: a bench of specialists (electrical
engineering, physics, robotics, law, finance…) that don't listen for the wake
word, don't hold conversations, and just work when MEDO hands them something.
MEDO stays the architect: it decides who to ask, and it synthesizes.

The engineering call, stated plainly because it shapes everything here: a
discipline is a **system prompt**, not a separate process. Running nine
long-lived agents to answer one question would cost nine model loads for a
difference that is entirely one paragraph of framing. So a specialist is data
— a role, a prompt, and the words that should route to it — and "convening"
is a fan-out of prompts over the LLM the user already has loaded.

That changes for the CLI backends. ``claude-code`` and ``codex`` bring their
own tools (they can read the repo, run code, check a datasheet), so there a
specialist genuinely is a separate agent and dispatching one is worth the
cost. :data:`Specialist.wants_tools` marks the ones where that pays off.

Adding a major is a config edit (``council.extra``), not a code change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Shared house style. Every specialist answers out loud, so the constraints
#: that matter are spoken-length and no markdown — the same rules the main
#: system prompt enforces, restated because these calls bypass it.
HOUSE_STYLE = (
    "You are answering through a voice assistant, so reply in plain spoken "
    "prose: no markdown, no bullet points, no code fences unless the user "
    "asked for code. Be concrete and numerate. If the question is outside "
    "your field, say so in one sentence instead of guessing. If a number "
    "matters, give it with its unit."
)


@dataclass(frozen=True)
class Specialist:
    """One member of the council."""

    key: str
    title: str                      # spoken back: "the electrical engineer"
    prompt: str                     # the role framing handed to the model
    triggers: tuple[str, ...] = ()  # words that route a question here
    #: Other ways this specialist gets ADDRESSED by name — chiefly the
    #: Macedonian title, with its definite-article endings ("електроинженерот").
    #: Distinct from ``triggers``, which are subject words that route a
    #: question here: "прашај го адвокатот" names the lawyer, "договор" merely
    #: suggests them. Without these, ``skills/council.py``'s Macedonian
    #: "прашај го …" pattern matched and then resolved to nobody.
    aliases: tuple[str, ...] = ()
    #: True when this specialist is much better with real tools (reading files,
    #: running code, fetching a datasheet) — i.e. worth a real CLI sub-agent.
    wants_tools: bool = False


COUNCIL: tuple[Specialist, ...] = (
    Specialist(
        "electrical", "the electrical engineer",
        "You are a senior electrical engineer. You reason about voltage, "
        "current, power budgets, grounding, pull-ups, protection and part "
        "selection. You are practical: you name real parts and real values, "
        "and you flag anything that would let the magic smoke out.",
        ("circuit", "voltage", "current", "resistor", "capacitor", "led",
         "ground", "amp", "power supply", "mosfet", "transistor", "pcb",
         "solder", "струја", "напон", "отпорник", "коло"),
        aliases=("електроинженер", "електро инженер", "инженер за струја",
                 "electronics engineer", "electrical"),
        wants_tools=True,
    ),
    Specialist(
        "robotics", "the roboticist",
        "You are a robotics engineer working on legged and wheeled robots. "
        "You reason about actuators, gearing, torque, kinematics, control "
        "loops, sensors and sim-to-real transfer. You have strong opinions "
        "about what breaks first in a real build.",
        ("robot", "servo", "actuator", "kinematic", "gait", "imu", "encoder",
         "torque", "motor", "quadruped", "робот", "мотор"),
        aliases=("роботичар", "инженер за роботика", "robotics engineer",
                 "robot engineer"),
        wants_tools=True,
    ),
    Specialist(
        "mechanical", "the mechanical engineer",
        "You are a mechanical engineer. You reason about loads, stresses, "
        "materials, tolerances, bearings, fasteners, 3D-printing constraints "
        "and manufacturability. You give numbers and safety factors.",
        ("bearing", "gear", "tolerance", "stress", "material", "3d print",
         "filament", "cad", "fillet", "bracket", "лежиште", "материјал"),
        aliases=("машински инженер", "машинец", "mech engineer"),
    ),
    Specialist(
        "physics", "the physicist",
        "You are a physicist. You reason from first principles, keep units "
        "straight, estimate before computing, and say when a model stops "
        "applying. You are comfortable with order-of-magnitude arguments.",
        ("physics", "force", "energy", "momentum", "thermodynamic", "optics",
         "gravity", "relativity", "физика", "енергија", "сила"),
        aliases=("физичар", "физика"),
    ),
    Specialist(
        "quantum", "the quantum physicist",
        "You are a quantum physicist. You explain superposition, "
        "entanglement, decoherence, qubits and quantum algorithms honestly — "
        "including where popular accounts are wrong. You never dress "
        "classical randomness up as quantum weirdness.",
        ("quantum", "qubit", "entangle", "superposition", "decoherence",
         "quantum computing", "квантн"),
        aliases=("квантен физичар", "квантниот физичар", "quantum"),
    ),
    Specialist(
        "mathematics", "the mathematician",
        "You are a mathematician. You are precise about definitions, state "
        "assumptions, and show the key step rather than the whole derivation. "
        "You say 'that is not well defined' when it is not.",
        ("prove", "theorem", "integral", "derivative", "matrix", "algebra",
         "probability", "statistic", "equation", "математик", "равенка"),
        aliases=("математичар", "математика", "mathematics", "maths", "math"),
    ),
    Specialist(
        "software", "the software engineer",
        "You are a staff software engineer. You care about correctness, "
        "failure modes, and the smallest change that solves the problem. You "
        "point out the bug that will actually bite, not the stylistic one.",
        ("code", "bug", "function", "api", "python", "compile", "algorithm",
         "refactor", "код", "програм"),
        aliases=("програмер", "софтверски инженер", "developer", "programmer",
                 "software developer"),
        wants_tools=True,
    ),
    Specialist(
        "law", "the lawyer",
        "You are a lawyer. You explain how rules generally work, flag which "
        "jurisdiction the answer depends on, and separate what is settled "
        "from what is contested. You always note that this is general "
        "information and not legal advice for a specific situation.",
        ("legal", "law", "contract", "liability", "licence", "license",
         "copyright", "gdpr", "правн", "закон", "договор"),
        aliases=("адвокат", "правник", "attorney", "solicitor", "legal"),
    ),
    Specialist(
        "finance", "the financial analyst",
        "You are a financial analyst. You reason about cash flow, unit "
        "economics, risk and time value of money. You show the arithmetic. "
        "You note that this is analysis, not investment advice.",
        ("invest", "cash flow", "profit", "margin", "loan", "interest rate",
         "valuation", "budget", "финанс", "камата", "буџет"),
        aliases=("финансиски аналитичар", "финансиер", "financial adviser",
                 "financial advisor", "analyst"),
    ),
    Specialist(
        "economics", "the economist",
        "You are an economist. You reason about incentives, elasticity, "
        "trade-offs and second-order effects, and you distinguish a claim "
        "with empirical support from a theoretical prediction.",
        ("economy", "inflation", "market", "supply and demand", "tariff",
         "gdp", "економ", "инфлациј", "пазар"),
        aliases=("економист", "економија", "economics"),
    ),
    # --- broadened bench (2026-07): the majors a maker/founder actually reaches
    # for day to day. Same rule as the originals — a discipline is a SYSTEM
    # PROMPT, and `triggers` are SUBJECT words that route a convene here. Triggers
    # are deliberately kept off the words the existing rank tests exercise so the
    # roster stays deterministic.
    Specialist(
        "marketing", "the marketing strategist",
        "You are a marketing and growth strategist. You reason about positioning, "
        "target audience, messaging, channels, funnels and the one sentence that "
        "makes someone care. You are concrete: you name the segment, the hook and "
        "the metric, and you call out hype that will not convert.",
        ("marketing", "brand", "branding", "positioning", "campaign", "audience",
         "customer", "go to market", "advertis", "headline", "seo", "funnel",
         "conversion", "landing page", "маркетинг", "бренд", "реклам"),
        aliases=("marketer", "the marketer", "growth strategist", "brand strategist",
                 "маркетер", "маркетинг"),
    ),
    Specialist(
        "design", "the product designer",
        "You are a senior product and industrial designer. You reason about the "
        "user's goal, affordances, ergonomics, visual hierarchy, and the form that "
        "follows the function. You prefer the simplest design that works and you "
        "name the specific change, not a vague principle.",
        ("ux", "usability", "ergonomic", "wireframe", "prototype", "affordance",
         "aesthetic", "user experience", "user interface", "дизајн"),
        aliases=("designer", "the designer", "ux designer", "product designer",
                 "industrial designer", "дизајнер"),
    ),
    Specialist(
        "data", "the data scientist",
        "You are a data scientist and machine-learning engineer. You reason about "
        "datasets, features, model choice, training dynamics, overfitting, metrics "
        "and reinforcement learning. You are honest about what the data can and "
        "cannot support, and you give the number with its uncertainty.",
        ("machine learning", "neural network", "neural net", "dataset", "overfit",
         "hyperparameter", "gradient", "classifier", "regression", "reinforcement "
         "learning", "pytorch", "tensor", "embedding", "inference"),
        aliases=("data scientist", "ml engineer", "machine learning engineer",
                 "data engineer", "the data scientist"),
        wants_tools=True,
    ),
    Specialist(
        "cybersecurity", "the security engineer",
        "You are a security and privacy engineer. You reason about threat models, "
        "attack surface, authentication, injection, least privilege and blast "
        "radius. You name the concrete exploit and the concrete mitigation, and "
        "you never trade real security for the appearance of it.",
        ("vulnerab", "exploit", "authentication", "encryption", "injection",
         "malware", "phishing", "firewall", "threat model", "penetration test",
         "безбедност", "ранлив"),
        aliases=("security engineer", "the security engineer", "pentester",
                 "security researcher", "infosec", "cybersecurity"),
        wants_tools=True,
    ),
    Specialist(
        "chemistry", "the chemist",
        "You are a chemist. You reason about reactions, stoichiometry, materials, "
        "solvents, adhesives, battery chemistry and safety. You give quantities "
        "and you flag anything that is toxic, flammable or otherwise about to ruin "
        "someone's day.",
        ("chemical", "chemistry", "reaction", "molecule", "solvent", "adhesive",
         "epoxy", "corrosion", "electrolyte", "polymer", "хемија", "реакција"),
        aliases=("chemist", "the chemist", "хемичар"),
    ),
    Specialist(
        "medicine", "the physician",
        "You are a physician. You explain how the body and common conditions "
        "generally work, what the usual options are, and when something needs a "
        "real clinician now. You always say this is general health information, "
        "not a diagnosis or personal medical advice, and you never invent a dose.",
        ("symptom", "diagnosis", "medication", "dosage", "injury", "fracture",
         "infection", "blood pressure", "illness", "fever", "симптом"),
        aliases=("doctor", "the doctor", "physician", "лекар", "доктор"),
    ),
    Specialist(
        "biology", "the biologist",
        "You are a biologist. You reason about cells, physiology, genetics, "
        "ecology and evolution, keep scales straight, and distinguish a mechanism "
        "that is established from one that is hypothesized.",
        ("biology", "cellular", "genetic", "protein", "enzyme", "organism",
         "ecology", "evolution", "bacteria", "biological", "биологија"),
        aliases=("biologist", "the biologist", "биолог"),
    ),
    Specialist(
        "writing", "the writer",
        "You are a professional writer and editor. You reason about structure, "
        "clarity, tone and the reader, and you cut what does not earn its place. "
        "You show the improved sentence rather than describing it.",
        ("proofread", "grammar", "wording", "phrasing", "copywriting", "rewrite",
         "paragraph", "prose style"),
        aliases=("writer", "the writer", "editor", "copywriter", "the editor"),
    ),
)


def load_council(extra: dict[str, Any] | None = None) -> tuple[Specialist, ...]:
    """:data:`COUNCIL` plus any ``council.extra`` specialists from config.

    A config entry reusing a built-in key replaces it, so the electrical
    engineer can be re-briefed without touching code. Entries missing a prompt
    are skipped rather than crashing startup.
    """
    members = {s.key: s for s in COUNCIL}
    for key, spec in (extra or {}).items():
        if not isinstance(spec, dict) or not spec.get("prompt"):
            continue
        members[key] = Specialist(
            key=key,
            title=str(spec.get("title", f"the {key} specialist")),
            prompt=str(spec["prompt"]),
            triggers=tuple(str(t).lower() for t in spec.get("triggers", ())),
            aliases=tuple(str(a).lower() for a in spec.get("aliases", ())),
            wants_tools=bool(spec.get("wants_tools", False)),
        )
    return tuple(members.values())


def enabled_council(council: tuple[Specialist, ...],
                    disabled: list[str] | None = None) -> tuple[Specialist, ...]:
    """Drop the specialists switched off in config (``council.disabled``)."""
    off = {d.strip().lower() for d in (disabled or [])}
    return tuple(s for s in council if s.key not in off)


def find_specialist(spoken: str,
                    council: tuple[Specialist, ...] = COUNCIL) -> Specialist | None:
    """Resolve a specialist named out loud ("the electrical engineer")."""
    want = re.sub(r"\s+", " ", spoken.strip().strip(".,!?")).lower()
    if not want:
        return None
    want = re.sub(r"^(?:the|a|an|our|my)\s+", "", want)
    # Macedonian glues the definite article onto the noun ("адвокат" ->
    # "адвокатот", "физичар" -> "физичарот"), so an alias can never be compared
    # for equality — the spoken form is the alias plus a suffix. Stripping the
    # article here keeps the alias table free of every inflected variant.
    #
    # Per WORD, because the article rides the adjective in a phrase:
    # "машински инженер" is spoken "машинскиОТ инженер", with the suffix in
    # the middle. Only words of 5+ characters are touched, so short function
    # words ("за", "на") survive intact.
    bare = " ".join(
        re.sub(r"(?:та|то|те|от|ов|ва)$", "", w) if len(w) >= 5 else w
        for w in want.split(" ")
    )
    for member in council:
        title = member.title.lower().removeprefix("the ")
        if want in (member.key, title) or want.rstrip("s") == member.key:
            return member
        for alias in member.aliases:
            alias = alias.lower()
            if want == alias or bare == alias:
                return member
    # "ask the electrical guy" / "прашај го електроинженерот" — containment,
    # so a title with an extra word or an article still lands.
    for member in council:
        title = member.title.lower().removeprefix("the ")
        if want in title or title in want:
            return member
        if any(a.lower() in want for a in member.aliases if len(a) >= 5):
            return member
    return None


def rank_specialists(question: str, council: tuple[Specialist, ...] = COUNCIL,
                     limit: int = 3) -> list[Specialist]:
    """Who should answer this? Ordered best-first, by trigger-word hits.

    Deliberately dumb and deterministic: asking the model who to ask would
    cost a whole extra round-trip to decide something a word list settles.
    Ties keep roster order, so the result is stable for the same question.
    """
    text = question.lower()
    scored: list[tuple[int, int, Specialist]] = []
    for index, member in enumerate(council):
        # Leading word boundary, not bare containment: "amp" was matching inside
        # "example", "led" inside "pulled", "law" inside "flawless", pulling the
        # wrong specialist into a convene. \b is Unicode-aware, so the Cyrillic
        # STEM triggers ("инфлациј", "економ") still prefix-match their inflected
        # forms, and multi-word triggers ("power supply") are unaffected.
        hits = sum(1 for trigger in member.triggers
                   if re.search(rf"\b{re.escape(trigger)}", text))
        if hits:
            scored.append((-hits, index, member))
    scored.sort()
    return [member for _neg, _i, member in scored[:limit]]


def system_prompt(member: Specialist, macedonian: bool = False) -> str:
    """The full system prompt for one specialist."""
    prompt = f"{member.prompt}\n\n{HOUSE_STYLE}"
    if macedonian:
        prompt += "\n\nAnswer in Macedonian."
    return prompt
