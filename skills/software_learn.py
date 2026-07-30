"""Learn an app's UI, then find / go to its controls (the learning foundation).

Three skills, all routing through the SAME registry + policy gate as any other:

* :class:`LearnAppSkill` — "learn how to use CapCut" scans the app (safely:
  reveal menus, back out) and remembers where its controls are.
* :class:`AppLocateSkill` — "where is the effects search in CapCut" answers from
  what was learned (read-only, no actuation).
* :class:`AppNavigateSkill` — "open the effects panel in CapCut" focuses the app
  and clicks the learned control (actuation → policy-gated like any PC control).

Only the user's own voice/text drives these — an instruction that arrives from
untrusted content (a document, a web page, another app) is refused.
"""

from __future__ import annotations

import re
from typing import Optional

from security.capabilities import Capability
from skills.base import Skill, SkillRequest, SkillResult
from software import knowledge
from software.knowledge import DEFAULT_MAPS_DIR


def _app_id(name: str) -> str:
    return (name or "").strip().lower()


def _untrusted(ctx: dict) -> bool:
    return bool(ctx.get("untrusted") or ctx.get("provenance") == "untrusted")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).strip(" ?.!")


class LearnAppSkill(Skill):
    """Scan an app the user names and remember its UI."""

    name = "learn_app"
    description = ("Learn how to use a specific local app by scanning its UI "
                   "(menus, buttons, panels) so you know where its controls are "
                   "for later. Use when the user says to learn/study/map an app.")
    controls_pc = True                        # focuses + reveals menus (actuation)
    capabilities = frozenset({Capability.CONTROL_INPUT})
    routing_phrases = ["learn how to use CapCut", "map Photoshop's interface",
                       "scan this app so you know it", "study the DaVinci UI"]

    def __init__(self, mechanisms, walker, *, base_dir=DEFAULT_MAPS_DIR,
                 vision=None) -> None:
        self._mech = mechanisms
        self._walker = walker
        self._base_dir = base_dir
        # Optional async callable(app_hint) -> list[UIElement]: the vision backup
        # (software.vision_probe.VisionProbe), run only when the UIA tree is thin.
        self._vision = vision
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (
            r"\blearn\s+(?:how\s+to\s+use|to\s+use)\s+(?:the\s+)?(?P<app>[\w .+-]{2,40}?)(?:\s+app)?\s*$",
            r"\blearn\s+(?:the\s+)?(?P<app>[\w .+-]{2,40}?)\s+(?:app|ui|interface)\s*$",
            r"\b(?:scan|map|explore|study)\s+(?:the\s+)?(?P<app>[\w .+-]{2,40}?)(?:'s)?\s*(?:app|ui|interface|window)?\s*$",
        )]

    def _app(self, request: SkillRequest) -> str:
        if request.match is not None:
            got = request.match.groupdict().get("app")
            if got:
                return _clean(got)
        return _clean(str(request.args.get("app", "")))

    async def execute(self, request: SkillRequest) -> SkillResult:
        if _untrusted(request.context or {}):
            return SkillResult("I only learn apps when you ask me directly.",
                               success=False, data={"refused": "untrusted"})
        app = self._app(request)
        if not app:
            return SkillResult("Which app should I learn?", success=False)
        # Bring it to the front; if it isn't open we can't scan it.
        if not self._mech.focus_app(app):
            return SkillResult(
                f"I couldn't find {app} open. Open it first, then say "
                f"“learn {app}”.", success=False,
                data={"reason": "not_open"})
        from software.ui_scan import is_thin, scan_app, vision_augment

        app_map = scan_app(self._walker, app_id=_app_id(app), display_name=app,
                           reveal_menus=True, window_hint=app)
        # Vision backup: when UIA saw too little (custom/Electron apps like
        # CapCut), look at a screenshot and add the controls it reveals.
        seen_vision = 0
        if self._vision is not None and is_thin(app_map):
            try:
                velems = await self._vision(app)
            except Exception:
                velems = []
            if velems:
                app_map = vision_augment(app_map, velems)
                seen_vision = sum(1 for e in app_map.elements
                                  if e.source == "vision")
        knowledge.save_map(app_map, self._base_dir)
        menus = ", ".join(list(app_map.menus)[:5])
        examples = ", ".join(
            f"“{e.name}”" for e in app_map.elements[:3] if e.name)
        extra = f" Menus: {menus}." if menus else ""
        vis = f" ({seen_vision} seen visually)" if seen_vision else ""
        tail = f" Try asking me to find {examples}." if examples else ""
        return SkillResult(
            f"Learned {app} — I mapped {app_map.element_count()} controls{vis}."
            f"{extra}{tail}", data={"app": app, "count": app_map.element_count(),
                                    "vision": seen_vision,
                                    "menus": list(app_map.menus)})

    def tool_schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": {
                "app": {"type": "string",
                        "description": "The app to learn, e.g. 'CapCut'."}},
                "required": ["app"]}}}


class _AppLookupSkill(Skill):
    """Shared parse + map lookup for the locate / navigate pair."""

    def __init__(self, mechanisms, *, base_dir=DEFAULT_MAPS_DIR) -> None:
        self._mech = mechanisms
        self._base_dir = base_dir

    def _parse(self, request: SkillRequest):
        app = target = ""
        if request.match is not None:
            gd = request.match.groupdict()
            app, target = _clean(gd.get("app", "")), _clean(gd.get("target", ""))
        app = app or _clean(str(request.args.get("app", "")))
        target = target or _clean(str(request.args.get("target", "")))
        return app, target

    def _lookup(self, app: str, target: str):
        app_map = knowledge.load_map(_app_id(app), self._base_dir)
        if app_map is None:
            return None, SkillResult(
                f"I haven't learned {app} yet. Say “learn {app}” and "
                f"I'll map it.", success=False, data={"reason": "not_learned"})
        hits = knowledge.search(app_map, target)
        if not hits:
            return None, SkillResult(
                f"I didn't find anything like “{target}” in what I "
                f"learned about {app}. I can re-learn it if it changed.",
                success=False, data={"reason": "no_match"})
        return (app_map, hits[0][0]), None

    def tool_schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": {
                "app": {"type": "string", "description": "The learned app."},
                "target": {"type": "string",
                           "description": "The control/panel to find."}},
                "required": ["app", "target"]}}}


class AppLocateSkill(_AppLookupSkill):
    """Tell the user WHERE a control is (read-only — no actuation)."""

    name = "locate_in_app"
    description = ("Say where a control/panel/feature is in an app you've "
                   "learned, without touching anything. Use for 'where is X in "
                   "APP'.")
    controls_pc = False
    routing_phrases = ["where is the effects search in CapCut",
                       "locate the export button in Premiere"]

    def __init__(self, mechanisms, *, base_dir=DEFAULT_MAPS_DIR) -> None:
        super().__init__(mechanisms, base_dir=base_dir)
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (
            r"\bwhere(?:'s| is| are)\s+(?:the\s+)?(?P<target>.+?)\s+in\s+(?P<app>[\w .+-]{2,40})\s*\??$",
            r"\b(?:find|locate|show me)\s+(?:the\s+)?(?P<target>.+?)\s+in\s+(?P<app>[\w .+-]{2,40})\s*\??$",
            r"\bin\s+(?P<app>[\w .+-]{2,40}?),?\s+where(?:'s| is)\s+(?:the\s+)?(?P<target>.+?)\s*\??$",
        )]

    async def execute(self, request: SkillRequest) -> SkillResult:
        if _untrusted(request.context or {}):
            return SkillResult("I only do that when you ask me directly.",
                               success=False, data={"refused": "untrusted"})
        app, target = self._parse(request)
        if not app or not target:
            return SkillResult("Tell me what to find and in which app.",
                               success=False)
        found, err = self._lookup(app, target)
        if err is not None:
            return err
        _, el = found
        return SkillResult(
            f"In {app}, “{el.name}” is {el.location()}. Say "
            f"“open {el.name} in {app}” and I'll go there.",
            data={"app": app, "name": el.name, "where": el.location()})


class AppNavigateSkill(_AppLookupSkill):
    """Focus the app and click a learned control (actuation → policy-gated)."""

    name = "navigate_in_app"
    description = ("Go to / click a control in an app you've learned — focuses "
                   "the app and activates the control. Prefer this to describing "
                   "steps. Use for 'open/go to/click X in APP'.")
    controls_pc = True
    capabilities = frozenset({Capability.CONTROL_INPUT})
    routing_phrases = ["open the effects panel in CapCut",
                       "go to the export button in Premiere",
                       "click the search box in CapCut"]

    def __init__(self, mechanisms, *, base_dir=DEFAULT_MAPS_DIR) -> None:
        super().__init__(mechanisms, base_dir=base_dir)
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (
            r"\b(?:open|go to|goto|click|press|select|activate)\s+(?:the\s+)?(?P<target>.+?)\s+in\s+(?P<app>[\w .+-]{2,40})\s*\??$",
            r"\bin\s+(?P<app>[\w .+-]{2,40}?),?\s+(?:open|go to|goto|click|press|select|activate)\s+(?:the\s+)?(?P<target>.+?)\s*\??$",
        )]

    async def execute(self, request: SkillRequest) -> SkillResult:
        if _untrusted(request.context or {}):
            return SkillResult("I only control apps when you ask me directly.",
                               success=False, data={"refused": "untrusted"})
        app, target = self._parse(request)
        if not app or not target:
            return SkillResult("Tell me what to open and in which app.",
                               success=False)
        found, err = self._lookup(app, target)
        if err is not None:
            return err
        _, el = found
        self._mech.focus_app(app)
        result = self._mech.ui_automation(app, el.name)
        if result.success:
            return SkillResult(f"Opened “{el.name}” in {app}.",
                               data={"app": app, "name": el.name,
                                     "verified": result.verified})
        # A vision-learned control UIA can't name: click where the model saw it,
        # re-resolved against the LIVE window (robust if it moved, same layout).
        if el.source == "vision" and el.vision_xy:
            from software.ui_scan import window_point

            rect = self._mech.foreground_rect()
            if rect:
                px, py = window_point(el.vision_xy, rect)
                click = self._mech.click_point(px, py)
                if click.success:
                    return SkillResult(
                        f"Clicked “{el.name}” in {app} where I saw it.",
                        data={"app": app, "name": el.name, "via": "vision",
                              "point": [px, py], "verified": False})
        return SkillResult(
            f"I know where “{el.name}” is in {app} ({el.location()}), "
            f"but couldn't activate it just now.", success=False,
            data={"app": app, "name": el.name})
