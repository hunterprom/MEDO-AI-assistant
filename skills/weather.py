"""Weather via the free, keyless Open-Meteo API.

Falls back to the configured default city (Skopje) when none is named. Any
network failure returns a clear spoken "I'm offline" rather than crashing.
Supports "today" and "tomorrow" so M4 follow-ups ("and tomorrow?") work.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from core import mk
from core.config import WeatherConfig
from skills.base import Skill, SkillRequest, SkillResult

def _fold(text: str) -> str:
    """Casefold and strip accents, so "Macedônia" and "Macedonia" compare equal."""
    stripped = "".join(c for c in unicodedata.normalize("NFD", text or "")
                       if not unicodedata.combining(c))
    return stripped.casefold().strip()


#: Spoken place names the geocoder doesn't index under the form people say.
#: "Macedonia" is the name locals use for the country the gazetteer files only
#: as "North Macedonia" (renamed 2019), so the query otherwise lands on one of
#: the several US villages of that name.
_PLACE_ALIASES = {
    "macedonia": "North Macedonia",
    "македонија": "North Macedonia",
    "makedonija": "North Macedonia",
    "holland": "Netherlands",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
}

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_OFFLINE = "I can't reach the weather service. I appear to be offline."
_OFFLINE_MK = "Не можам да ја добијам прогнозата — изгледа дека сум офлајн."

# Condensed WMO weather-code descriptions.
_WMO = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "snow showers", 95: "thunderstorms",
    96: "thunderstorms with hail", 99: "thunderstorms with hail",
}

_WMO_MK = {
    0: "ведро", 1: "претежно ведро", 2: "делумно облачно", 3: "облачно",
    45: "магливо", 48: "магливо", 51: "слаба ројна", 53: "ројна", 55: "силна ројна",
    61: "слаб дожд", 63: "дожд", 65: "силен дожд", 66: "леден дожд", 67: "леден дожд",
    71: "слаб снег", 73: "снег", 75: "силен снег", 77: "снежни зрна",
    80: "плускавици", 81: "плускавици", 82: "силни плускавици",
    85: "снежни плускавици", 86: "снежни плускавици", 95: "грмежи",
    96: "грмежи со град", 99: "грмежи со град",
}

#: City names may be Cyrillic ("времето во Скопје"), so \w — not [a-z] — and a
#: leading "во/за" is stripped by the pattern rather than the geocoder.
_CITY = r"[\w .'-]+"

#: The greedy _CITY swallows a trailing time word ("weather for tomorrow" ->
#: city "tomorrow" -> geocode fails -> the default-city forecast is never
#: reached). Strip it; `when` is derived from the raw text separately.
_TRAILING_TIME = re.compile(
    r"\s*\b(?:tomorrow|today|tonight|now|right\s+now|later|this\s+(?:week|weekend|"
    r"morning|afternoon|evening)|next\s+week|утре|денес|вечер\w*|сега)\b\s*$",
    re.IGNORECASE)


def _strip_time_words(city: str) -> str:
    return _TRAILING_TIME.sub("", city).strip()


#: A trailing "in/for/at <X>" that is plainly not a place, so the city fallback
#: doesn't geocode "the moment" and report it can't find it.
_NOT_A_CITY = {"the moment", "the minute", "now", "home", "work",
               "the weekend", "the week", "the day", "school",
               # common "my current location" nouns — a leading article is
               # stripped before this check, so "the office" is caught too.
               "office", "house", "gym", "room", "building", "apartment",
               "flat", "garden", "backyard", "kitchen", "bedroom", "bathroom",
               "hotel", "hospital"}


#: Determiners that mean "the default location", never a real place to geocode
#: ("my house", "this area", "next quarter"). Articles (the/a/an) are NOT here:
#: real cities lead with them ("The Hague", "the Netherlands").
_NON_CITY_LEAD = re.compile(
    r"(?:my|your|our|his|her|their|this|that|next|last)\b", re.IGNORECASE)


def _looks_like_city(candidate: str, *, allow_article: bool = False) -> bool:
    """A captured place is a real city only when it isn't a common non-place
    ('work', 'home', 'next quarter') and carries no possessive/deictic
    determiner. ``allow_article`` keeps a leading the/a/an — the pattern's own
    'in/for/at <city>' group frames a proper name like 'The Hague', whereas a
    bare trailing tail ('in the office') should still reject the article."""
    candidate = candidate.strip(" ?.!")
    if not candidate:
        return False
    # Ignore a leading article for the non-place check, so "the office" is
    # rejected the same as "office" — while a real "The Hague" survives, its
    # core word not being a non-place.
    core = re.sub(r"^(?:the|a|an)\s+", "", candidate, flags=re.IGNORECASE)
    if candidate.lower() in _NOT_A_CITY or core.lower() in _NOT_A_CITY:
        return False
    if _NON_CITY_LEAD.match(candidate):
        return False
    if not allow_article and re.match(r"(?:the|a|an)\b", candidate, re.IGNORECASE):
        return False
    return True


def _city_from_tail(text: str) -> str | None:
    """A trailing 'in/for/at <place>' -> the place, when it looks like a real
    city, else None. A real city is a proper noun with no leading determiner —
    so 'in Dubai' yields 'Dubai' but 'in the office' / 'at the wedding' yield
    None (those are valid weather questions about the default location)."""
    tail = re.search(r"\b(?:in|for|at)\s+(?P<c>[\w .'-]+?)\s*[?.!]*$",
                     _strip_time_words(text), re.IGNORECASE)
    if not tail:
        return None
    candidate = tail.group("c").strip(" ?.!")
    return candidate if _looks_like_city(candidate) else None


class WeatherSkill(Skill):
    name = "weather"
    description = "Report current weather or tomorrow's forecast for a city."
    routing_phrases = [
        "what's the weather like", "is it going to rain", "do I need a jacket",
        "how hot is it outside", "what's tomorrow's forecast",
    ]

    patterns = [
        # A bare \bweather\b used to fire on any sentence with the word ("I'm
        # feeling under the weather", "nice weather", "fair-weather friends").
        # Require a question frame, an "(in|for|at) <city>" location, or a bare
        # "weather" command. "how's the weather" is covered by its own pattern.
        re.compile(rf"\bwhat(?:'?s| is)\s+(?:the\s+)?weather(?:\s+like)?"
                   rf"(?:\s+(?:in|for|at)\s+(?P<city>{_CITY}))?", re.IGNORECASE),
        re.compile(rf"\bweather(?:\s+like)?\s+(?:in|for|at)\s+(?P<cityb>{_CITY})"
                   rf"|^\s*(?:the\s+)?weather\s*[?.!]*$", re.IGNORECASE),
        # "forecast" alone answered "sales/revenue forecast for Q3" with weather;
        # keep bare weather "forecast" but not a business one named right before.
        re.compile(rf"\b(?<!sales\s)(?<!revenue\s)(?<!budget\s)(?<!market\s)"
                   rf"(?<!financial\s)(?<!economic\s)(?<!traffic\s)(?<!earnings\s)"
                   rf"forecast\b(?:\s+(?:in|for)\s+(?P<city2>{_CITY}))?", re.IGNORECASE),
        # "temperature" alone answered cooking/hardware/body questions ("what
        # temperature to cook chicken", "GPU temperature") with the outdoor temp.
        # Require a weather frame.
        re.compile(r"\bhow\s+(?:hot|cold)\b(?!\s+(?:should|to|do|does|can|would|is\s+the))|"
                   # bare "temperature" is weather UNLESS a hardware/body word is
                   # named right before it (GPU/body temperature) or it's a
                   # cooking/instruction frame right after ("temperature to cook",
                   # "...should be"). Keeps "temperature tomorrow / for tomorrow /
                   # be tomorrow" as weather.
                   r"(?<!gpu )(?<!cpu )(?<!body )(?<!oven )(?<!water )(?<!engine )"
                   r"(?<!cooking )\btemperature\b"
                   r"(?!\s+(?:to|should|of|inside|when|at\s+which)\b)", re.IGNORECASE),
        # Bare precipitation questions used to fabricate on the LLM path:
        # "will it snow", "is it going to rain", plain "is it raining/snowing".
        re.compile(r"\b(?:will\s+it|is\s+it\s+going\s+to)\s+(?:rain|snow)\b", re.IGNORECASE),
        re.compile(r"\bis\s+it\s+(?:raining|snowing)\b", re.IGNORECASE),
        # Everyday phrasings that mean "give me the weather".
        re.compile(r"\bdo\s+i\s+need\s+(?:a|an|my)\s+(?:jacket|coat|umbrella|"
                   r"raincoat|sweater)\b", re.IGNORECASE),
        re.compile(r"\bwhat(?:'?s| is)\s+it\s+like\s+outside\b", re.IGNORECASE),
        re.compile(r"\bhow(?:'?s| is)\s+(?:the\s+weather|it\s+outside)\b", re.IGNORECASE),
        re.compile(r"\bis\s+it\s+(?:hot|cold|sunny|raining|snowing|windy)\s+"
                   r"(?:out|outside|today)\b", re.IGNORECASE),
        # MK. "време" alone is skipped on purpose — it means both "weather" and
        # "time", so only the unambiguous phrasings are claimed here.
        re.compile(rf"\bкакво\s+е\s+времето\b(?:\s+(?:во|за)\s+(?P<city3>{_CITY}))?",
                   re.IGNORECASE),
        re.compile(rf"\bвремето\s+(?:во|за)\s+(?P<city4>{_CITY})", re.IGNORECASE),
        re.compile(rf"\bпрогноза(?:та)?\b(?:\s+(?:во|за)\s+(?P<city5>{_CITY}))?",
                   re.IGNORECASE),
        re.compile(r"\bколку\s+степени\b|\bтемператур(?:а|ата)\b", re.IGNORECASE),
        re.compile(r"\bќе\s+врне\b|\bдали\s+ќе\s+врне\b", re.IGNORECASE),
    ]

    #: "направи апликација за прогноза" asks MEDO to BUILD something; the bare
    #: "прогноза" pattern above claimed it and read out today's temperature
    #: instead — a live transcript showed exactly that, three times running.
    #: Weather words name the SUBJECT of the build here, not the question.
    _BUILD_REQUEST = re.compile(
        rf"\b(?:{mk.MAKE}){mk.CLITICS}\s+(?:ед[нeaо]\w*\s+)?"
        r"(?:апликациј\w*|аплкациј\w*|програм\w*|алатк\w*|скрипт\w*|сајт\w*|"
        r"веб\s*стран\w*|игр[аи]\w*|модел\w*)"
        r"|\b(?:make|build|create|write|code|scaffold|design)\s+(?:me\s+)?"
        r"(?:a|an)\s+(?:\w+\s+){0,2}?"
        r"(?:app|application|program|tool|script|website|web\s*app|widget|"
        r"dashboard|game)\b",
        re.IGNORECASE)

    def match(self, text: str):
        if self._BUILD_REQUEST.search(text or ""):
            return None
        return super().match(text)

    def __init__(self, config: WeatherConfig) -> None:
        self._config = config

    async def _lookup(self, client: Any, name: str) -> tuple[float, float, str] | None:
        """Geocode ``name``, choosing the place a person would have meant.

        Open-Meteo's first result is NOT the best one: it fuzzy-matches
        alternate names, so "Macedonia" led with Jonesboro, Louisiana (pop.
        4587) and MEDO cheerfully reported Louisiana weather for the user's own
        country. Ask for several and rank them ourselves — an exact name match
        first, then a capital, then population.
        """
        resp = await client.get(_GEOCODE_URL, params={"name": name, "count": 8})
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            return None
        r = max(results, key=lambda x: self._geo_rank(x, name))
        return r["latitude"], r["longitude"], r["name"]

    @staticmethod
    def _geo_rank(result: dict, query: str) -> tuple[int, int, int]:
        """Sort key for a geocoder hit: (exact name, is a capital, population)."""
        name = str(result.get("name") or "")
        exact = int(_fold(name) == _fold(query))
        capital = int(str(result.get("feature_code") or "") == "PPLC")
        try:
            population = int(result.get("population") or 0)
        except (TypeError, ValueError):
            population = 0
        return exact, capital, population

    async def _geocode(self, client: Any, city: str) -> tuple[float, float, str] | None:
        """Find a city, retrying romanised when it was spoken in Cyrillic.

        Open-Meteo's geocoder indexes Latin names: "Скопје" returns nothing
        while "Skopje" resolves, so a Macedonian weather question would fail on
        its own capital without this fallback.
        """
        city = _PLACE_ALIASES.get(_fold(city), city)
        found = await self._lookup(client, city)
        if found is None and mk.is_cyrillic(city):
            found = await self._lookup(client, mk.to_latin(city))
        return found

    async def execute(self, request: SkillRequest) -> SkillResult:
        import httpx

        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        city = _strip_time_words(
            (request.args.get("city") or gd.get("city") or gd.get("cityb")
             or gd.get("city2") or gd.get("city3") or gd.get("city4")
             or gd.get("city5") or "").strip(" ?.!"))
        if city and not _looks_like_city(city, allow_article=True):
            # A pattern's greedy city group can capture a non-place ("weather at
            # work", "forecast for next quarter"); treat it as the default
            # location instead of geocoding it as a foreign city. Articles are
            # kept so a real "The Hague" / "the Netherlands" still geocodes.
            city = ""
        if not city:
            # Several patterns ("how hot is it in Dubai", "is it raining in
            # Paris", "do I need a jacket in London") carry no city GROUP, so
            # honour a trailing "in/for/at <place>" before defaulting to Skopje.
            city = _city_from_tail(request.text) or ""
        when = (request.args.get("when") or "").lower()
        if "tomorrow" in request.text.lower() or "утре" in request.text.lower():
            when = "tomorrow"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if city:
                    geo = await self._geocode(client, city)
                    if geo is None:
                        return SkillResult(
                            f"Не најдов место со име {city}." if speak_mk
                            else f"I couldn't find a place called {city}.",
                            success=False)
                    lat, lon, place = geo
                    # The geocoder only speaks Latin, so it hands back "Skopje"
                    # for a city the user pronounced "Скопје". Say back what
                    # they said rather than dropping a Latin island into a
                    # Cyrillic sentence.
                    if speak_mk and mk.is_cyrillic(city):
                        place = city
                else:
                    lat, lon, place = self._config.latitude, self._config.longitude, self._config.default_city
                if speak_mk and not mk.is_cyrillic(place):
                    place = mk.to_cyrillic(place) or place

                resp = await client.get(_FORECAST_URL, params={
                    "latitude": lat, "longitude": lon,
                    "current": "temperature_2m,weather_code",
                    "daily": "temperature_2m_max,temperature_2m_min,weather_code",
                    "timezone": "auto", "forecast_days": 2,
                })
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError):
            # ValueError covers a non-JSON 200 — a captive-portal / proxy login
            # page slips past raise_for_status, and resp.json() then raises
            # JSONDecodeError (a ValueError). Degrade to the offline message.
            return SkillResult(_OFFLINE_MK if speak_mk else _OFFLINE, success=False)

        codes = _WMO_MK if speak_mk else _WMO
        unclear = "нејасно" if speak_mk else "unclear"
        try:
            if when == "tomorrow":
                daily = data["daily"]
                cond = codes.get(daily["weather_code"][1], unclear)
                hi, lo = round(daily["temperature_2m_max"][1]), round(daily["temperature_2m_min"][1])
                return SkillResult(
                    f"Утре во {place}: {cond}, помеѓу {lo} и {hi} степени."
                    if speak_mk else
                    f"Tomorrow in {place}: {cond}, between {lo} and {hi} degrees.",
                    data={"place": place, "when": "tomorrow"},
                )
            cur = data["current"]
            cond = codes.get(cur["weather_code"], unclear)
            temp = round(cur["temperature_2m"])
        except (KeyError, IndexError, TypeError):
            # A 200 whose JSON is missing the fields we need (API shape change,
            # error object) — treat as unavailable rather than crash the turn.
            return SkillResult(_OFFLINE_MK if speak_mk else _OFFLINE, success=False)
        return SkillResult(
            f"Во {place} е {temp} степени и {cond}." if speak_mk
            else f"It's {temp} degrees and {cond} in {place}.",
            data={"place": place, "when": "now", "temp_c": temp},
        )

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "city name; omit for the default"},
                        "when": {"type": "string", "enum": ["now", "tomorrow"]},
                    },
                    "required": [],
                },
            },
        }
