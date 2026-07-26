"""Weather via the free, keyless Open-Meteo API.

Falls back to the configured default city (Skopje) when none is named. Any
network failure returns a clear spoken "I'm offline" rather than crashing.
Supports "today" and "tomorrow" so M4 follow-ups ("and tomorrow?") work.
"""

from __future__ import annotations

import re
from typing import Any

from core import mk
from core.config import WeatherConfig
from skills.base import Skill, SkillRequest, SkillResult

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
#: in execute() doesn't geocode "the moment" and report it can't find it.
_NOT_A_CITY = {"the moment", "the minute", "now", "home", "work",
               "the weekend", "the week", "the day", "school"}


class WeatherSkill(Skill):
    name = "weather"
    description = "Report current weather or tomorrow's forecast for a city."
    routing_phrases = [
        "what's the weather like", "is it going to rain", "do I need a jacket",
        "how hot is it outside", "what's tomorrow's forecast",
    ]

    patterns = [
        # "what's the weather LIKE in London" — the intervening "like" pushed
        # the city out of reach, so it silently reported the default city.
        re.compile(rf"\bweather\b(?:\s+like)?(?:\s+(?:in|for|at)\s+(?P<city>{_CITY}))?",
                   re.IGNORECASE),
        re.compile(rf"\bforecast\b(?:\s+(?:in|for)\s+(?P<city2>{_CITY}))?", re.IGNORECASE),
        re.compile(r"\b(?:how\s+(?:hot|cold)|temperature)\b", re.IGNORECASE),
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

    def __init__(self, config: WeatherConfig) -> None:
        self._config = config

    async def _lookup(self, client: Any, name: str) -> tuple[float, float, str] | None:
        resp = await client.get(_GEOCODE_URL, params={"name": name, "count": 1})
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            return None
        r = results[0]
        return r["latitude"], r["longitude"], r["name"]

    async def _geocode(self, client: Any, city: str) -> tuple[float, float, str] | None:
        """Find a city, retrying romanised when it was spoken in Cyrillic.

        Open-Meteo's geocoder indexes Latin names: "Скопје" returns nothing
        while "Skopje" resolves, so a Macedonian weather question would fail on
        its own capital without this fallback.
        """
        found = await self._lookup(client, city)
        if found is None and mk.is_cyrillic(city):
            found = await self._lookup(client, mk.to_latin(city))
        return found

    async def execute(self, request: SkillRequest) -> SkillResult:
        import httpx

        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        city = _strip_time_words(
            (request.args.get("city") or gd.get("city") or gd.get("city2")
             or gd.get("city3") or gd.get("city4") or gd.get("city5")
             or "").strip(" ?.!"))
        if not city:
            # Several patterns ("how hot is it in Dubai", "is it raining in
            # Paris", "do I need a jacket in London") carry no city GROUP, so
            # honour a trailing "in/for/at <place>" before defaulting to Skopje.
            tail = re.search(r"\b(?:in|for|at)\s+(?P<c>[\w .'-]+?)\s*[?.!]*$",
                             _strip_time_words(request.text), re.IGNORECASE)
            if tail:
                candidate = tail.group("c").strip(" ?.!")
                if candidate and candidate.lower() not in _NOT_A_CITY:
                    city = candidate
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
                else:
                    lat, lon, place = self._config.latitude, self._config.longitude, self._config.default_city

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
