"""Weather via the free, keyless Open-Meteo API.

Falls back to the configured default city (Skopje) when none is named. Any
network failure returns a clear spoken "I'm offline" rather than crashing.
Supports "today" and "tomorrow" so M4 follow-ups ("and tomorrow?") work.
"""

from __future__ import annotations

import re
from typing import Any

from core.config import WeatherConfig
from skills.base import Skill, SkillRequest, SkillResult

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_OFFLINE = "I can't reach the weather service. I appear to be offline."

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


class WeatherSkill(Skill):
    name = "weather"
    description = "Report current weather or tomorrow's forecast for a city."

    patterns = [
        re.compile(r"\bweather\b(?:\s+(?:in|for|at)\s+(?P<city>[a-z .'-]+))?", re.IGNORECASE),
        re.compile(r"\bforecast\b(?:\s+(?:in|for)\s+(?P<city2>[a-z .'-]+))?", re.IGNORECASE),
        re.compile(r"\b(?:how\s+(?:hot|cold)|temperature)\b", re.IGNORECASE),
        re.compile(r"\b(?:will\s+it|is\s+it\s+going\s+to)\s+rain\b", re.IGNORECASE),
    ]

    def __init__(self, config: WeatherConfig) -> None:
        self._config = config

    async def _geocode(self, client: Any, city: str) -> tuple[float, float, str] | None:
        resp = await client.get(_GEOCODE_URL, params={"name": city, "count": 1})
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            return None
        r = results[0]
        return r["latitude"], r["longitude"], r["name"]

    async def execute(self, request: SkillRequest) -> SkillResult:
        import httpx

        gd = request.match.groupdict() if request.match else {}
        city = (request.args.get("city") or gd.get("city") or gd.get("city2") or "").strip(" ?.!")
        when = (request.args.get("when") or "").lower()
        if "tomorrow" in request.text.lower():
            when = "tomorrow"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if city:
                    geo = await self._geocode(client, city)
                    if geo is None:
                        return SkillResult(f"I couldn't find a place called {city}.", success=False)
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
        except httpx.HTTPError:
            return SkillResult(_OFFLINE, success=False)

        if when == "tomorrow":
            daily = data["daily"]
            cond = _WMO.get(daily["weather_code"][1], "unclear")
            hi, lo = round(daily["temperature_2m_max"][1]), round(daily["temperature_2m_min"][1])
            return SkillResult(
                f"Tomorrow in {place}: {cond}, between {lo} and {hi} degrees.",
                data={"place": place, "when": "tomorrow"},
            )
        cur = data["current"]
        cond = _WMO.get(cur["weather_code"], "unclear")
        temp = round(cur["temperature_2m"])
        return SkillResult(
            f"It's {temp} degrees and {cond} in {place}.",
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
