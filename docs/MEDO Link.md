# MEDO Link

> **EXPERIMENTAL.** One protocol, two thin clients, one reference device.
> The protocol is stable but young — pin your MEDO version. This is not a
> public support contract.

Turn the companion API (`:8710`) into a device layer: an ESP32, Raspberry Pi,
or anything with HTTP uploads a **manifest** describing its capabilities, and
every capability instantly becomes

- an **LLM tool** — the capability's `description` *is* the tool description
  the model reads, so write it for the model ("Walk the robot dog forward…
  It PHYSICALLY MOVES"), and
- optional **fast-path voice patterns** (regexes; no LLM round-trip).

Capabilities flagged `requires_confirmation: true` route through MEDO's
existing spoken yes/no safety gate (English + Macedonian) *before* dispatch —
the council's rule: anything that moves, heats, or spends confirms first.

## Security

- Every `/link/*` request needs the bearer token from `secrets.local.yaml`
  → `remote.token` (`Authorization: Bearer …`, or `?token=` for websockets).
- LAN only — the port must never be forwarded. Same rule as everything
  else on `:8710`.

## Protocol

| Step | Call | Notes |
|------|------|-------|
| register | `POST /link/register` with the manifest JSON | Re-POST to update; capabilities are replaced atomically. Manifests persist — MEDO restores them on restart. 422 + `errors[]` on an invalid manifest. |
| receive (http_poll) | `GET /link/commands/{device_id}` every ~1.5 s | Returns `{"commands": [{"id", "capability", "params"}, …]}` (usually empty). Polling is also your **heartbeat** — stop polling for ~20 s and MEDO treats the device as offline. |
| receive (websocket) | `GET /link/ws?device_id=…&token=…` | Commands arrive as `{"type": "command", "id", "capability", "params"}` frames. |
| answer | `POST /link/result` with `{"device_id", "id", "ok", "message"}` | `message` is spoken by MEDO ("Walked 3 steps."). Websocket devices may instead send the same JSON as a frame. |
| list | `GET /link/devices` | For UIs: name, online, transport, capability count (the HUD CONFIG tab shows this). |

Timeouts: a dispatched command that gets no result within 10 s is answered
"didn't respond"; an offline device's tools stay registered and invocations
say so instead of dispatching.

## Manifest

Formal schema: [`link/manifest-schema.json`](../link/manifest-schema.json).
Reference device: [`link/examples/robodog-manifest.json`](../link/examples/robodog-manifest.json)
(+ a stub sketch showing where real firmware plugs in).

Minimal example:

```json
{
  "device_id": "lamp",
  "name": "the desk lamp",
  "capabilities": [
    {"name": "on", "description": "Turn the desk lamp on.",
     "fast_patterns": ["\\blamp on\\b"]}
  ]
}
```

## Clients

- **Python (RPi)**: `pip install -e link/clients/python`, see
  `link/clients/python/example_led.py`. ~100 lines, asyncio + aiohttp.
- **Arduino/ESP32**: copy `link/clients/arduino/MedoLink.h` next to your
  sketch. Needs the ESP32 core (WiFi/HTTPClient) + ArduinoJson 7.x.
  ~200 lines, HTTP polling.

## Why this design

Manifest-driven tools mean **zero per-device code in MEDO** — the same
mechanism that serves built-in skills, plugins, and MCP tools serves devices.
HTTP/WS over MQTT because the API already exists and a broker is one more
always-on dependency; per-capability confirmation because the safety decision
belongs to the device author who knows which commands move motors. See
`docs/Decisions.md`.
