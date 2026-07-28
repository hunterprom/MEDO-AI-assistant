// MedoLink.h — EXPERIMENTAL single-header MEDO Link client for ESP32/Arduino.
//
// One protocol, HTTP polling (works on every core with WiFi + HTTPClient):
//   1) POST /link/register  with your manifest JSON (+ bearer token)
//   2) GET  /link/commands/{device_id} every ~1.5 s (this is your heartbeat)
//   3) run the matching callback, POST /link/result with its outcome
//
// Dependencies (tested versions):
//   - ESP32 Arduino core 2.x/3.x (WiFi.h, HTTPClient.h)
//   - ArduinoJson 7.x  (https://arduinojson.org)
// The bearer token comes from MEDO's secrets.local.yaml -> remote.token.
//
// Usage:
//   #include "MedoLink.h"
//   MedoLink link("192.168.1.20:8710", "TOKEN", MANIFEST_JSON);
//   void setup() {
//     WiFi.begin(ssid, pass); /* ...wait for WL_CONNECTED... */
//     link.onCommand([](const char* cap, JsonObjectConst params,
//                       String& msg) -> bool {
//       if (strcmp(cap, "led_on") == 0) { digitalWrite(2, HIGH); msg = "On."; return true; }
//       msg = "unknown capability"; return false;
//     });
//     link.begin();                       // registers the manifest
//   }
//   void loop() { link.poll(); delay(100); }
//
// EXPERIMENTAL: the protocol is stable but young — pin your MEDO version.

#ifndef MEDO_LINK_H
#define MEDO_LINK_H

#include <Arduino.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

class MedoLink {
 public:
  // handler(capability, params, out message) -> ok?
  typedef bool (*CommandHandler)(const char* capability,
                                 JsonObjectConst params, String& message);

  MedoLink(const char* hostPort, const char* token, const char* manifestJson,
           unsigned long pollMs = 1500)
      : _base(String("http://") + hostPort), _token(token),
        _manifest(manifestJson), _pollMs(pollMs) {}

  void onCommand(CommandHandler handler) { _handler = handler; }

  // Registers the manifest; call after WiFi is up. True on HTTP 200.
  bool begin() {
    JsonDocument doc;
    if (deserializeJson(doc, _manifest)) return false;
    _deviceId = String((const char*)(doc["device_id"] | ""));
    if (_deviceId.isEmpty()) return false;
    int code = _post("/link/register", _manifest);
    _registered = (code == 200);
    return _registered;
  }

  // Call from loop(); fetches + answers queued commands at the poll cadence.
  void poll() {
    if (!_registered || millis() - _lastPoll < _pollMs) return;
    _lastPoll = millis();
    HTTPClient http;
    http.begin(_base + "/link/commands/" + _deviceId);
    _auth(http);
    if (http.GET() != 200) { http.end(); return; }
    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, http.getString());
    http.end();
    if (err) return;
    for (JsonObjectConst cmd : doc["commands"].as<JsonArrayConst>()) {
      _dispatch(cmd);
    }
  }

 private:
  void _dispatch(JsonObjectConst cmd) {
    String message;
    bool ok = false;
    if (_handler != nullptr) {
      ok = _handler(cmd["capability"] | "", cmd["params"].as<JsonObjectConst>(),
                    message);
    } else {
      message = "no handler installed";
    }
    JsonDocument out;
    out["device_id"] = _deviceId;
    out["id"] = cmd["id"];
    out["ok"] = ok;
    out["message"] = message;
    String body;
    serializeJson(out, body);
    _post("/link/result", body.c_str());
  }

  int _post(const char* path, const char* json) {
    HTTPClient http;
    http.begin(_base + path);
    http.addHeader("Content-Type", "application/json");
    _auth(http);
    int code = http.POST((uint8_t*)json, strlen(json));
    http.end();
    return code;
  }

  void _auth(HTTPClient& http) {
    if (_token.length()) http.addHeader("Authorization", "Bearer " + _token);
  }

  String _base, _token, _manifest, _deviceId;
  unsigned long _pollMs, _lastPoll = 0;
  bool _registered = false;
  CommandHandler _handler = nullptr;
};

#endif  // MEDO_LINK_H
