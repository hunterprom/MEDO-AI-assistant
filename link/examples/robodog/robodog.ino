// robodog.ino — STUB reference sketch for the MEDO Link robot dog.
//
// This shows exactly where your existing servo/gait control code plugs into
// the MedoLink callbacks; the movement functions are left as TODOs on
// purpose (Matej wires the real firmware). walk_forward and demo_mode carry
// requires_confirmation in the manifest, so MEDO asks a spoken yes/no
// (English or Macedonian) BEFORE the command ever reaches this sketch.
//
// Libraries: ESP32 Arduino core, ArduinoJson 7.x. Copy MedoLink.h next to
// this sketch (or add link/clients/arduino to your library path).

#include <WiFi.h>
#include "MedoLink.h"

const char* WIFI_SSID = "your-network";
const char* WIFI_PASS = "your-password";
const char* MEDO_HOST = "192.168.1.20:8710";   // machine running MEDO --serve
const char* MEDO_TOKEN = "paste remote.token from secrets.local.yaml";

// Keep this in sync with link/examples/robodog-manifest.json.
const char* MANIFEST = R"json({
  "device_id": "robodog",
  "name": "the robot dog",
  "transport": "http_poll",
  "capabilities": [
    {"name": "sit", "description": "Make the robot dog sit down."},
    {"name": "stand", "description": "Make the robot dog stand up."},
    {"name": "walk_forward", "description": "Walk forward N steps. PHYSICALLY MOVES.",
     "params": {"steps": {"type": "integer", "description": "steps, 1-20"}},
     "requires_confirmation": true},
    {"name": "turn", "description": "Turn in place.",
     "params": {"direction": {"type": "string"}, "degrees": {"type": "integer"}}},
    {"name": "demo_mode", "description": "Full movement demo. PHYSICALLY MOVES.",
     "requires_confirmation": true}
  ]
})json";

MedoLink link(MEDO_HOST, MEDO_TOKEN, MANIFEST);

bool handleCommand(const char* cap, JsonObjectConst params, String& msg) {
  if (strcmp(cap, "sit") == 0) {
    // TODO(Matej): dogSit();
    msg = "Sitting.";
    return true;
  }
  if (strcmp(cap, "stand") == 0) {
    // TODO(Matej): dogStand();
    msg = "Standing.";
    return true;
  }
  if (strcmp(cap, "walk_forward") == 0) {
    int steps = constrain((int)(params["steps"] | 3), 1, 20);
    // TODO(Matej): dogWalk(steps);
    msg = String("Walked ") + steps + " steps.";
    return true;
  }
  if (strcmp(cap, "turn") == 0) {
    const char* dir = params["direction"] | "left";
    int degrees = constrain((int)(params["degrees"] | 90), 15, 180);
    // TODO(Matej): dogTurn(dir, degrees);
    msg = String("Turned ") + dir + " " + degrees + " degrees.";
    return true;
  }
  if (strcmp(cap, "demo_mode") == 0) {
    // TODO(Matej): dogDemo();
    msg = "Demo complete.";
    return true;
  }
  msg = "unknown capability";
  return false;
}

void setup() {
  Serial.begin(115200);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) delay(200);
  link.onCommand(handleCommand);
  while (!link.begin()) {           // registers the manifest with MEDO
    Serial.println("MEDO not reachable, retrying...");
    delay(2000);
  }
  Serial.println("Registered with MEDO.");
}

void loop() {
  link.poll();
  delay(100);
}
