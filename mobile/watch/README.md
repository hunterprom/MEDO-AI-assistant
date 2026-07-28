# Jarvis Watch — Wear OS companion

Speak to your watch; Jarvis answers. A Flutter app for Wear OS (Galaxy
Watch4 and newer, incl. Watch8 Classic) that does speech-to-text on the
watch, sends the transcript to the Jarvis v2 companion API over Wi-Fi,
then shows the reply and speaks it aloud.

```
watch mic ──STT──▶ text ──HTTP POST /ask──▶ Jarvis (Mac/Windows)
watch TTS ◀── reply text ◀───────────────── Intent Router
```

## Run the server

On the computer that runs Jarvis:

```sh
python main.py --serve        # REPL + companion API on port 8710
```

(Or set `remote.enabled: true` in `config.yaml` to serve always.)
The watch and the computer must be on the same network. No auth — LAN only,
never port-forward it.

## Run on the emulator

```sh
# Wear OS 5 AVD created for this project:
~/Library/Android/sdk/emulator/emulator -avd Wear_OS_Round &
cd watch && flutter run
```

The default server address `10.0.2.2:8710` already points at the host
machine. The emulator has no real speech service — use the on-screen
keyboard button instead of the mic.

## Run on the real watch

1. On the watch: Settings → About watch → Software → tap **Software
   version** 5× to enable developer options, then enable **ADB debugging**
   and **Wireless debugging** (watch and computer on the same Wi-Fi).
2. Pair/connect: `adb pair <watch-ip>:<pair-port>` (code shown on watch),
   then `adb connect <watch-ip>:<port>`.
3. `cd watch && flutter run` (or `flutter install`).
4. In the app: gear icon → set your computer's LAN address, e.g.
   `192.168.1.20:8710` → **Save & test** → look for "Connected to Jarvis ✓".
5. Tap the mic and talk.

Note: the on-watch speech recognizer is the system one (Google/Samsung),
so the watch→text step may use their voice services; everything from the
text onward stays 100 % local.

## Gesture activation (hands-free)

With **Gesture activation** on (gear icon → "Double wrist-flick to talk",
enabled by default), a quick **double flick of the wrist** starts listening
— exactly as if you had tapped the mic button — with a haptic buzz as
confirmation. Detection only runs while the app is idle; it pauses while
MEDO is already listening, thinking or speaking, and resumes afterwards.

The **sensitivity slider** (8–25 m/s²) sets how hard each flick must be:
lower values trigger on a gentle flick (but risk false triggers while
walking or gesturing), higher values need a deliberate snap. The default
of 14 m/s² (~1.4 g of gravity-free jerk) sits comfortably above everyday
arm movement. Both settings persist across launches.

### Why wrist-flick and not a true finger pinch?

Wear OS (including One UI Watch on the Galaxy Watch8 Classic) detects the
pinch/double-pinch "Universal gestures" inside its **accessibility
service** and does not expose those events to third-party apps — there is
no API a Flutter app can subscribe to. What apps *can* read is the raw
accelerometer, so the app detects a double wrist-flick instead: two
acceleration-magnitude spikes above the threshold within 700 ms, with a
2 s cooldown after firing (see `lib/gesture_trigger.dart` for the math).

Samsung's system-level **Universal gestures** (Settings → Accessibility →
Interaction and dexterity → Universal gestures) can still *complement*
this: configure the system pinch/double-pinch to an action such as
"open selected item" or a shortcut that launches this app, then use the
in-app wrist-flick to start listening. The two operate at different
layers — pinch gets you into the app hands-free, flick talks to MEDO
hands-free.

## Layout

- `lib/jarvis_client.dart` — HTTP client for `/ping` and `/ask`
- `lib/home_screen.dart`   — tap-to-talk UI (mirrors Jarvis's IDLE→LISTENING→THINKING→SPEAKING states)
- `lib/gesture_trigger.dart` — double wrist-flick detector (accelerometer)
- `lib/settings_screen.dart` — server address + connection test + gesture toggle/sensitivity
- `lib/settings.dart`      — persisted settings
