# MEDO — phone app (on-device)

A standalone Android voice assistant. Unlike `../watch/` (a thin client that
offloads to the desktop MEDO over the LAN), **this app does the work itself on
the phone** — speech in, skill routing, and speech out all run locally. No PC,
no account, no gestures.

## How it works

```
tap-to-talk ─► Android speech recognizer (on-device) ─► Intent Router
                                                            │
              ┌─────────────────────────────────────────────┤
              │ LINKED?  route the whole utterance to the desktop MEDO (medo_link)
              │ else FAST: a local skill matched                          ─► Android TTS
              │      else → the chosen chat backend (below)                  (on-device)
              └─────────────────────────────────────────────┘
```

- **Fast-path skills run 100% on the phone.** Only the ones that *fetch* data
  (weather, web search, news) or the chosen chat backend use the network.
- **AI model is your pick** (Settings → *AI model*), for open-ended questions no
  skill caught. The answer comes back **inline in the chat**:
  - **On-device** — fully local; replies "add a model" instead of guessing.
  - **AI model** — choose **ChatGPT, Claude, DeepSeek, Gemini, or Groq** (or a
    Custom OpenAI-compatible endpoint) and paste that provider's API key once.
    MEDO calls the model's API and speaks/shows the reply here. Keys are stored
    per provider so you can keep several set up and switch between them.

  > Note: the *installed* ChatGPT/Claude/… **apps** can't be queried by another
  > app and made to hand their answer back — Android has no such API. Getting the
  > reply inside MEDO is done by calling the provider's API with a key, which is
  > what this option does.
- **Connect to MEDO** (Settings → *Connect to MEDO*) links to the desktop MEDO
  over Wi-Fi using the **exact same pairing as the watch** (UDP discovery →
  6-digit code → bearer token), shared through the `medo_link` package. Toggle
  *Route everything to MEDO* and every request runs on the PC's full brain.
- Your **notes, remembered facts, and settings never leave the device**
  (`shared_preferences`).

## Shared connection package

The MEDO link (discovery + pairing + `/ask`) lives in `../packages/medo_link/`
— a pure-Dart package (`http` + `dart:io`, no Flutter) consumed by both the
watch and the phone, so the connection logic exists once.

## Skills (all on-device)

| Group | Examples |
|-------|----------|
| Time & date | "what time is it", "what day is it" |
| Timers & reminders | "set a timer for 5 minutes", "remind me in 10 to stretch", "wake me in an hour", "cancel timers" |
| Notes | "take a note buy milk", "read my notes", "clear notes" |
| Memory | "remember that my locker code is 42", "what do you remember", "forget the locker code" |
| Weather | "weather in London", "do I need a jacket", "will it rain" |
| Web search | "search for the tallest mountain", "who is Ada Lovelace", "what is a black hole" |
| Camera / vision | "what do you see", "look at this", "take a picture", "read this label" — opens the camera, then the AI model describes the shot in the chat |
| News | "the news", "world news", "what's happening" |
| Calculator & convert | "what is 12 x 8 + 3", "convert 10 km to miles", "20 celsius to fahrenheit", "50 usd to eur" |
| Phone actions | "turn on the flashlight", "volume up", "set volume to 30", "open Spotify", "call 070123456", "text 070123456 saying on my way" |

Desktop-only MEDO skills (type, screenshot, pointer, window management,
brightness, power) don't apply on a phone and are intentionally absent.

## Build & run

Uses the same toolchain as the watch app (Flutter 3.19.6; Gradle 8.5 / AGP
8.1.4 for Java 21):

```
cd phone
flutter pub get            # offline works if the deps are cached
flutter analyze            # clean
flutter test               # skill tests
flutter build apk          # or: flutter run  (device/emulator attached)
```

Permissions requested on first use: **microphone** (speech) and internet. The
camera is used through the system camera app (`ACTION_IMAGE_CAPTURE`), so MEDO
itself doesn't hold the CAMERA permission — the torch uses `setTorchMode`, which
needs none.

### Vision / "what do you see"

Saying *"what do you see"* (or tapping the camera button) opens the camera,
downscales the shot to ≤1024 px, and sends it to your selected model's vision
endpoint; the description comes back in the chat. Vision-capable providers:
**ChatGPT** (`gpt-4o-mini`), **Claude** (`claude-3-5-sonnet`), **Gemini**
(`gemini-1.5-flash`), and **Groq** (Llama-4 Scout). DeepSeek's API has no vision
model, so pick one of the others for camera use. MEDO checks the model can see
*before* opening the camera.

## Native bridge

Flashlight, volume, open-app, call, and SMS have no pure-Dart API, so they go
through one Kotlin `MethodChannel` (`medo/native`) in
`android/.../MainActivity.kt` — no extra plugins. Every call is best-effort and
degrades to a spoken "I couldn't…" rather than crashing.

## Known limitations

- **Timers fire while the app is running.** They use an in-app timer + spoken
  announcement; a true background alarm that survives the app being closed
  needs the `flutter_local_notifications` plugin (a follow-up — it wasn't in
  the offline package cache this was built against).
- **Currency conversion is an offline estimate** from a small built-in rate
  table (no live FX), so it says "approximate".
- **Wake word is not included** by design — tap-to-talk only (lighter on
  battery, no always-on mic service).
- Speech recognition quality is whatever the device's Android engine provides.
