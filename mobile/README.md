# MEDO — mobile

The phone and watch companions to the desktop MEDO, plus the shared networking
package they both use. Everything Flutter/Dart lives here.

```
mobile/
├── phone/               # Android phone app — on-device assistant that can also
│                        #   link to the desktop MEDO over the LAN
├── watch/               # Wear OS app — a thin client that offloads to desktop MEDO
└── packages/
    └── medo_link/       # shared pure-Dart client: LAN discovery + pairing + /ask
```

## The three pieces

- **`phone/`** — a standalone Android voice assistant (tap-to-talk → on-device
  speech + skill routing → TTS). It can also *link* to the desktop MEDO and route
  every request to the PC's full brain. See [`phone/README.md`](phone/README.md).
- **`watch/`** — a Wear OS client: it discovers the desktop MEDO on the Wi-Fi,
  connects, and sends utterances to it. See [`watch/README.md`](watch/README.md).
- **`packages/medo_link/`** — the "connect to MEDO" logic (UDP discovery, pairing,
  the `MedoClient` for `/ping` and `/ask`), factored out so the phone and watch
  share one implementation instead of two.

## Easy connect (no typing)

Both apps connect to the desktop MEDO with **approve-on-PC** pairing: the app
discovers MEDO on the LAN and taps *Connect*; a request appears in the desktop
HUD's **Device fleet** panel, and one **Approve** click hands the app its token.
No codes to read off the screen and type. (The legacy 6-digit-code pairing still
works as a fallback.) The server side is `remote/server.py`; the token is the
companion API's bearer token.

## Build

Each app is a normal Flutter module (Flutter 3.19.6; JDK 17+):

```
cd phone   # or: cd watch
flutter pub get
flutter analyze
flutter test
flutter build apk --release
```

**No local toolchain?** Every push builds both APKs in the cloud — see
[`.github/workflows/mobile-apk.yml`](../.github/workflows/mobile-apk.yml). Grab
`app-release.apk` from the run's **Artifacts** and sideload it to your phone.
