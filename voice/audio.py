"""Microphone and speaker I/O built on ``sounddevice``.

Everything here is blocking (PortAudio is a blocking C API); the async voice loop
in ``main.py`` drives it via ``asyncio.to_thread`` so the event loop — and the
companion API — stay responsive.

Audio format across the pipeline: 16 kHz, mono, 16-bit PCM. That's what
openWakeWord and faster-whisper both expect, so no resampling is needed until TTS
playback (Piper voices have their own sample rate).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# 80 ms at 16 kHz. openWakeWord recommends feeding it 1280-sample chunks.
FRAME_SAMPLES = 1280


# Host API whose devices sounddevice can't read with the blocking API — a name
# match landing on one of these would "resolve" and then fail to open.
_UNUSABLE_HOSTAPI = "Windows WDM-KS"


def _usable_inputs() -> list[tuple[int, str, str]]:
    """(index, name, hostapi_name) for every input device we can actually read."""
    import sounddevice as sd

    apis = [a["name"] for a in sd.query_hostapis()]
    out = []
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        api = apis[dev["hostapi"]] if dev["hostapi"] < len(apis) else "?"
        if api == _UNUSABLE_HOSTAPI:
            continue
        out.append((i, str(dev["name"]), api))
    return out


# Host APIs ranked by how reliably we can open them at 16 kHz mono, best first.
# MME and DirectSound resample to any rate; WASAPI often refuses a non-native
# rate (PaErrorCode -9997, verified on this machine); WDM-KS is dropped entirely
# by _usable_inputs. A device is used via its best-ranked host API.
_HOSTAPI_RANK = {"MME": 0, "Windows DirectSound": 1, "Windows WASAPI": 2}


def _device_key(name: str) -> str:
    """Group a mic's per-host-API duplicates. MME truncates device names to 31
    chars, so that prefix is what the copies of one physical mic share."""
    return name[:31].strip().lower()


def _dedup_inputs(usable: list[tuple[int, str, str]]) -> list[dict]:
    """Collapse the per-host-API duplicates in ``usable`` to one entry per
    physical mic: the index on the most-openable host API, paired with the
    SHORTEST name seen for it. The shortest is the truncation-safe form (a
    substring of every longer copy), so a name persisted from the picker still
    resolves — and resolves to the openable endpoint — later."""
    picked: dict[str, dict] = {}
    for i, name, api in usable:
        key = _device_key(name)
        rank = _HOSTAPI_RANK.get(api, 9)
        cur = picked.get(key)
        if cur is None:
            picked[key] = {"index": i, "name": name, "rank": rank}
            continue
        if rank < cur["rank"]:              # a more-openable endpoint for this mic
            cur["index"], cur["rank"] = i, rank
        if len(name) < len(cur["name"]):    # keep the truncation-safe (shortest) name
            cur["name"] = name
    return [{"index": p["index"], "name": p["name"]}
            for p in sorted(picked.values(), key=lambda p: p["index"])]


def list_input_devices() -> list[dict]:
    """Every physical microphone, across ALL host APIs, listed once (``[]`` on
    error).

    Windows enumerates each endpoint separately per host API (MME, DirectSound,
    WASAPI, WDM-KS): a raw dump shows one mic up to four times, and the WDM-KS
    copies can't be opened at all. We drop WDM-KS (see ``_usable_inputs``) and
    collapse the rest to one entry per mic, keeping the index on the host API
    most likely to open at 16 kHz. So a mic that only appears under WASAPI or
    DirectSound — e.g. a Bluetooth headset's Hands-Free mic — is now detected
    too, where the old MME-only picker hid it. Never raises.
    """
    try:
        return _dedup_inputs(_usable_inputs())
    except Exception:
        logger.debug("could not list input devices", exc_info=True)
        return []


def resolve_input_device(device):
    """Resolve ``audio.input_device`` to a PortAudio index (or None = default).

    Accepts an int index (passed through), ``None`` (system default), a
    case-insensitive name substring like ``"FHD Webcam"`` (stable across the
    re-indexing PortAudio does on every reboot), or a **priority list** of any
    of those — e.g. ``["A25", "FHD Webcam"]`` means "the headphones' mic
    whenever they're connected, the webcam otherwise". Only endpoints on host
    APIs we can actually read are matched (WDM-KS pins enumerate even for
    disconnected devices and don't support blocking reads). Raises with a
    helpful hint when nothing usable matches so a typo doesn't silently fall
    back to a dead default.
    """
    if isinstance(device, (list, tuple)):
        for entry in device:
            try:
                return resolve_input_device(entry)
            except RuntimeError:
                continue
        raise RuntimeError(
            f"none of the preferred input devices {list(device)!r} is currently "
            f"available; pick one in the HUD CONFIG tab or list them with "
            f"'python -m voice.wakeword'"
        )
    if not isinstance(device, str):
        return device
    want = device.strip().lower()
    matches = [(i, name, api) for i, name, api in _usable_inputs()
               if want in name.lower()]
    if matches:
        # One mic matches on several host APIs; open it via the one we can
        # actually read at 16 kHz (MME > DirectSound > WASAPI), not whichever
        # PortAudio happened to enumerate first.
        matches.sort(key=lambda m: _HOSTAPI_RANK.get(m[2], 9))
        return matches[0][0]
    raise RuntimeError(
        f"no usable input device name contains {device!r}; pick one in the HUD "
        f"CONFIG tab or list them with 'python -m voice.wakeword'"
    )


def normalize_peak(
    audio: np.ndarray,
    target: float = 0.6,
    min_peak: float = 1e-4,
    max_gain: float = 25.0,
) -> np.ndarray:
    """Scale a float waveform so its peak sits at ``target`` (quiet-mic rescue).

    Webcam/onboard mics often record speech peaking at 0.01–0.05, which hurts
    Whisper accuracy. Boosting the clip to a healthy peak costs nothing when the
    audio is already loud (it can also attenuate). ``min_peak`` skips silent
    clips entirely, and ``max_gain`` (~+28 dB) caps the boost so a clip that
    barely crossed the recording gate — a cough, a chair squeak — can't be
    amplified into loud garbage that Whisper hallucinates words from.
    """
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < min_peak:
        return audio
    return (audio * min(target / peak, max_gain)).astype(np.float32)


def frame_rms(frame: np.ndarray) -> float:
    """Root-mean-square level of an int16 frame, normalized to 0.0–1.0."""
    if frame.size == 0:
        return 0.0
    # float64 math avoids int overflow when squaring int16 values.
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) / 32768.0)


class Microphone:
    """A mono 16-bit input stream you read one frame at a time.

    Use as a context manager so the PortAudio stream is always closed::

        with Microphone(16000) as mic:
            frame = mic.read_frame()
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_samples: int = FRAME_SAMPLES,
        device: int | str | list | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = device
        # The concrete PortAudio index open() resolved to (None = default).
        # The voice loop compares this against a fresh resolve to hot-swap when
        # a higher-priority device (e.g. headphones) appears or disappears.
        self.resolved_index: int | None = None
        self._stream = None  # sounddevice.InputStream, created on open()

    def open(self) -> "Microphone":
        import sounddevice as sd

        self.resolved_index = resolve_input_device(self.device)
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            channels=1,
            dtype="int16",
            device=self.resolved_index,
        )
        self._stream.start()
        return self

    def read_frame(self) -> np.ndarray:
        """Read one frame as a 1-D int16 array (blocks until it's available)."""
        if self._stream is None:
            raise RuntimeError("microphone not open")
        data, overflowed = self._stream.read(self.frame_samples)
        if overflowed:
            logger.debug("microphone input overflow (frame dropped upstream)")
        return data.reshape(-1)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "Microphone":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()


def record_until_silence(
    mic: "Microphone",
    *,
    silence_threshold: float,
    silence_duration_s: float,
    max_seconds: float = 12.0,
    start_timeout_s: float = 6.0,
    preroll_s: float = 0.3,
) -> np.ndarray:
    """Capture speech from ``mic`` until the user stops talking.

    Waits for speech to begin (energy above an adaptive threshold), then records
    until ``silence_duration_s`` of continuous quiet, a ``max_seconds`` cap, or —
    if the user never speaks — ``start_timeout_s``. A ``preroll_s`` buffer of audio
    *before* onset is prepended so the first word isn't clipped (this noticeably
    improves transcription).

    The first few frames (before any speech is expected) are used to measure the
    ambient noise floor, and onset/offset detection then uses
    ``max(silence_threshold, ambient * 2.5)`` — so detection keeps working in a
    noisy room where the fixed ``silence_threshold`` would sit below the hiss.
    Those calibration frames still land in the pre-roll buffer, so nothing is
    lost if the user starts talking right away. Returns a mono 16 kHz float32
    waveform in [-1, 1], ready for STT (empty array if nothing was said).
    """
    from collections import deque

    frames_per_second = mic.sample_rate / mic.frame_samples
    quiet_needed = max(1, int(silence_duration_s * frames_per_second))
    max_frames = int(max_seconds * frames_per_second)
    start_deadline = int(start_timeout_s * frames_per_second)
    preroll_frames = max(0, int(preroll_s * frames_per_second))
    ambient_frames = 4  # ~320 ms of pre-speech audio to gauge the noise floor

    captured: list[np.ndarray] = []
    preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
    ambient_levels: list[float] = []
    effective_threshold = silence_threshold
    started = False
    quiet_run = 0

    for i in range(max_frames):
        frame = mic.read_frame()
        level = frame_rms(frame)
        if not started and len(ambient_levels) < ambient_frames:
            # Calibration: assume the user hasn't spoken yet (recording starts
            # right after the wake word, so these frames are ambient room noise).
            ambient_levels.append(level)
            preroll.append(frame)
            if len(ambient_levels) == ambient_frames:
                ambient = float(np.mean(ambient_levels))
                effective_threshold = max(silence_threshold, ambient * 2.5)
                if effective_threshold > silence_threshold:
                    logger.debug(
                        "ambient noise %.4f raised VAD threshold to %.4f",
                        ambient,
                        effective_threshold,
                    )
            if i >= start_deadline:
                break  # timeout shorter than the calibration window
            continue
        loud = level >= effective_threshold
        if not started:
            if loud:
                started = True
                captured.extend(preroll)   # keep the run-up so nothing is clipped
                captured.append(frame)
            else:
                preroll.append(frame)
                if i >= start_deadline:
                    break  # user never started speaking
        else:
            captured.append(frame)
            quiet_run = 0 if loud else quiet_run + 1
            if quiet_run >= quiet_needed:
                break

    if not captured:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(captured).astype(np.float32) / 32768.0


class Speaker:
    """Plays float or int16 waveforms through the default output device."""

    def __init__(self, device: int | None = None) -> None:
        self.device = device

    def play(self, samples: np.ndarray, sample_rate: int, blocking: bool = True) -> None:
        """Play a mono waveform. Blocks until playback finishes by default."""
        import sounddevice as sd

        sd.play(samples, samplerate=sample_rate, device=self.device)
        if blocking:
            sd.wait()

    def stop(self) -> None:
        import sounddevice as sd

        sd.stop()
