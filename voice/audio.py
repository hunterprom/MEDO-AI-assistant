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


def list_input_devices() -> list[dict]:
    """``[{"index", "name"}]`` for every input-capable device (``[]`` on error).

    Used by the HUD microphone picker so a user can choose the mic the wake word
    listens on without editing config. Never raises.
    """
    try:
        import sounddevice as sd

        return [
            {"index": i, "name": str(dev["name"])}
            for i, dev in enumerate(sd.query_devices())
            if dev.get("max_input_channels", 0) > 0
        ]
    except Exception:
        logger.debug("could not list input devices", exc_info=True)
        return []


def resolve_input_device(device: int | str | None):
    """Turn a device name-substring into its index; pass ints/None through.

    Lets ``audio.input_device`` be a stable name like ``"FHD Webcam"`` instead of
    a PortAudio index that can change across reboots. Raises with a helpful hint
    when nothing matches so a typo doesn't silently fall back to a dead default.
    """
    if not isinstance(device, str):
        return device
    import sounddevice as sd

    want = device.strip().lower()
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and want in str(dev["name"]).lower():
            return i
    raise RuntimeError(
        f"no input device name contains {device!r}; list them with "
        f"'python -m voice.wakeword'"
    )


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
        device: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = device
        self._stream = None  # sounddevice.InputStream, created on open()

    def open(self) -> "Microphone":
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            channels=1,
            dtype="int16",
            device=resolve_input_device(self.device),
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
