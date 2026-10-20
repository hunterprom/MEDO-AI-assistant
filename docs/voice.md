# Voice

How MEDO speaks. S1 (engine layer) and S2 (streaming) are done. Barge-in
(S3), prosody (S4) and per-language QA (S5) are separate steps.

## The honest bar

MEDO is **STT → LLM → TTS**. Text is the bottleneck. A true speech-to-speech
model hears *how* you said something and answers in kind; MEDO hears *what* you
said, thinks in text, and speaks the result. It will not equal GPT-style voice
mode, and nothing in this document claims otherwise.

What is achievable, and what these steps build: a natural neural voice per
language, speech that starts almost immediately, interruption that actually
interrupts, and delivery the model can shape. Markedly more human — not the
same thing.

## What changed in S1

MEDO spoke **fifteen of its sixteen languages through Microsoft's cloud**
(`edge-tts`). That was a defensible trade when it was made — the alternative
was shipping sixteen voice models — but it meant MEDO's *voice*, the most
audible thing about it, was the one part that wasn't local.

Those fifteen now run **on-device**.

| | before | after |
|---|---|---|
| non-English synthesis | 1078–1458 ms, cloud | **146–299 ms**, local |
| English synthesis | 116 ms (Piper, local) | 194 ms (same voices, permissive runtime) |
| VRAM | 0 | **0** (measured: delta 0 MiB) |
| offline | English only | **15 of 16 languages** |
| runtime licence | GPL-3.0-or-later | **Apache-2.0** |

That last row matters if MEDO is ever sold. The installed `piper-tts` package
is GPL-3.0-or-later and MEDO called it directly. `sherpa-onnx` is Apache-2.0
and plays the *same* Piper `.onnx` voices, so the exposure goes away without
losing the voices.

## Architecture

Nothing calls a TTS engine directly — the rule the LLM layer already follows
for brains. `voice/providers.py`:

```
VoiceBox  ── first provider that can speak the language wins
  ├─ SherpaProvider     local, CPU. Piper (VITS) + Kokoro + mimic3. 15 langs.
  ├─ EdgeProvider       cloud. ONLY for a language with no local voice.
  └─ LegacyPiperProvider  piper-tts, English only, GPL. Safety net.
```

A provider that fails or returns silence hands on to the next. When **nothing**
can speak a language the caller gets empty audio and must *show* the reply —
never voice it in another language's voice. That is not a nicety: handing
Japanese to an English voice is the documented "reads it out character by
character" bug.

## The voice map

`voice/voices.py`. Every entry was read from the sherpa-onnx release asset list
and, for multi-speaker models, from the ONNX metadata itself.

| lang | engine | voice | tier |
|---|---|---|---|
| en | **Kokoro** | `bf_emma` (id 21), `lang="en"` | **high** |
| de | Piper | `de_DE-thorsten-medium` | medium |
| fr | Piper | `fr_FR-siwis-medium` | medium |
| es | Piper | `es_ES-miro-high` | **high** |
| it | Piper | `it_IT-paola-medium` | medium |
| pt | Piper | `pt_PT-miro-high` | **high** |
| nl | Piper | `nl_NL-ronnie-medium` | medium |
| pl | Piper | `pl_PL-darkman-medium` | medium |
| ru | Piper | `ru_RU-irina-medium` | medium |
| tr | Piper | `tr_TR-fahrettin-medium` | medium |
| el | Piper | `el_GR-rapunzelina-low` | **low** — the only Greek voice Piper has |
| zh | Piper | `zh_CN-huayan-medium` | medium |
| hi | Piper | `hi_IN-priyamvada-medium` | medium |
| ja | Kokoro v1_0 | `jf_alpha` (id 37), `lang="ja"` | medium |
| ko | mimic3 | `ko_KO-kss_low` | **low** — the only Korean TTS in the catalogue |
| mk | edge-tts | `mk-MK-MarijaNeural` | **cloud** — see below |

Tiers are honest, not marketing: Greek and Korean genuinely only have weak
voices available, and the HUD and `--demo` surface that rather than letting
those languages sound mysteriously worse than their neighbours.

### Three things the build got wrong first

Worth recording, because each was found by *using* the map rather than reading
it, and each would have shipped as "that language sounds a bit off":

1. **`kokoro-multi-lang-v1_1` was chosen for Japanese** on a write-up saying
   Kokoro covers ja/ko. Its own metadata: *"Kokoro v1.1-zh … supporting
   English, Chinese"*, 103 speakers, all English or Chinese. v1_0 is the
   nine-language one. Check the model, not the summary.
2. **Kokoro without an explicit `lang` phonemizes everything as en-US.**
   Japanese came out **12.9 s** for a sentence that takes **4.2 s** — the
   wrong-language bug, just quiet enough to miss.
3. **Voices arrive at wildly different levels.** Same sentence: `es_ES` peaked
   at 3870, `tr_TR` at 32746. An eight-fold swing reads as MEDO being broken,
   so the provider RMS-normalises with a peak ceiling.

German is `thorsten-medium`, not `-high`: measured 181 ms against 966 ms for
identical audio length. Not a rule about tiers — `es`/`pt` at `miro-high` run
at 179/228 ms and stay high; that one model is simply heavy.

## "It still sounds robotic"

The first cut of this table put English on Piper's `en_US-lessac-medium`, and
the honest answer to that feedback is that **S1 and S2 optimised the wrong
axis**. Coverage (cloud → local) and latency (dead air 252 ms → 7 ms) don't
touch timbre. Piper is a small 2021-era VITS model chosen for being tiny and
fast; flat delivery is exactly what it trades away. Streaming a robotic voice
sooner just gets you robotic sooner.

English is now **Kokoro `bf_emma`**. What it costs, measured on the same 6.8 s
sentence:

| | Piper lessac | Kokoro |
|---|---|---|
| synthesis | **410 ms** | 3300 ms |
| RTF | 0.061 | 0.48 |
| VRAM | 0 | 0 (still CPU) |
| disk | 63 MB | 325 MB |

~8× slower, still 2× faster than real time, so streaming stays ahead of
playback. About a second more before the first sentence starts.

**Kokoro covers 8 of the 16** — en, zh, ja, es, fr, it, pt, hi — and nothing
for de, nl, pl, ru, tr, el, ko, mk. So the eight will sound better than the
seven on Piper. That inconsistency is real and is preferable to all sixteen
sounding flat. Switching another language is one line in `tts.voices`; hear it
first with `--demo`.

## Choosing voices without touching code

`config.yaml` → `tts.voices`. Anything not listed keeps the built-in choice.

```yaml
tts:
  voices:
    en: {engine: kokoro, voice: bf_emma}
    de: {engine: piper,  voice: de_DE-thorsten-high}
    mk: {engine: edge,   voice: mk-MK-AleksandarNeural}
```

Kokoro speakers are named, not numbered — the prefix is language + gender:
`af_`/`am_` American, `bf_`/`bm_` British, `ef_`/`em_` Spanish, `ff_` French,
`hf_`/`hm_` Hindi, `if_`/`im_` Italian, `jf_`/`jm_` Japanese, `pf_`/`pm_`
Brazilian Portuguese, `zf_`/`zm_` Chinese. The full 53 are in
`voice/voices.KOKORO_SPEAKERS`.

A malformed entry is skipped with a warning and that language keeps its
built-in voice — one typo never costs the whole voice stack. `tts.speed` is
honoured too; the new path was silently ignoring it.

### Hearing them

```
python -m voice.tts --demo en                              # play it
python -m voice.tts --demo en --voice kokoro:af_heart      # try another
python -m voice.tts --demo all --write voice-samples       # write .wav files
```

One neutral sentence per language, identical content everywhere, so voices are
compared on delivery rather than on what they happen to be saying.

**A trap worth recording:** `en-gb` is *not* a valid espeak voice in the
shipped `espeak-ng-data` — espeak's base `en` **is** British. Asking for
`en-gb` fails the entire synthesis with "Failed to set eSpeak-ng voice", which
surfaces as an assistant that simply says nothing. Every code in
`_KOKORO_LANGS` was checked against `lang/*/` on disk, and a test keeps them
there.

## Macedonian — the honest part

**No permissively-licensed local neural Macedonian voice exists.** Verified,
not assumed: not in Piper's 35 languages, not in either Kokoro, not in
Chatterbox Multilingual's 23 languages or its 6 language packs. Meta's MMS
covers vast numbers of languages but is **CC-BY-NC-4.0**, so it cannot ship in
something sold.

So mk stays on the cloud voice — **the single declared exception**, behind
`tts.allow_cloud`. Everything else ignores that flag, and a test asserts mk is
the only language on it, so the exception cannot quietly grow.

Two alternatives, both explicit choices:

- `tts.offline_fallback_voices: true` — Macedonian read by the **Serbian**
  Piper voice. Fully offline, understandable, and audibly Serbian. Off by
  default. (Making this automatic was a bug: with the stand-in returned
  whenever the cloud was refused, the local provider answered "yes, I speak
  Macedonian" and would have used the Serbian voice with nobody choosing it.)
- `allow_cloud: false` and no fallback — mk replies are **shown, not spoken**.

### The path to an excellent mk voice

This is the flagship follow-up, and it is a real project, not a weekend.

**Common Voice MK is the wrong dataset.** Common Voice is ASR data: many
speakers, phone microphones, uneven noise. TTS fine-tuning wants the opposite —
one speaker, one microphone, one room, consistent delivery.

Runbook stub:

1. **Record 5–20 hours, one speaker.** Phonetically varied Macedonian prose,
   quiet room, fixed mic position, 22.05 kHz+ mono. Consistency beats volume:
   5 clean hours outperform 20 uneven ones.
2. **Transcribe and align** to Piper's training format (LJSpeech layout:
   `wavs/` + `metadata.csv`).
3. **Fine-tune from a related checkpoint** rather than from scratch — the
   Serbian or Russian Piper voice shares most of the Cyrillic phoneme
   inventory, so it converges far faster.
4. **Export to ONNX**, drop it in `models/tts/`, add the entry to
   `voice/voices.py`, and delete the cloud exception.

Korean is the same problem one tier down and is the second candidate.

## Configuration

Everything is in `config.yaml` under `tts:` — `local_voices`, `voices_dir`,
`auto_download`, `max_loaded_voices`, `num_threads`, `allow_cloud`,
`offline_fallback_voices`. See the comments there.

Voices download on first use (~30–70 MB each) and are cached. `espeak-ng-data`
ships inside every Piper archive and is identical in all of them, so it is
hoisted to one shared copy — 18 MB total instead of 18 MB × 13.

## Measurements

RTX 3060 12 GB (already sharing the GPU with Ollama and vision), CPU synthesis,
warm model, one sentence:

```
en  194 ms  RTF 0.067      pl  207 ms  RTF 0.066
de 1033 ms  RTF 0.406 (*)  ru  225 ms  RTF 0.078
fr  198 ms  RTF 0.070      tr  146 ms  RTF 0.067
es  179 ms  RTF 0.070      el  148 ms  RTF 0.056
it  183 ms  RTF 0.063      zh  193 ms  RTF 0.070
pt  228 ms  RTF 0.070      hi  299 ms  RTF 0.073
nl  282 ms  RTF 0.089      ja 2777 ms  RTF 0.659 (Kokoro is ~10x Piper)
                           ko  255 ms  RTF 0.084
```

(*) measured on `thorsten-high`, since replaced by `thorsten-medium` at 181 ms.

**VRAM delta: 0 MiB.** All of it is CPU.

Japanese at 2777 ms is the one language where local is *slower* than the cloud
was (~1078 ms). Kokoro is an order of magnitude heavier than Piper. It stays
because the alternative is no Japanese voice at all, and S2's streaming hides
most of it.

## S2 — streaming synthesis

Speech starts before the reply is finished. Sentence-streaming already existed
(`voice/loop.py`), so S2 was measuring it and fixing what the measurement
found. Two defects, both real:

### Three languages never streamed at all

`drain_sentences` looked for `.!?` **followed by whitespace**. Chinese and
Japanese end sentences with the full-width `。` and put no space after it;
Hindi uses the danda `।`. So **zh, ja and hi drained zero sentences** and
waited for the entire reply before a word was spoken.

The pattern now has three branches — alphabetic (space required, so "Dr." and
"3.5" still don't split), CJK (the mark itself is the boundary), and Devanagari
— and length is measured in `speech_weight`, not characters, because 16 Chinese
characters and 44 English ones are both about 2.8 s of speech.

### Every sentence boundary was a silent gap

The speaker synthesized a sentence, played it, and only *then* started
synthesizing the next — so each boundary cost a full synthesis of dead air.
It now runs two stages with a bounded queue, synthesizing one sentence ahead
of playback.

### Measured

Simulated LLM at ~160 chars/s, real segmentation, real synthesis, playback
slept for the audio's own duration.

| lang | TTFA before | TTFA after | dead air before | after |
|---|---|---|---|---|
| en | 445 ms | 459 ms | 252 ms | **7 ms** |
| de | 465 ms | 466 ms | 200 ms | **9 ms** |
| ru | 486 ms | 487 ms | 245 ms | **14 ms** |
| zh | **680 ms** (no streaming) | **274 ms** | 275 ms | **15 ms** |
| ja | **5508 ms** (no streaming) | **2622 ms** | 3058 ms | **8 ms** |
| hi | **1173 ms** (no streaming) | **592 ms** | 271 ms | **13 ms** |

Two honest notes. **Pipelining does not improve TTFA** — nothing can
synthesize the first sentence before it exists; the TTFA gains are entirely
from the segmentation fix, and en/de/ru (which already streamed) see none.
And **Japanese is still 2.6 s** because Kokoro is ~10x heavier than Piper;
streaming hides the gaps but not the first synthesis.

## Interview note

> MEDO's voice is a swappable multilingual neural TTS with streaming synthesis,
> instant barge-in, and LLM-driven prosody — natural and dynamic across sixteen
> languages, running entirely on-device at zero VRAM cost, with an honest path
> to an excellent Macedonian voice.
