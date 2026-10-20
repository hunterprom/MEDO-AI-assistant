#!/usr/bin/env python3
"""MEDO — lite launcher for low-resource machines.

Runs the SAME app as ``main.py`` but tuned to stay light: on a typical laptop
MEDO's own process idles around ~300 MB and a few percent CPU while listening
for the wake word. It does this purely through ``MEDO_*`` environment
overrides — no core files change, so nothing forks from the normal build.

    python medo-lite.py                # lite voice + companion API
    python medo-lite.py --hud          # add the HUD (still light)
    python medo-lite.py --with-vision  # re-enable gesture vision (heavy!)
    python medo-lite.py --once "what time is it"

What lite changes (and why):

    STT       tiny model, int8 compute, greedy (beam=1)   ← biggest CPU saver
              in transcription; tiny+int8 is ~4x lighter than small.
    Vision    OFF by default                              ← the MediaPipe
              sidecar (separate process, 15 fps) is the real CPU/RAM hog;
              turning it off is what makes the low targets reachable.
              Pass --with-vision to accept the cost.
    Embeddings OFF (memory.embed_model="")                ← disables the local
              embedding model + its Ollama calls AND the background document
              RAG indexer. Facts still work (newest-N instead of semantic).
    LLM       smallest installed Ollama model, num_ctx 2048, keep_alive 5m
              ← a big model is optional; lite prefers a 1-3B if you have one.
    HUD       OFF unless --hud.

Everything else — wake word, fast-path skills, LLM chat, TTS, memory, the
companion API + watch — is preserved. Heavy extras (gesture pointer, semantic
document search) are the deliberate trade for the footprint.
"""

from __future__ import annotations

import os
import sys
import urllib.request


def _smallest_installed_ollama_model(host: str, cap_b: float = 4.0) -> str | None:
    """Pick the smallest installed Ollama model at/under ``cap_b`` billion params.

    Keeps the LLM light without pulling anything. Returns None when Ollama
    isn't reachable or has no models — the app then runs fast-path only.
    """
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=2) as r:
            import json

            models = json.load(r).get("models", [])
    except Exception:
        return None
    if not models:
        return None

    def size_b(m: dict) -> float:
        # Prefer the reported parameter count; fall back to on-disk bytes.
        p = (m.get("details") or {}).get("parameter_size", "")
        try:
            if p.lower().endswith("b"):
                return float(p[:-1])
        except ValueError:
            pass
        return m.get("size", 0) / 1e9  # bytes -> ~GB as a rough proxy

    ranked = sorted(models, key=size_b)
    under = [m for m in ranked if size_b(m) <= cap_b] or ranked
    return under[0].get("name")


def _setdefault_env(overrides: dict[str, str]) -> None:
    """Apply overrides without clobbering anything the user set explicitly."""
    for key, value in overrides.items():
        os.environ.setdefault(key, value)


def main() -> None:
    argv = sys.argv[1:]
    with_vision = "--with-vision" in argv
    argv = [a for a in argv if a != "--with-vision"]

    host = os.environ.get("MEDO_LLM__HOST", "http://localhost:11434")

    # Pin the ONNX/OpenMP runtimes to a single thread. The always-on wake-word
    # model would otherwise fan inference across every core — measured idle
    # load drops from ~8% of a core (multi-thread) to ~1% of the whole system
    # this way, which is the point of lite. Set before onnxruntime imports.
    for k in ("OMP_NUM_THREADS", "ORT_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")

    overrides = {
        # --- STT: the transcription CPU cost, cut ~4x ---
        "MEDO_STT__MODEL": "tiny",
        "MEDO_STT__COMPUTE_TYPE": "int8",
        "MEDO_STT__BEAM_SIZE": "1",
        # --- vision: the sidecar is the real hog; off unless asked ---
        "MEDO_VISION__ENABLED": "true" if with_vision else "false",
        # --- memory: no embedding model => no embed load, no doc RAG indexer ---
        "MEDO_MEMORY__EMBED_MODEL": "",
        "MEDO_MEMORY__MAX_TURNS": "6",
        # --- LLM: keep context small and don't pin the model in VRAM for long ---
        "MEDO_LLM__NUM_CTX": "2048",
        "MEDO_LLM__KEEP_ALIVE": "5m",
    }
    # Prefer a small local model when the provider is (or defaults to) Ollama
    # and the user hasn't pinned one.
    if os.environ.get("MEDO_LLM__PROVIDER", "ollama") == "ollama" \
            and "MEDO_LLM__DEFAULT_MODEL" not in os.environ:
        small = _smallest_installed_ollama_model(host)
        if small:
            overrides["MEDO_LLM__DEFAULT_MODEL"] = small

    _setdefault_env(overrides)

    banner = "MEDO lite — vision %s, STT tiny/int8, embeddings off" % (
        "ON (heavy)" if with_vision else "off")
    print(f"\033[36m[medo-lite]\033[0m {banner}")
    if os.environ.get("MEDO_LLM__DEFAULT_MODEL"):
        print(f"\033[36m[medo-lite]\033[0m LLM: {os.environ['MEDO_LLM__DEFAULT_MODEL']}")

    # Delegate to the real entry point (late import so the env is set first).
    sys.argv = [sys.argv[0], *argv]
    from main import main as medo_main

    medo_main()


if __name__ == "__main__":
    main()
