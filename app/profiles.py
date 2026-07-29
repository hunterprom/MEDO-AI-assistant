"""Hardware profiles — pick a model set that actually FITS the user's machine.

The honest rule (approved 2026-07-30): every auto-selected tier gets a brain that
fits its GPU, so nobody is quietly handed a model that offloads to system RAM and
crawls. ``qwen3:30b`` (~18 GB) is reserved for 24 GB+ cards (``max``) or a manual
override — a 12-16 GB card auto-gets ``qwen2.5:14b`` (9 GB), which fits.

Model tags + sizes were verified against ollama.com/library on 2026-07-30
(llama3.2 1b=1.3 GB / 3b=2.0 GB; qwen2.5 7b=4.7 GB / 14b=9.0 GB). The full/max
tiers reuse the models MEDO already runs. This module is pure data + selection —
no detection, no I/O — so tier mapping is exhaustively unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Profile:
    name: str                 # lite | balanced | full | max
    llm: str                  # ollama tag for the main brain
    whisper: str              # faster-whisper size (tiny/base/small/large-v3-turbo)
    vision: Optional[str]     # ollama VLM tag, or None (vision off on lite)
    embed: str                # embedding model (memory + semantic routing)
    min_vram_gb: float        # the VRAM at/above which this tier is auto-picked
    download_gb: float        # approx one-time download total, for the UI
    blurb: str                # plain-language explanation shown to the user

    def ollama_models(self) -> List[str]:
        """Everything the wizard must `ollama pull` for this profile (whisper is
        a separate faster-whisper download, handled by the wizard)."""
        models = [self.llm]
        if self.vision:
            models.append(self.vision)
        models.append(self.embed)
        return models


_EMBED = "nomic-embed-text"
_VLM = "qwen2.5vl:3b"

#: Very weak / CPU-only with little RAM — the smallest brain that still converses.
LITE_MIN = Profile(
    name="lite", llm="llama3.2:1b", whisper="tiny", vision=None, embed=_EMBED,
    min_vram_gb=0.0, download_gb=1.8,
    blurb=("Your computer doesn't have much to spare for AI, so MEDO uses its "
           "smallest brain. It runs fully offline and privately — it's slower "
           "and keeps answers simple, but it works."))

LITE = Profile(
    name="lite", llm="llama3.2:3b", whisper="base", vision=None, embed=_EMBED,
    min_vram_gb=0.0, download_gb=2.5,
    blurb=("Your computer doesn't have a dedicated AI graphics card, so MEDO "
           "uses a lighter brain. It runs fully offline and privately — a bit "
           "slower and simpler than on a gaming PC, but it works well for "
           "everyday things."))

BALANCED = Profile(
    name="balanced", llm="qwen2.5:7b", whisper="small", vision=_VLM, embed=_EMBED,
    min_vram_gb=6.0, download_gb=7.5,
    blurb=("Your graphics card can run a capable brain. MEDO will be snappy for "
           "everyday questions and can see your screen and camera — all offline."))

FULL = Profile(
    name="full", llm="qwen2.5:14b", whisper="large-v3-turbo", vision=_VLM,
    embed=_EMBED, min_vram_gb=12.0, download_gb=14.0,
    blurb=("You've got a strong graphics card — MEDO uses a bigger, smarter "
           "brain that still fits your GPU comfortably, so it stays fast."))

MAX = Profile(
    name="max", llm="qwen3:30b", whisper="large-v3-turbo", vision=_VLM,
    embed=_EMBED, min_vram_gb=24.0, download_gb=23.0,
    blurb=("Your graphics card is powerful enough for MEDO's largest brain. "
           "This is as capable as MEDO gets, running entirely on your machine."))

#: All tiers a user can MANUALLY choose in Settings (both lite variants collapse
#: to one visible 'lite'; qwen3:30b is reachable via 'max' on any machine).
ALL_PROFILES = [LITE, BALANCED, FULL, MAX]
BY_NAME = {"lite": LITE, "balanced": BALANCED, "full": FULL, "max": MAX}


def select_profile(hw) -> Profile:
    """Auto-pick the best-FITTING profile for detected hardware. Fits a brain to
    the GPU; unknown/degenerate hardware falls back to a safe lite default and
    never raises. ``hw`` is duck-typed (see app.hardware.HardwareInfo)."""
    has_gpu = bool(getattr(hw, "has_cuda_gpu", False))
    vram = float(getattr(hw, "vram_gb", 0.0) or 0.0) if has_gpu else 0.0
    ram = float(getattr(hw, "ram_gb", 0.0) or 0.0)

    if vram >= MAX.min_vram_gb:
        return MAX
    if vram >= FULL.min_vram_gb:
        return FULL
    if vram >= BALANCED.min_vram_gb:
        return BALANCED
    # lite tier: only drop to the 1b brain when we KNOW the machine is very
    # constrained (no GPU AND little RAM). Unknown RAM (0) is treated as adequate.
    if not has_gpu and 0.0 < ram < 8.0:
        return LITE_MIN
    return LITE


def capability_warning(hw) -> Optional[str]:
    """A kind heads-up when even the lite tier will struggle — shown by the
    wizard so a weak machine is told the truth, not crashed. None when fine."""
    has_gpu = bool(getattr(hw, "has_cuda_gpu", False))
    ram = float(getattr(hw, "ram_gb", 0.0) or 0.0)
    if not has_gpu and 0.0 < ram < 4.0:
        return ("Heads up: this computer has no AI graphics card and limited "
                "memory, so MEDO will run but may be slow to answer. For a "
                "smoother experience you'd want 8 GB+ of memory, or a computer "
                "with an NVIDIA graphics card.")
    return None
