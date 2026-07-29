"""Detect the machine so we can auto-fit a model profile (S2).

Detection probes are INJECTABLE and the nvidia-smi parsing is a pure function, so
the mapping is unit-tested without any real GPU. Everything degrades safely: a
missing nvidia-smi, an AMD/Intel GPU, or a probe that raises all resolve to "no
CUDA GPU" (→ the lite tier), never an exception.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareInfo:
    has_cuda_gpu: bool
    gpu_name: Optional[str]
    vram_gb: float            # 0.0 when none/unknown
    ram_gb: float             # 0.0 when unknown
    cpu_name: str
    cpu_cores: int

    def summary(self) -> str:
        gpu = (f"{self.gpu_name} ({self.vram_gb:.0f} GB)" if self.has_cuda_gpu
               else "no dedicated AI graphics card")
        return (f"{gpu}, {self.ram_gb:.0f} GB memory, "
                f"{self.cpu_cores}-core {self.cpu_name}")


def parse_nvidia_smi(output: str) -> Tuple[Optional[str], float]:
    """Parse `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` into
    (name, vram_gb) for the LARGEST GPU. Returns (None, 0.0) on anything odd.

    A line looks like: ``NVIDIA GeForce RTX 3060, 12288 MiB``.
    """
    best_name: Optional[str] = None
    best_gb = 0.0
    for line in (output or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        mem = parts[1].lower().replace("mib", "").replace("mb", "").strip()
        try:
            mib = float(mem)
        except ValueError:
            continue
        gb = mib / 1024.0
        if gb > best_gb:
            best_gb, best_name = gb, name
    return best_name, round(best_gb, 1)


def _run_nvidia_smi() -> Optional[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=6.0)
        if out.returncode == 0:
            return out.stdout
    except (OSError, subprocess.SubprocessError):
        pass          # no NVIDIA driver / tool -> not a CUDA machine
    return None


def _total_ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        return 0.0


def _cpu() -> Tuple[str, int]:
    name = platform.processor() or platform.machine() or "CPU"
    cores = 0
    try:
        import psutil
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 0
    except Exception:
        import os
        cores = os.cpu_count() or 0
    return name, int(cores)


def detect_hardware(*, nvidia_smi: Callable[[], Optional[str]] = _run_nvidia_smi,
                    ram_gb: Callable[[], float] = _total_ram_gb,
                    cpu: Callable[[], Tuple[str, int]] = _cpu) -> HardwareInfo:
    """Best-effort machine detection. All probes injectable for tests; each is
    guarded so a failure yields the safe 'no GPU' / unknown result, not a crash."""
    name: Optional[str] = None
    vram = 0.0
    try:
        raw = nvidia_smi()
        if raw:
            name, vram = parse_nvidia_smi(raw)
    except Exception:
        logger.warning("GPU detection failed; assuming no CUDA GPU", exc_info=True)
    has_gpu = bool(name) and vram > 0.0
    try:
        ram = ram_gb()
    except Exception:
        ram = 0.0
    try:
        cpu_name, cores = cpu()
    except Exception:
        cpu_name, cores = "CPU", 0
    return HardwareInfo(has_cuda_gpu=has_gpu, gpu_name=name, vram_gb=vram,
                        ram_gb=ram, cpu_name=cpu_name, cpu_cores=cores)
