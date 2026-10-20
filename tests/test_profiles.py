"""Hardware -> profile mapping (S2): the part that makes MEDO actually usable.

Pins the approved tiering (fits a brain to the GPU: 30B only at 24 GB+), the
no-GPU/unknown fallbacks, nvidia-smi parsing, and per-user settings persistence
with a manual override. No real GPU or APPDATA is touched.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app import profiles, user_settings
from app.hardware import HardwareInfo, detect_hardware, parse_nvidia_smi


@dataclass
class _HW:
    has_cuda_gpu: bool = False
    vram_gb: float = 0.0
    ram_gb: float = 16.0


# -- tier mapping -------------------------------------------------------------

@pytest.mark.parametrize("vram,expected", [
    (24.0, "max"), (48.0, "max"),
    (16.0, "full"), (12.0, "full"),
    (11.0, "balanced"), (8.0, "balanced"), (6.0, "balanced"),
])
def test_gpu_vram_tiers_map_correctly(vram, expected):
    hw = _HW(has_cuda_gpu=True, vram_gb=vram)
    assert profiles.select_profile(hw).name == expected


def test_thirty_b_is_reserved_for_24gb_plus():
    # The honest rule: a 12-16 GB card must NOT auto-get qwen3:30b (it offloads).
    assert profiles.select_profile(_HW(True, 12.0)).llm == "qwen2.5:14b"
    assert profiles.select_profile(_HW(True, 16.0)).llm == "qwen2.5:14b"
    assert profiles.select_profile(_HW(True, 24.0)).llm == "qwen3:30b"


def test_no_gpu_maps_to_lite():
    assert profiles.select_profile(_HW(has_cuda_gpu=False, ram_gb=16.0)).name == "lite"
    # a weak GPU under the balanced bar is still lite
    assert profiles.select_profile(_HW(True, 4.0)).name == "lite"


def test_cpu_only_low_ram_drops_to_the_smallest_brain():
    p = profiles.select_profile(_HW(has_cuda_gpu=False, ram_gb=6.0))
    assert p.name == "lite" and p.llm == "llama3.2:1b"       # LITE_MIN
    # adequate RAM without a GPU still gets the 3b lite brain
    assert profiles.select_profile(_HW(False, ram_gb=16.0)).llm == "llama3.2:3b"


def test_unknown_hardware_falls_back_to_safe_lite():
    # everything zero/unknown -> lite, never a crash (ram 0 treated as adequate)
    assert profiles.select_profile(_HW(False, 0.0, 0.0)).name == "lite"
    assert profiles.select_profile(object()).name == "lite"  # duck-typed, missing attrs


def test_lite_turns_vision_off_others_keep_it():
    assert profiles.select_profile(_HW(False, ram_gb=16)).vision is None
    assert profiles.select_profile(_HW(True, 8.0)).vision == "qwen2.5vl:3b"


def test_profile_lists_its_ollama_pulls():
    assert profiles.FULL.ollama_models() == \
        ["qwen2.5:14b", "qwen2.5vl:3b", "nomic-embed-text"]
    assert profiles.LITE.ollama_models() == ["llama3.2:3b", "nomic-embed-text"]  # no vision


def test_capability_warning_only_for_very_weak_machines():
    assert profiles.capability_warning(_HW(False, ram_gb=2.0)) is not None
    assert profiles.capability_warning(_HW(False, ram_gb=16.0)) is None
    assert profiles.capability_warning(_HW(True, 8.0)) is None


# -- nvidia-smi parsing + detection fallbacks ---------------------------------

def test_parse_nvidia_smi_picks_the_largest_gpu():
    out = ("NVIDIA GeForce RTX 3060, 12288 MiB\n"
           "NVIDIA GeForce GT 1030, 2048 MiB\n")
    name, gb = parse_nvidia_smi(out)
    assert name == "NVIDIA GeForce RTX 3060" and gb == pytest.approx(12.0, abs=0.1)


def test_parse_nvidia_smi_handles_garbage():
    assert parse_nvidia_smi("") == (None, 0.0)
    assert parse_nvidia_smi("no gpus here") == (None, 0.0)


def test_detect_hardware_without_nvidia_is_no_gpu():
    hw = detect_hardware(nvidia_smi=lambda: None, ram_gb=lambda: 16.0,
                         cpu=lambda: ("Test CPU", 8))
    assert hw.has_cuda_gpu is False and hw.vram_gb == 0.0
    assert hw.ram_gb == 16.0 and hw.cpu_cores == 8


def test_detect_hardware_survives_probe_errors():
    def boom():
        raise RuntimeError("driver crash")
    hw = detect_hardware(nvidia_smi=boom, ram_gb=boom, cpu=boom)
    assert hw.has_cuda_gpu is False and hw.ram_gb == 0.0     # safe, no exception


def test_detect_hardware_reads_a_real_gpu():
    hw = detect_hardware(
        nvidia_smi=lambda: "NVIDIA GeForce RTX 4090, 24564 MiB\n",
        ram_gb=lambda: 64.0, cpu=lambda: ("Ryzen", 16))
    assert hw.has_cuda_gpu and hw.vram_gb == pytest.approx(24.0, abs=0.1)
    assert profiles.select_profile(hw).name == "max"


# -- per-user settings persistence + override ---------------------------------

def test_profile_choice_persists(tmp_path):
    p = tmp_path / "settings.json"
    assert user_settings.get_profile(p) is None
    user_settings.set_profile("balanced", override=False,
                              explanation="picked for your GPU", path=p)
    assert user_settings.get_profile(p) == "balanced"
    assert user_settings.is_override(p) is False
    assert user_settings.load(p)["profile_explanation"] == "picked for your GPU"


def test_manual_override_is_recorded(tmp_path):
    p = tmp_path / "settings.json"
    user_settings.set_profile("max", override=True, path=p)
    assert user_settings.get_profile(p) == "max"
    assert user_settings.is_override(p) is True


def test_settings_survive_a_corrupt_file(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("}{ not json", encoding="utf-8")
    assert user_settings.load(p) == {}                       # never raises
    user_settings.set_profile("lite", override=False, path=p)  # and recovers
    assert user_settings.get_profile(p) == "lite"


def test_user_dir_is_under_appdata_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert user_settings.user_dir() == tmp_path / "MEDO"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
