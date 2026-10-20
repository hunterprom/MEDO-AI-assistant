"""M10 see_bench: mocked capture/vision/OCR, config gating, inventory."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import load_settings
from skills.base import SkillRequest, SkillResult
from skills.bench import (
    BenchInventory,
    BenchSkill,
    center_crop_jpeg,
    extract_markings,
)

FAKE_JPEG = b"\xff\xd8\xff\xe0 not a real jpeg"


def _fake_embedder(texts):
    """Deterministic 2-d vectors: resistor-ish -> [1,0], everything else [0,1]."""
    out = []
    for t in texts:
        vec = [1.0, 0.0] if "resistor" in t.lower() else [0.0, 1.0]
        out.append(np.asarray(vec, dtype=np.float32))
    return out


def _skill(tmp_path, *, enabled=True, describe_speech="Looks like a voltage regulator",
           ocr_text="AMS1117-3.3\nrohs 2024", embedder=None,
           fetch_fails=False) -> tuple[BenchSkill, BenchInventory, dict]:
    settings = load_settings()
    settings.vision.bench.enabled = enabled
    inventory = BenchInventory(tmp_path / "bench.db", embedder)
    calls = {"fetched": 0}

    async def fetch():
        calls["fetched"] += 1
        if fetch_fails:
            raise ConnectionError("sidecar down")
        return FAKE_JPEG

    async def describe(image_b64: str) -> SkillResult:
        return SkillResult(describe_speech)

    return BenchSkill(settings, inventory, fetch_frame=fetch,
                      describe=describe, ocr=lambda jpeg: ocr_text), inventory, calls


# --- pure helpers ------------------------------------------------------------


def test_extract_markings_finds_part_numbers_only():
    text = "made in TAIWAN\nAMS1117-3.3  rohs\nLOT 4 CE ok STM32F103C8T6"
    assert extract_markings(text) == ["AMS1117-3.3", "STM32F103C8T6"]
    assert extract_markings("no markings here") == []           # needs a digit
    assert extract_markings("") == []


def test_center_crop_survives_undecodable_bytes():
    assert center_crop_jpeg(b"junk") == b"junk"  # optimization, never a failure


# --- config gating -----------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_config_refuses_without_touching_the_camera(tmp_path):
    skill, _, calls = _skill(tmp_path, enabled=False)
    r = await skill.execute(SkillRequest(text="what's on my bench"))
    assert not r.success and "disabled" in r.speech
    assert calls["fetched"] == 0


@pytest.mark.asyncio
async def test_camera_unreachable_degrades(tmp_path):
    skill, _, _ = _skill(tmp_path, fetch_fails=True)
    r = await skill.execute(SkillRequest(text="identify this part"))
    assert not r.success and "sidecar" in r.speech


# --- identification ----------------------------------------------------------


@pytest.mark.asyncio
async def test_identify_merges_vision_and_ocr_markings(tmp_path):
    skill, _, calls = _skill(tmp_path)
    r = await skill.execute(SkillRequest(text="identify this part"))
    assert r.success and calls["fetched"] == 1
    assert r.speech.startswith("Looks like a voltage regulator.")
    assert "Markings: AMS1117-3.3" in r.speech
    assert r.data["markings"] == ["AMS1117-3.3"]


@pytest.mark.asyncio
async def test_ocr_less_answer_still_speaks(tmp_path):
    skill, _, _ = _skill(tmp_path, ocr_text="")  # tesseract missing -> ""
    r = await skill.execute(SkillRequest(text="што е ова"))
    assert r.success and "Markings" not in r.speech


# --- inventory ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_this_part_appends_to_inventory(tmp_path):
    skill, inventory, _ = _skill(tmp_path, describe_speech="A 10k resistor")
    r = await skill.execute(SkillRequest(text="log this part"))
    assert r.success and r.speech.startswith("Logged.")
    assert inventory.search("resistor") == ["A 10k resistor. Markings: AMS1117-3.3."]


@pytest.mark.asyncio
async def test_inventory_semantic_search_with_embedder(tmp_path):
    skill, inventory, calls = _skill(tmp_path, embedder=_fake_embedder)
    inventory.add("a 10k resistor", "104")
    inventory.add("blue LED", "")
    r = await skill.execute(SkillRequest(
        text="do i have any 10k resistors",
        match=next(p.search("do i have any 10k resistors")
                   for p in skill.patterns if p.search("do i have any 10k resistors"))))
    assert r.success
    assert r.data["hits"][0] == "a 10k resistor"   # semantic top hit
    assert calls["fetched"] == 0                    # recall never opens the camera


@pytest.mark.asyncio
async def test_inventory_search_works_even_when_bench_disabled(tmp_path):
    skill, inventory, _ = _skill(tmp_path, enabled=False)
    inventory.add("blue LED", "")
    match = skill.match("do i have any LEDs")
    r = await skill.execute(SkillRequest(text="do i have any LEDs", match=match))
    assert r.success and "blue LED" in r.speech


@pytest.mark.asyncio
async def test_empty_inventory_says_so(tmp_path):
    skill, _, _ = _skill(tmp_path)
    match = skill.match("do i have any op amps")
    r = await skill.execute(SkillRequest(text="do i have any op amps", match=match))
    assert "Nothing in the bench inventory" in r.speech


def test_do_i_have_any_non_part_questions_do_not_hijack_bench(tmp_path):
    # "do I have any meetings/emails/..." must NOT be claimed by the bench —
    # they aren't parts, and LocateAppSkill declines an "any"-led name, so a
    # too-broad guard sent them here and searched electronics for "meetings".
    skill, _, _ = _skill(tmp_path)
    for text in (
        "do i have any meetings today",
        "do i have any unread emails",
        "do i have any new messages",
        "do i have any money in my account",
        "do i have any appointments tomorrow",
        "do i have any notifications",
    ):
        assert skill.match(text) is None, text
    # …but genuine part questions still route to the inventory.
    for text in ("do i have any 10k resistors", "do i have any LEDs",
                 "do i have any op amps"):
        assert skill.match(text) is not None, text
