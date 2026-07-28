"""Import skill: documents reach the RAG index, images reach it as descriptions.

The whitelist test is the important one — importing is the one operation that
reads a file MEDO was not previously pointed at.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from core.config import Settings
from core.docindex import DocumentIndex
from core.safety import PathWhitelist
from skills.base import SkillRequest, SkillResult
from skills.importer import (
    ImportFileSkill,
    classify_suffix,
    sidecar_text,
    unique_destination,
)


class FakeEmbedder:
    """Deterministic 8-dim vectors — no Ollama, but real numpy round-tripping."""

    def __call__(self, texts):
        import numpy as np

        out = []
        for t in texts:
            v = np.zeros(8, dtype="float32")
            for i, ch in enumerate(t.lower()):
                v[ord(ch) % 8] += 1.0
                if i > 400:
                    break
            out.append(v)
        return out


def build(tmp_path, describe=None, with_index=True):
    """A skill wired to a real (temp) index, whitelist and store."""
    allowed = tmp_path / "allowed"
    allowed.mkdir(exist_ok=True)
    settings = Settings()
    settings.memory.import_dir = str(tmp_path / "store")
    index = (DocumentIndex(tmp_path / "docs.db", FakeEmbedder(), [allowed])
             if with_index else None)
    skill = ImportFileSkill(settings, PathWhitelist([str(allowed)]), index,
                            describe=describe)
    return skill, allowed, index


def run(skill, text, **args):
    match = None
    for pattern in skill.patterns:
        match = pattern.search(text)
        if match:
            break
    return asyncio.run(skill.execute(
        SkillRequest(text=text, match=match, args=args)))


# -- routing -------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "import this file C:/tmp/spec.pdf",
    "import the document",
    "import ~/Downloads/notes.md",
    "learn this file report.pdf",
    "внеси документ",
    "внеси ја сликата",
    "внеси ~/Downloads/spec.pdf",
])
def test_patterns_claim_import_phrasings(text):
    assert any(p.search(text) for p in ImportFileSkill.patterns), text


@pytest.mark.parametrize("text", [
    "what did that document say",
    "open my documents",
    "search my documents for the invoice",
    "important meeting tomorrow",          # 'import' is a substring of nothing here
    # Verbs that belong to skills registered earlier — importing must not
    # steal them (see the pattern comment).
    "read this file notes.md",
    "remember this file notes.md",
    "запамти го овој фајл",
])
def test_patterns_leave_other_document_talk_alone(text):
    assert not any(p.search(text) for p in ImportFileSkill.patterns), text


def test_import_does_not_steal_verbs_from_earlier_skills():
    """Routing through the real registry, not the patterns in isolation."""
    from core.config import load_settings
    from core.docindex import DocumentIndex
    from main import Announcer, build_registry

    reg = build_registry(load_settings(), Announcer(),
                         doc_index=DocumentIndex(":memory:", None, []))
    expected = {
        "import this file C:/tmp/spec.pdf": "import_file",
        "внеси го документот договор.pdf": "import_file",
        "read notes.md": "notes",
        "remember this file notes.md": "remember_fact",
        "search my documents for the invoice": "search_documents",
    }
    for text, skill in expected.items():
        found = reg.find_match(text)
        assert found is not None and found[0].name == skill, text


def test_declares_pc_control():
    # It copies files on this machine; the router must gate it.
    assert ImportFileSkill.controls_pc is True


# -- pure helpers --------------------------------------------------------------

def test_classify_suffix():
    from pathlib import Path

    assert classify_suffix(Path("a.PDF")) == "document"
    assert classify_suffix(Path("a.jpeg")) == "image"
    assert classify_suffix(Path("a.exe")) == ""
    assert classify_suffix(Path("a")) == ""


def test_unique_destination_never_overwrites(tmp_path):
    (tmp_path / "spec.pdf").write_bytes(b"first")
    second = unique_destination(tmp_path, "spec.pdf")
    assert second != tmp_path / "spec.pdf"
    assert second.suffix == ".pdf"


def test_sidecar_records_the_original_path(tmp_path):
    text = sidecar_text("diagram.png", tmp_path / "orig" / "diagram.png", "a resistor")
    assert "diagram.png" in text
    assert "orig" in text
    assert "a resistor" in text


# -- documents -----------------------------------------------------------------

def test_document_import_is_searchable_afterwards(tmp_path):
    skill, allowed, index = build(tmp_path)
    doc = allowed / "warranty.md"
    doc.write_text("The kettle warranty lasts three years from purchase.",
                   encoding="utf-8")

    result = run(skill, f"import this file {doc}")

    assert result.success and result.data["imported"]
    assert result.data["chunks"] >= 1
    # the copy lives in the managed store, the original is untouched
    from pathlib import Path

    assert Path(result.data["path"]).parent == tmp_path / "store"
    assert doc.exists()
    # and it is retrievable through the existing documents search
    hits = index.search("kettle warranty")
    assert any("warranty" in h["text"] for h in hits)


def test_reimporting_the_same_file_does_not_double_its_chunks(tmp_path):
    skill, allowed, index = build(tmp_path)
    doc = allowed / "note.md"
    doc.write_text("one line of text about lithium batteries", encoding="utf-8")

    first = run(skill, f"import this file {doc}")
    # a second import lands at a new path (nothing is overwritten), but
    # re-indexing the SAME path must replace, not accumulate
    index.index_file(__import__("pathlib").Path(first.data["path"]))
    index.index_file(__import__("pathlib").Path(first.data["path"]))
    assert index.stats()["chunks"] == first.data["chunks"]


def test_document_import_says_so_when_it_cannot_index(tmp_path):
    skill, allowed, _ = build(tmp_path, with_index=False)
    doc = allowed / "spec.md"
    doc.write_text("some text", encoding="utf-8")

    result = run(skill, f"import this file {doc}")

    assert result.data["imported"] and result.data["chunks"] == 0
    # honest: it must not imply the document is ready to be asked about
    assert "couldn't index" in result.speech.lower()


# -- images --------------------------------------------------------------------

def _png_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (16, 16), (10, 10, 10)).save(buf, format="PNG")
    return buf.getvalue()


def test_image_import_stores_and_indexes_the_description(tmp_path):
    seen = {}

    async def describe(image_b64, prompt):
        seen["b64"] = image_b64
        seen["prompt"] = prompt
        return SkillResult("An Arduino Uno with a 220 ohm resistor on pin 13.")

    skill, allowed, index = build(tmp_path, describe=describe)
    img = allowed / "diagram.png"
    img.write_bytes(_png_bytes())

    result = run(skill, f"import this picture {img}")

    assert result.data["described"] and result.data["kind"] == "image"
    assert seen["b64"]                       # the image actually reached the model
    from pathlib import Path

    sidecar = Path(result.data["sidecar"])
    assert sidecar.exists() and "resistor" in sidecar.read_text(encoding="utf-8")
    # "what was in that diagram I imported" is an ordinary RAG query
    hits = index.search("arduino resistor diagram")
    assert any("resistor" in h["text"] for h in hits)


def test_image_import_is_honest_when_the_vision_model_fails(tmp_path):
    async def describe(image_b64, prompt):
        return SkillResult("can't reach my vision model", success=False)

    skill, allowed, _ = build(tmp_path, describe=describe)
    img = allowed / "photo.jpg"
    img.write_bytes(_png_bytes())

    result = run(skill, f"import this picture {img}")

    assert result.data["described"] is False
    assert "couldn't describe" in result.speech.lower()


def test_image_import_survives_a_raising_vision_model(tmp_path):
    async def describe(image_b64, prompt):
        raise RuntimeError("ollama exploded")

    skill, allowed, _ = build(tmp_path, describe=describe)
    img = allowed / "photo.png"
    img.write_bytes(_png_bytes())

    result = run(skill, f"import this picture {img}")
    assert result.data["imported"] and result.data["described"] is False


# -- safety --------------------------------------------------------------------

def test_import_outside_the_whitelist_is_refused(tmp_path):
    skill, _allowed, _ = build(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    secret = outside / "passwords.md"
    secret.write_text("hunter2", encoding="utf-8")

    result = run(skill, f"import this file {secret}")

    assert not result.success
    assert result.data["reason"] == "outside"
    # nothing was copied
    assert not (tmp_path / "store").exists() or not list((tmp_path / "store").iterdir())


def test_executables_are_not_importable(tmp_path):
    skill, allowed, _ = build(tmp_path)
    exe = allowed / "installer.exe"
    exe.write_bytes(b"MZ")

    result = run(skill, f"import this file {exe}")

    assert not result.success and result.data["reason"] == "unsupported"
    assert not (tmp_path / "store").exists() or not list((tmp_path / "store").iterdir())


def test_missing_file_is_reported_not_invented(tmp_path):
    skill, allowed, _ = build(tmp_path)
    result = run(skill, f"import this file {allowed / 'nope.md'}")
    assert not result.success and result.data["reason"] == "not found"


def test_bare_import_asks_which_file(tmp_path):
    skill, _allowed, _ = build(tmp_path)
    result = run(skill, "import the document")
    assert not result.success and result.data["reason"] == "which file"
    assert "which file" in result.speech.lower()


def test_macedonian_request_is_answered_in_macedonian(tmp_path):
    skill, allowed, _ = build(tmp_path)
    doc = allowed / "договор.md"
    doc.write_text("текст на договорот", encoding="utf-8")

    result = run(skill, f"внеси документ {doc}")

    assert result.data["imported"]
    assert re.search(r"[Ѐ-ӿ]", result.speech)


def test_tool_schema_is_well_formed(tmp_path):
    skill, _allowed, _ = build(tmp_path)
    schema = skill.tool_schema()
    assert schema["function"]["name"] == "import_file"
    assert schema["function"]["parameters"]["required"] == ["path"]
