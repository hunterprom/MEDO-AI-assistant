"""The make-document skill: request -> compose -> render -> saved file.

A fake ``compose`` stands in for the LLM (canned Markdown) and file-opening is
stubbed, so the test proves the skill picks the right kind + format, writes a
real file, and degrades cleanly without a model — no LLM, no app windows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skills.base import SkillRequest
from skills.documents_gen import MakeDocumentSkill

FAKE_MD = "## Overview\nA short point about it.\n\n- one\n- two\n\n## Details\nMore.\n"


async def fake_compose(instruction: str) -> str:
    return FAKE_MD


@pytest.fixture(autouse=True)
def _no_open(monkeypatch):
    # Never actually launch Word/PowerPoint during a test.
    monkeypatch.setattr("skills.documents_gen.open_path", lambda *a, **k: None)


def _skill(tmp_path, compose=fake_compose):
    return MakeDocumentSkill(compose, out_dir=tmp_path)


async def _run(skill, text):
    return await skill.execute(SkillRequest(text=text, match=skill.match(text)))


@pytest.mark.asyncio
async def test_presentation_request_writes_a_pptx(tmp_path):
    result = await _run(_skill(tmp_path), "make a presentation about solar batteries")
    assert result.success and result.data["kind"] == "presentation"
    path = Path(result.data["path"])
    assert path.exists() and path.suffix == ".pptx"


@pytest.mark.asyncio
async def test_document_request_writes_a_docx(tmp_path):
    result = await _run(_skill(tmp_path), "write a report about the Q3 results")
    assert result.success and result.data["kind"] == "document"
    assert Path(result.data["path"]).suffix == ".docx"


@pytest.mark.asyncio
async def test_format_override_to_html(tmp_path):
    result = await _run(_skill(tmp_path), "write a report about cats as html")
    assert Path(result.data["path"]).suffix == ".html"
    # the "as html" hint is stripped from the subject/title
    assert "cats" in Path(result.data["path"]).stem.lower()
    assert "html" not in Path(result.data["path"]).stem.lower()


@pytest.mark.asyncio
async def test_no_topic_asks_what_about(tmp_path):
    result = await _run(_skill(tmp_path), "make a report")
    assert result.success is False and "about" in result.speech.lower()


@pytest.mark.asyncio
async def test_no_model_declines_cleanly(tmp_path):
    result = await _run(MakeDocumentSkill(None, out_dir=tmp_path),
                        "make a report about cats")
    assert result.success is False and "offline" in result.speech.lower()


@pytest.mark.asyncio
async def test_empty_content_does_not_save(tmp_path):
    async def empty(_instruction):
        return ""
    result = await _run(MakeDocumentSkill(empty, out_dir=tmp_path),
                        "make a report about cats")
    assert result.success is False
    assert not any(tmp_path.iterdir())          # nothing written


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
