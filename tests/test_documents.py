"""The document renderer: Markdown -> docx / pptx / html / slides / md / txt.

The office formats are opened back up with their own libraries to prove the
content really landed (not just that a file appeared); the stdlib formats are
checked as text. Pure + offline — no LLM, no network.
"""

from __future__ import annotations

import zipfile

import pytest

from core.documents import (
    DocumentError,
    available_formats,
    parse_markdown,
    write,
)

SAMPLE = """\
## Introduction
This is the **first** section.

- point one
- point two

## Details
Some detail text.

1. step one
2. step two
"""


def test_parse_markdown_blocks():
    blocks = parse_markdown(SAMPLE)
    kinds = [k for k, _ in blocks]
    assert kinds == ["h2", "p", "bullets", "h2", "p", "numbered"]
    assert blocks[2] == ("bullets", ["point one", "point two"])
    assert blocks[5] == ("numbered", ["step one", "step two"])


def test_available_formats_includes_office_and_stdlib():
    fmts = available_formats()
    assert {"html", "slides", "md", "txt"} <= set(fmts)   # always
    assert {"docx", "pptx"} <= set(fmts)                  # installed in this env


def test_html_has_structure_and_inline_bold(tmp_path):
    out = write(SAMPLE, tmp_path / "d.html", "html", title="My Report")
    text = out.read_text(encoding="utf-8")
    assert "<h1>My Report</h1>" in text
    assert "<h2>Introduction</h2>" in text
    assert "<strong>first</strong>" in text
    assert "<li>point one</li>" in text and "<ol>" in text


def test_slides_makes_one_section_per_heading(tmp_path):
    out = write(SAMPLE, tmp_path / "d.html", "slides", title="Deck")
    text = out.read_text(encoding="utf-8")
    # title slide + one per h2
    assert text.count('class="slide"') == 3


def test_txt_strips_markup(tmp_path):
    out = write(SAMPLE, tmp_path / "d.txt", "txt", title="Plain")
    text = out.read_text(encoding="utf-8")
    assert "point one" in text and "**" not in text and "first" in text


def test_docx_is_a_valid_document_with_the_content(tmp_path):
    from docx import Document

    out = write(SAMPLE, tmp_path / "d.docx", "docx", title="Doc Title")
    assert zipfile.is_zipfile(out)                         # docx == a zip package
    doc = Document(str(out))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Doc Title" in text and "Introduction" in text
    assert "point one" in text and "first" in text
    # the **first** span rendered as a bold run
    runs = [r for p in doc.paragraphs for r in p.runs]
    assert any(r.bold and "first" in r.text for r in runs)


def test_pptx_has_a_slide_per_heading(tmp_path):
    from pptx import Presentation

    out = write(SAMPLE, tmp_path / "d.pptx", "pptx", title="Deck Title")
    assert zipfile.is_zipfile(out)
    prs = Presentation(str(out))
    titles = [s.shapes.title.text for s in prs.slides if s.shapes.title]
    assert "Deck Title" in titles and "Introduction" in titles and "Details" in titles
    all_text = "\n".join(
        shape.text_frame.text for s in prs.slides
        for shape in s.shapes if shape.has_text_frame)
    assert "point one" in all_text and "step one" in all_text


def test_unknown_format_is_a_clean_error(tmp_path):
    with pytest.raises(DocumentError, match="unknown format"):
        write(SAMPLE, tmp_path / "d.xyz", "xyz")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
