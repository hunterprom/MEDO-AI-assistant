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
    parse_table,
    write,
)

TABLE = "| Name | Role |\n|---|---|\n| Ada | Engineer |\n| Bob | Designer |\n"

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


def test_parse_table_extracts_rows_without_the_separator():
    rows = parse_table(TABLE)
    assert rows == [["Name", "Role"], ["Ada", "Engineer"], ["Bob", "Designer"]]
    assert parse_table("no table here") is None


def test_xlsx_is_a_valid_workbook_with_the_table(tmp_path):
    assert "xlsx" in available_formats()
    out = write(TABLE, tmp_path / "d.xlsx", "xlsx", title="Roster")
    assert zipfile.is_zipfile(out)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert any("worksheets/sheet1.xml" in n for n in names)
        strings = z.read("xl/sharedStrings.xml").decode("utf-8")
        assert "Ada" in strings and "Role" in strings


def test_xlsx_does_not_write_live_formulas(tmp_path):
    # Cell text is LLM-composed and can be injection-tainted, so a cell starting
    # with '=' must be a literal string, never a formula that runs on open.
    tainted = ('| Item | Note |\n|---|---|\n'
               '| =1+2 | =HYPERLINK("http://evil") |\n')
    out = write(tainted, tmp_path / "inj.xlsx", "xlsx", title="T")
    with zipfile.ZipFile(out) as z:
        sheet = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
        assert "<f>" not in sheet                     # no formula elements at all
        strings = z.read("xl/sharedStrings.xml").decode("utf-8")
        assert "=1+2" in strings                       # the '=' cell is literal text


def test_unknown_format_is_a_clean_error(tmp_path):
    with pytest.raises(DocumentError, match="unknown format"):
        write(SAMPLE, tmp_path / "d.xyz", "xyz")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
