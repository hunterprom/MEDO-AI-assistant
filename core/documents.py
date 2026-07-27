"""Turn Markdown into real deliverables — Word docs, slide decks, web pages.

MEDO composes a document as Markdown (LLMs are good at that), and this renders it
to the file the user actually wanted. The split is deliberate: content generation
is one concern, formatting is another, and only the second half needs libraries.

Formats:

======  ============================  ==============================
fmt     output                        needs
======  ============================  ==============================
docx    Word document                 python-docx
pptx    PowerPoint deck               python-pptx
html    styled self-contained page    nothing (stdlib)
slides  self-contained HTML slide deck nothing (stdlib)
md      the Markdown itself           nothing
txt     plain text (markup stripped)  nothing
======  ============================  ==============================

The dependency-free formats always work, so the capability degrades gracefully
on a machine without the office libraries rather than failing outright.
"""

from __future__ import annotations

import html as _html
import re
from pathlib import Path

#: Every format the renderer knows how to emit.
ALL_FORMATS = ("docx", "pptx", "xlsx", "html", "slides", "md", "txt")
#: These need an office library; the rest are stdlib-only.
_LIB_FORMATS = {"docx": "docx", "pptx": "pptx", "xlsx": "xlsxwriter"}


class DocumentError(Exception):
    """A document couldn't be rendered (unknown format, missing library, …)."""


# --- a tiny Markdown model ---------------------------------------------------
#
# Just the subset MEDO generates: headings, paragraphs, bullet and numbered
# lists, plus **bold** inline. A block is ``(kind, value)`` where value is a
# string (headings/paragraphs) or a list of strings (list items).

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^[-*]\s+(.*)$")
_NUMBERED = re.compile(r"^\d+[.)]\s+(.*)$")
_BOLD = re.compile(r"\*\*(.+?)\*\*")


def parse_markdown(md: str) -> list[tuple[str, object]]:
    """Markdown text -> a flat list of ``(kind, value)`` blocks."""
    blocks: list[tuple[str, object]] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            blocks.append(("p", " ".join(para).strip()))
            para.clear()

    lines = md.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            flush()
            i += 1
            continue
        h = _HEADING.match(stripped)
        if h:
            flush()
            level = min(len(h.group(1)), 3)
            blocks.append((f"h{level}", h.group(2).strip()))
            i += 1
            continue
        if _BULLET.match(stripped):
            flush()
            items = []
            while i < len(lines) and _BULLET.match(lines[i].strip()):
                items.append(_BULLET.match(lines[i].strip()).group(1).strip())
                i += 1
            blocks.append(("bullets", items))
            continue
        if _NUMBERED.match(stripped):
            flush()
            items = []
            while i < len(lines) and _NUMBERED.match(lines[i].strip()):
                items.append(_NUMBERED.match(lines[i].strip()).group(1).strip())
                i += 1
            blocks.append(("numbered", items))
            continue
        para.append(stripped)
        i += 1
    flush()
    return blocks


def _strip_markup(text: str) -> str:
    return _BOLD.sub(r"\1", text)


_TABLE_SEP = re.compile(r"^:?-{2,}:?$")


def parse_table(md: str) -> list[list[str]] | None:
    """The first GitHub-style Markdown table as rows of cells, or None."""
    rows: list[list[str]] = []
    for raw in md.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            if rows:
                break                       # the table ended
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and all(_TABLE_SEP.match(c) for c in cells if c):
            continue                        # the |---|---| separator row
        rows.append([_strip_markup(c) for c in cells])
    return rows or None


# --- public entry point ------------------------------------------------------

def available_formats() -> list[str]:
    """The formats usable right now (office libs may be absent)."""
    out = []
    for fmt in ALL_FORMATS:
        lib = _LIB_FORMATS.get(fmt)
        if lib is None:
            out.append(fmt)
            continue
        try:
            __import__(lib)
            out.append(fmt)
        except ImportError:
            pass
    return out


def write(markdown: str, out_path: str | Path, fmt: str,
          title: str | None = None) -> Path:
    """Render ``markdown`` to ``out_path`` in ``fmt``; returns the written path.

    Raises :class:`DocumentError` for an unknown format or a missing library.
    """
    fmt = (fmt or "").lower().lstrip(".")
    if fmt not in ALL_FORMATS:
        raise DocumentError(
            f"unknown format {fmt!r} — choose one of {', '.join(ALL_FORMATS)}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks = parse_markdown(markdown)
    renderer = {
        "docx": _to_docx, "pptx": _to_pptx, "xlsx": _to_xlsx, "html": _to_html,
        "slides": _to_slides, "md": _to_md, "txt": _to_txt,
    }[fmt]
    renderer(blocks, out, title, markdown)
    return out


# --- renderers ---------------------------------------------------------------

def _to_md(blocks, out: Path, title, markdown: str) -> None:
    text = markdown if not title else f"# {title}\n\n{markdown}"
    out.write_text(text.strip() + "\n", encoding="utf-8")


def _to_txt(blocks, out: Path, title, markdown: str) -> None:
    lines = []
    if title:
        lines += [title, "=" * len(title), ""]
    for kind, val in blocks:
        if kind in ("h1", "h2", "h3"):
            lines += ["", _strip_markup(val), ""]
        elif kind == "p":
            lines += [_strip_markup(val), ""]
        elif kind == "bullets":
            lines += [f"  - {_strip_markup(it)}" for it in val] + [""]
        elif kind == "numbered":
            lines += [f"  {n}. {_strip_markup(it)}" for n, it in enumerate(val, 1)] + [""]
    out.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _inline_html(text: str) -> str:
    # escape first, then re-introduce the one bit of markup we support.
    return _BOLD.sub(r"<strong>\1</strong>", _html.escape(text))


def _blocks_html(blocks) -> str:
    parts = []
    for kind, val in blocks:
        if kind in ("h1", "h2", "h3"):
            parts.append(f"<{kind}>{_inline_html(val)}</{kind}>")
        elif kind == "p":
            parts.append(f"<p>{_inline_html(val)}</p>")
        elif kind == "bullets":
            parts.append("<ul>" + "".join(
                f"<li>{_inline_html(it)}</li>" for it in val) + "</ul>")
        elif kind == "numbered":
            parts.append("<ol>" + "".join(
                f"<li>{_inline_html(it)}</li>" for it in val) + "</ol>")
    return "\n".join(parts)


_HTML_CSS = (
    "body{max-width:46rem;margin:2.5rem auto;padding:0 1.2rem;"
    "font:16px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;color:#1a2230}"
    "h1{font-size:2rem}h2{margin-top:2rem}code{background:#eef2f7;padding:.1em .3em}"
    "ul,ol{padding-left:1.4rem}"
)


def _to_html(blocks, out: Path, title, markdown: str) -> None:
    heading = f"<h1>{_html.escape(title)}</h1>\n" if title else ""
    page = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{_html.escape(title or 'Document')}</title>"
            f"<style>{_HTML_CSS}</style></head><body>\n"
            f"{heading}{_blocks_html(blocks)}\n</body></html>\n")
    out.write_text(page, encoding="utf-8")


_SLIDES_CSS = (
    "body{margin:0;font:20px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;"
    "background:#0f1720;color:#eaf2ff}"
    ".slide{min-height:100vh;box-sizing:border-box;padding:6vh 8vw;"
    "display:flex;flex-direction:column;justify-content:center;"
    "border-bottom:1px solid #223}"
    ".slide h1,.slide h2{margin:0 0 .6em;color:#8fd0ff}"
    ".slide li{margin:.35em 0}"
)


def _slide_groups(blocks) -> list[tuple[str, list]]:
    """Group blocks into (slide_title, body_blocks) — a heading starts a slide."""
    slides: list[tuple[str, list]] = []
    cur_title, cur_body = None, []
    for kind, val in blocks:
        if kind in ("h1", "h2"):
            if cur_title is not None or cur_body:
                slides.append((cur_title or "", cur_body))
            cur_title, cur_body = val, []
        else:
            cur_body.append((kind, val))
    if cur_title is not None or cur_body:
        slides.append((cur_title or "", cur_body))
    return slides


def _to_slides(blocks, out: Path, title, markdown: str) -> None:
    groups = _slide_groups(blocks)
    slides_html = []
    if title:
        slides_html.append(f"<section class=\"slide\"><h1>{_html.escape(title)}</h1></section>")
    for stitle, body in groups:
        head = f"<h2>{_inline_html(stitle)}</h2>" if stitle else ""
        slides_html.append(f"<section class=\"slide\">{head}\n{_blocks_html(body)}</section>")
    page = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{_html.escape(title or 'Slides')}</title>"
            f"<style>{_SLIDES_CSS}</style></head><body>\n"
            + "\n".join(slides_html) + "\n</body></html>\n")
    out.write_text(page, encoding="utf-8")


def _add_docx_runs(paragraph, text: str) -> None:
    # Split on **bold** so those spans render bold, the rest plain.
    for i, chunk in enumerate(_BOLD.split(text)):
        if chunk:
            paragraph.add_run(chunk).bold = bool(i % 2)


def _to_docx(blocks, out: Path, title, markdown: str) -> None:
    try:
        from docx import Document
    except ImportError as exc:
        raise DocumentError("python-docx isn't installed — pip install python-docx") from exc
    doc = Document()
    if title:
        doc.add_heading(title, 0)
    for kind, val in blocks:
        if kind == "h1":
            doc.add_heading(_strip_markup(val), 1)
        elif kind == "h2":
            doc.add_heading(_strip_markup(val), 2)
        elif kind == "h3":
            doc.add_heading(_strip_markup(val), 3)
        elif kind == "p":
            _add_docx_runs(doc.add_paragraph(), val)
        elif kind == "bullets":
            for it in val:
                _add_docx_runs(doc.add_paragraph(style="List Bullet"), it)
        elif kind == "numbered":
            for it in val:
                _add_docx_runs(doc.add_paragraph(style="List Number"), it)
    doc.save(str(out))


def _to_pptx(blocks, out: Path, title, markdown: str) -> None:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise DocumentError("python-pptx isn't installed — pip install python-pptx") from exc
    prs = Presentation()
    if title:
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        slide.shapes.title.text = title
    for stitle, body in _slide_groups(blocks):
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = _strip_markup(stitle) or "Slide"
        tf = slide.placeholders[1].text_frame
        tf.clear()
        first = True
        for kind, val in body:
            items = val if isinstance(val, list) else [val]
            for it in items:
                para = tf.paragraphs[0] if first else tf.add_paragraph()
                para.text = _strip_markup(it)
                if kind in ("bullets", "numbered"):
                    para.level = 1
                first = False
    prs.save(str(out))


def _to_xlsx(blocks, out: Path, title, markdown: str) -> None:
    try:
        import xlsxwriter
    except ImportError as exc:
        raise DocumentError("xlsxwriter isn't installed — pip install xlsxwriter") from exc
    rows = parse_table(markdown)
    if rows is None:
        # No table in the content — fall back to one column of the block text, so
        # a spreadsheet request never produces an empty file.
        rows = [[_strip_markup(v)] for k, v in blocks
                if k in ("h1", "h2", "h3", "p")]
        rows += [[_strip_markup(it)]
                 for k, v in blocks if k in ("bullets", "numbered") for it in v]
    # strings_to_formulas=False: cell text comes from the (LLM-composed, possibly
    # injection-tainted) content, so a cell like "=cmd|'/c calc'!A1" or
    # "=HYPERLINK(...)" must be written as a LITERAL string, never interpreted as
    # a live formula that runs when the file is opened. write_string below is the
    # belt to this option's braces.
    wb = xlsxwriter.Workbook(str(out), {"strings_to_formulas": False})
    ws = wb.add_worksheet((title or "Sheet")[:31])
    header = wb.add_format({"bold": True})
    widths: dict[int, int] = {}
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            ws.write_string(r, c, cell, header if r == 0 else None)
            widths[c] = max(widths.get(c, 10), min(len(cell) + 2, 60))
    for c, w in widths.items():
        ws.set_column(c, c, w)
    wb.close()
