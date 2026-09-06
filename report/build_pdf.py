"""Render report/report.md (plus the figures) to report/report.pdf.

Kept deliberately small: markdown -> HTML -> PyMuPDF Story -> PDF, with a
compact stylesheet so the finished report fits the 5-page limit.

Usage:
    python report/build_pdf.py
"""
from __future__ import annotations

import io
from pathlib import Path

import markdown
import pymupdf

HERE = Path(__file__).resolve().parent
MD = HERE / "report.md"
PDF = HERE / "report.pdf"
FIGDIR = HERE / "figures"

CSS = """
body { font-family: sans-serif; font-size: 9pt; line-height: 1.16; }
h1 { font-size: 15pt; margin: 0 0 4pt 0; }
h2 { font-size: 11.5pt; margin: 9pt 0 3pt 0; }
h3 { font-size: 9.6pt; margin: 6pt 0 2pt 0; }
p { margin: 3.5pt 0; }
ul, ol { margin: 3pt 0; padding-left: 16pt; }
li { margin: 1.5pt 0; }
table { border-collapse: collapse; font-size: 9pt; margin: 4pt 0; width: 100%; }
th, td { border-bottom: 0.4pt solid #bbb; padding: 2.5pt 3.5pt; text-align: left; vertical-align: top; }
th { font-weight: bold; }
tr { page-break-inside: avoid; }
img { width: 80%; margin: 4pt 0; }
hr { border: none; border-top: 0.5pt solid #ccc; margin: 6pt 0; }
code { font-family: monospace; font-size: 9pt; }
"""


def main() -> None:
    html_body = markdown.markdown(MD.read_text(encoding="utf-8"),
                                  extensions=["tables", "sane_lists"])
    html = f"<html><head><style>{CSS}</style></head><body>{html_body}</body></html>"

    page_rect = pymupdf.paper_rect("letter")
    margin = 40
    where = page_rect + (margin, margin, -margin, -margin)

    story = pymupdf.Story(html=html, archive=str(FIGDIR))
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    pages = 0
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
        pages += 1
    writer.close()

    # DocumentWriter emits uncompressed content streams (~8 MB for this report);
    # re-save with stream compression and dead-object cleanup.
    doc = pymupdf.open("pdf", buffer.getvalue())
    doc.save(str(PDF), garbage=4, deflate=True, deflate_images=True,
             deflate_fonts=True, use_objstms=1)
    doc.close()
    print(f"wrote {PDF.relative_to(HERE.parent)} "
          f"({pages} pages, {PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
