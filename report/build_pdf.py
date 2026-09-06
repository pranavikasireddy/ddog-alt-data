"""Render report/report.md (plus the figures) to report/report.pdf.

Kept deliberately small: markdown -> HTML -> PyMuPDF Story -> PDF, with a
compact stylesheet so the finished report fits the 5-page limit.

Usage:
    python report/build_pdf.py
"""
from __future__ import annotations

from pathlib import Path

import markdown
import pymupdf

HERE = Path(__file__).resolve().parent
MD = HERE / "report.md"
PDF = HERE / "report.pdf"
FIGDIR = HERE / "figures"

CSS = """
body { font-family: sans-serif; font-size: 8.2pt; line-height: 1.28; }
h1 { font-size: 13.5pt; margin: 0 0 3pt 0; }
h2 { font-size: 10pt; margin: 8pt 0 2pt 0; }
h3 { font-size: 8.8pt; margin: 5pt 0 2pt 0; }
p { margin: 2.5pt 0; }
ul { margin: 2pt 0; padding-left: 13pt; }
li { margin: 1pt 0; }
table { border-collapse: collapse; font-size: 7pt; margin: 3pt 0; width: 100%; }
th, td { border-bottom: 0.4pt solid #bbb; padding: 2pt 3pt; text-align: left; vertical-align: top; }
th { font-weight: bold; }
tr { page-break-inside: avoid; }
img { width: 78%; margin: 3pt 0; }
hr { border: none; border-top: 0.5pt solid #ccc; margin: 5pt 0; }
code { font-family: monospace; font-size: 7.4pt; }
"""


def main() -> None:
    html_body = markdown.markdown(MD.read_text(encoding="utf-8"),
                                  extensions=["tables", "sane_lists"])
    html = f"<html><head><style>{CSS}</style></head><body>{html_body}</body></html>"

    page_rect = pymupdf.paper_rect("letter")
    margin = 40
    where = page_rect + (margin, margin, -margin, -margin)

    story = pymupdf.Story(html=html, archive=str(FIGDIR))
    writer = pymupdf.DocumentWriter(str(PDF))
    pages = 0
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
        pages += 1
    writer.close()
    print(f"wrote {PDF.relative_to(HERE.parent)} ({pages} pages)")


if __name__ == "__main__":
    main()
