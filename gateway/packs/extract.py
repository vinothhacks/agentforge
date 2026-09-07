"""Shared text extraction: txt/md, PDF, DOCX.

DOCX uses stdlib zipfile + ElementTree so we add no dependency and work offline.
Both ingest (rag) and the read tool (fs_read) call through here so a file can
never be searchable but unreadable, or the reverse.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

TEXT_EXTS = {".txt", ".md", ".json", ".csv", ".yaml", ".yml", ".log", ".ini", ".xml"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
EXTRACTABLE_EXTS = TEXT_EXTS | PDF_EXTS | DOCX_EXTS


def clean_pdf_text(raw: str, layout: bool) -> str:
    text = raw.replace("\x00", "")
    text = re.sub(r"-\n(?=[A-Za-z])", "", text)
    if layout:
        text = re.sub(r"[ \t]{2,}", " | ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Return page records and whether any text layer was found."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[dict[str, Any]] = []
    has_text = False
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text(extraction_mode="layout") or ""
            layout = True
        except TypeError:
            raw = page.extract_text() or ""
            layout = False
        if len((raw or "").strip()) < 20:
            raw = page.extract_text() or ""
            layout = False
        text = clean_pdf_text(raw or "", layout=layout)
        if len(text) > 20:
            has_text = True
        pages.append({"page": i, "text": text})
    return pages, has_text


def _para_text(node: ElementTree.Element) -> str:
    """Concatenate runs in one <w:p>, honouring tabs and soft breaks."""
    out: list[str] = []
    for child in node.iter():
        tag = child.tag
        if tag == W_NS + "t":
            out.append(child.text or "")
        elif tag == W_NS + "tab":
            out.append("\t")
        elif tag in (W_NS + "br", W_NS + "cr"):
            out.append("\n")
    return "".join(out)


def _block_lines(node: ElementTree.Element) -> list[str]:
    """Walk body children in document order. Table rows stay on one line."""
    lines: list[str] = []
    for child in node:
        tag = child.tag
        if tag == W_NS + "p":
            lines.append(_para_text(child))
        elif tag == W_NS + "tbl":
            for row in child.findall(W_NS + "tr"):
                cells = []
                for cell in row.findall(W_NS + "tc"):
                    cell_text = " ".join(
                        t for t in (_para_text(p) for p in cell.findall(W_NS + "p")) if t.strip()
                    )
                    cells.append(cell_text.strip())
                if any(cells):
                    lines.append(" | ".join(cells))
        elif tag == W_NS + "sdt":
            content = child.find(W_NS + "sdtContent")
            if content is not None:
                lines.extend(_block_lines(content))
    return lines


def extract_docx(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Return a single page record for a .docx. No page count exists pre-render."""
    try:
        with zipfile.ZipFile(path) as zf:
            with zf.open("word/document.xml") as fh:
                tree = ElementTree.parse(fh)
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, OSError):
        return [{"page": 1, "text": ""}], False
    body = tree.getroot().find(W_NS + "body")
    if body is None:
        return [{"page": 1, "text": ""}], False
    lines = _block_lines(body)
    text = "\n".join(line.rstrip() for line in lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return [{"page": 1, "text": text}], len(text) > 20


def extract_text_file(path: Path) -> list[dict[str, Any]]:
    return [{"page": 1, "text": path.read_text(encoding="utf-8", errors="replace")}]


def extract_pages(path: Path) -> tuple[list[dict[str, Any]], bool, str]:
    """Dispatch on suffix. Returns (pages, has_text_layer, kind)."""
    suffix = path.suffix.lower()
    if suffix in PDF_EXTS:
        pages, has_text = extract_pdf(path)
        return pages, has_text, "pdf"
    if suffix in DOCX_EXTS:
        pages, has_text = extract_docx(path)
        return pages, has_text, "docx"
    pages = extract_text_file(path)
    return pages, bool(pages[0]["text"].strip()), "text"
