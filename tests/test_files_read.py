"""Phase 1: fs_list glob/count/recursion and fs_read for text, PDF and DOCX."""

from __future__ import annotations

import zipfile
from pathlib import Path

from gateway.packs.extract import extract_docx
from gateway.packs.files import fs_list, fs_read

DOCX_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Vinoth N</w:t></w:r></w:p>
    <w:p><w:r><w:t>GenAI Engineer</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>2019-2026</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Skill</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Python</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>
"""


def make_docx(path: Path, xml: str = DOCX_XML) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", xml)
    return path


def build_workspace(root: Path) -> Path:
    (root / "a_Resume.pdf").write_bytes(b"%PDF-1.4 stub")
    (root / "b_Resume.pdf").write_bytes(b"%PDF-1.4 stub")
    (root / "c_Resume.pdf").write_bytes(b"%PDF-1.4 stub")
    make_docx(root / "Vinoth_Resume.docx")
    (root / "README_resume_usage_notes.md").write_text("Use the PDF for ATS.", encoding="utf-8")
    (root / "notes.txt").write_text("Status: DRAFT", encoding="utf-8")
    sub = root / "archive"
    sub.mkdir()
    (sub / "old_Resume.pdf").write_bytes(b"%PDF-1.4 stub")
    hidden = root / ".agentforge"
    hidden.mkdir(exist_ok=True)
    (hidden / "chunks.jsonl").write_text("{}", encoding="utf-8")
    return root


# --------------------------------------------------------------------- fs_list


def test_list_skips_internal_dirs(tmp_path: Path):
    build_workspace(tmp_path)
    out = fs_list(tmp_path)
    names = {e["name"] for e in out["entries"]}
    assert ".agentforge" not in names
    assert "a_Resume.pdf" in names and "archive" in names


def test_glob_count_is_exact(tmp_path: Path):
    """The Q10 regression: the model must read count, not tally a list."""
    build_workspace(tmp_path)
    out = fs_list(tmp_path, glob="*.pdf")
    assert out["count"] == 3
    assert out["file_count"] == 3
    assert sorted(out["paths"]) == ["a_Resume.pdf", "b_Resume.pdf", "c_Resume.pdf"]


def test_glob_recursive_counts_subfolders(tmp_path: Path):
    build_workspace(tmp_path)
    out = fs_list(tmp_path, glob="*.pdf", recursive=True)
    assert out["count"] == 4
    assert "archive/old_Resume.pdf" in out["paths"]


def test_glob_never_counts_directories(tmp_path: Path):
    build_workspace(tmp_path)
    assert fs_list(tmp_path, glob="*.docx")["count"] == 1
    assert fs_list(tmp_path, glob="*")["dir_count"] == 0


def test_bare_extension_forms_are_tolerated(tmp_path: Path):
    build_workspace(tmp_path)
    for pattern in ("*.pdf", ".pdf", "pdf", "*.PDF"):
        assert fs_list(tmp_path, glob=pattern)["count"] == 3, pattern


def test_substring_glob(tmp_path: Path):
    build_workspace(tmp_path)
    # Matching is case-insensitive on every OS so the same model output gives
    # the same answer on Windows and Linux. README_resume_usage_notes.md is a
    # legitimate hit for *Resume*.
    out = fs_list(tmp_path, glob="*Resume*")
    assert out["count"] == 5
    assert "README_resume_usage_notes.md" in out["paths"]


def test_list_path_escape(tmp_path: Path):
    root = tmp_path / "ws"
    root.mkdir()
    assert fs_list(root, "../")["error"] == "path_escape"


def test_list_missing_dir(tmp_path: Path):
    assert fs_list(tmp_path, "nope")["error"] == "wrong_file"


# --------------------------------------------------------------------- fs_read


def test_read_text_and_truncation(tmp_path: Path):
    (tmp_path / "a.md").write_text("x" * 5000, encoding="utf-8")
    out = fs_read(tmp_path, "a.md", max_chars=1000)
    assert out["truncated"] is True and len(out["text"]) == 1000
    assert out["kind"] == "text"


def test_read_docx_returns_text(tmp_path: Path):
    make_docx(tmp_path / "r.docx")
    out = fs_read(tmp_path, "r.docx")
    assert out["kind"] == "docx"
    assert "Vinoth N" in out["text"]
    assert "GenAI Engineer" in out["text"]
    assert "Skill | Python" in out["text"], "table rows should stay on one line"


def test_read_empty_docx_reports_no_text_layer(tmp_path: Path):
    empty = """<?xml version="1.0"?><w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body/></w:document>"""
    make_docx(tmp_path / "blank.docx", empty)
    assert fs_read(tmp_path, "blank.docx")["error"] == "no_text_layer"


def test_corrupt_docx_does_not_raise(tmp_path: Path):
    (tmp_path / "bad.docx").write_bytes(b"not a zip")
    out = fs_read(tmp_path, "bad.docx")
    assert out["error"] == "no_text_layer"
    pages, has_text = extract_docx(tmp_path / "bad.docx")
    assert has_text is False and pages[0]["text"] == ""


def test_read_path_escape_and_missing(tmp_path: Path):
    root = tmp_path / "ws"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")
    assert fs_read(root, "../secret.txt")["error"] == "path_escape"
    assert fs_read(root, "ghost.md")["error"] == "wrong_file"
