"""Phase 2 safety rails. Pure stdlib - no FastAPI, no model, no network."""

from __future__ import annotations

import json
from pathlib import Path

from gateway.packs.files_write import (
    ALLOWED_EXTS,
    MAX_WRITE_BYTES,
    confirm_pending,
    fs_edit,
    fs_mkdir,
    fs_write,
    read_audit,
)


# ------------------------------------------------------------------ permission


def test_write_denied_by_default(tmp_path: Path):
    out = fs_write(tmp_path, "a.md", "hello", permission="deny")
    assert out["error"] == "permission_denied"
    assert not (tmp_path / "a.md").exists()


def test_edit_and_mkdir_also_respect_deny(tmp_path: Path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    assert fs_edit(tmp_path, "a.md", "x", "y", permission="deny")["error"] == "permission_denied"
    assert fs_mkdir(tmp_path, "sub", permission="deny")["error"] == "permission_denied"
    assert (tmp_path / "a.md").read_text(encoding="utf-8") == "x"
    assert not (tmp_path / "sub").exists()


def test_unknown_permission_is_refused(tmp_path: Path):
    out = fs_write(tmp_path, "a.md", "hi", permission="yolo")
    assert out["error"] == "permission_denied"


# ------------------------------------------------------------------- sandbox


def test_path_escape_creates_nothing(tmp_path: Path):
    root = tmp_path / "ws"
    root.mkdir()
    for rel in ("../outside.txt", "..\\outside.txt", "../../outside.txt"):
        out = fs_write(root, rel, "escaped", permission="allow")
        assert out["error"] == "path_escape", rel
    assert not (tmp_path / "outside.txt").exists()
    assert list(root.iterdir()) == []


def test_absolute_path_is_confined_not_escaped(tmp_path: Path):
    """An absolute POSIX path is re-rooted inside the workspace, never followed.

    The invariant that matters is 'nothing lands outside root', and the caller
    is told the real relative path it landed on.
    """
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "evil.txt"
    out = fs_write(root, str(outside), "x", permission="allow")
    assert not outside.exists(), "absolute path must not escape the workspace"
    if out.get("ok"):
        landed = root / out["path"]
        assert landed.is_file()
        assert root in landed.parents
        assert not Path(out["path"]).is_absolute()
    else:
        assert out["error"] in {"path_escape", "bad_extension", "blocked_path"}


def test_drive_letter_and_unc_are_rejected(tmp_path: Path):
    for rel in ("C:\\evil.txt", "c:/evil.txt", "\\\\server\\share\\evil.txt", "//server/share/e.txt"):
        out = fs_write(tmp_path, rel, "x", permission="allow")
        assert out["error"] == "path_escape", rel


def test_windows_traversal_blocked_on_every_os(tmp_path: Path):
    """A backslash path is a real separator on Windows and a legal filename on
    Linux. Normalising up front makes the same model output mean the same thing."""
    root = tmp_path / "ws"
    root.mkdir()
    out = fs_write(root, "..\\outside.txt", "escaped", permission="allow")
    assert out["error"] == "path_escape"
    assert not (tmp_path / "outside.txt").exists()
    assert list(root.iterdir()) == [], "no literal '..\\outside.txt' file either"


def test_reserved_device_names_rejected(tmp_path: Path):
    for rel in ("NUL.txt", "con.md", "COM1.txt", "sub/prn.txt"):
        out = fs_write(tmp_path, rel, "x", permission="allow")
        assert out["error"] == "path_escape", rel


def test_blocked_directories(tmp_path: Path):
    for rel in (".agentforge/x.md", ".git/x.md", ".venv/x.md", "node_modules/x.md"):
        out = fs_write(tmp_path, rel, "x", permission="allow")
        assert out["error"] == "blocked_path", rel
    assert not (tmp_path / ".agentforge" / "x.md").exists()


def test_cannot_write_workspace_root(tmp_path: Path):
    assert fs_write(tmp_path, ".", "x", permission="allow")["error"] == "bad_path"
    assert fs_write(tmp_path, "", "x", permission="allow")["error"] == "bad_path"


# ---------------------------------------------------------------- extensions


def test_extension_allowlist(tmp_path: Path):
    for ext in sorted(ALLOWED_EXTS):
        out = fs_write(tmp_path, f"ok{ext}", "content", permission="allow")
        assert out.get("ok"), out
    for bad in ("resume.pdf", "resume.docx", "run.exe", "run.sh", "lib.py", "noext"):
        out = fs_write(tmp_path, bad, "x", permission="allow")
        assert out["error"] == "bad_extension", bad
        assert not (tmp_path / bad).exists()


def test_pdf_and_docx_are_read_only(tmp_path: Path):
    pdf = tmp_path / "resume.pdf"
    pdf.write_bytes(b"%PDF-1.4 original")
    out = fs_edit(tmp_path, "resume.pdf", "original", "hacked", permission="allow")
    assert out["error"] == "bad_extension"
    assert pdf.read_bytes() == b"%PDF-1.4 original"


# ---------------------------------------------------------------------- size


def test_size_cap(tmp_path: Path):
    out = fs_write(tmp_path, "big.txt", "a" * (MAX_WRITE_BYTES + 1), permission="allow")
    assert out["error"] == "too_large"
    assert not (tmp_path / "big.txt").exists()


def test_size_cap_counts_utf8_bytes(tmp_path: Path):
    # 3-byte characters: half the char count still blows the byte budget.
    payload = "€" * (MAX_WRITE_BYTES // 2)
    assert fs_write(tmp_path, "big.txt", payload, permission="allow")["error"] == "too_large"


# ------------------------------------------------------------------- fs_write


def test_create_then_refuse_duplicate(tmp_path: Path):
    first = fs_write(tmp_path, "summary.md", "# One\n", permission="allow")
    assert first["ok"] and first["created"]
    assert (tmp_path / "summary.md").read_text(encoding="utf-8") == "# One\n"
    second = fs_write(tmp_path, "summary.md", "# Two\n", permission="allow")
    assert second["error"] == "exists"
    assert (tmp_path / "summary.md").read_text(encoding="utf-8") == "# One\n"


def test_overwrite_takes_a_backup(tmp_path: Path):
    fs_write(tmp_path, "n.md", "original\n", permission="allow")
    out = fs_write(tmp_path, "n.md", "replaced\n", mode="overwrite", permission="allow")
    assert out["ok"]
    assert (tmp_path / "n.md").read_text(encoding="utf-8") == "replaced\n"
    assert out["backup"], "overwrite must snapshot the previous bytes"
    assert (tmp_path / out["backup"]).read_text(encoding="utf-8") == "original\n"


def test_nested_create_makes_parents(tmp_path: Path):
    out = fs_write(tmp_path, "reports/2026/q1.md", "hi", permission="allow")
    assert out["ok"]
    assert (tmp_path / "reports" / "2026" / "q1.md").is_file()


def test_bad_mode_rejected(tmp_path: Path):
    assert fs_write(tmp_path, "a.md", "x", mode="append", permission="allow")["error"] == "bad_mode"


# -------------------------------------------------------------------- fs_edit


def test_edit_exact_single_match(tmp_path: Path):
    p = tmp_path / "README_resume_usage_notes.md"
    p.write_text("Owner: X\nRole: Automation Lead\n", encoding="utf-8")
    out = fs_edit(tmp_path, p.name, "Owner: X", "Owner: Y", permission="allow")
    assert out["ok"] and out["replacements"] == 1
    assert p.read_text(encoding="utf-8") == "Owner: Y\nRole: Automation Lead\n"


def test_edit_no_match(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text("hello\n", encoding="utf-8")
    out = fs_edit(tmp_path, "a.md", "nope", "yes", permission="allow")
    assert out["error"] == "no_match"
    assert p.read_text(encoding="utf-8") == "hello\n"


def test_edit_ambiguous_match_changes_nothing(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text("x\nx\nx\n", encoding="utf-8")
    out = fs_edit(tmp_path, "a.md", "x", "y", permission="allow")
    assert out["error"] == "ambiguous_match" and out["count"] == 3
    assert p.read_text(encoding="utf-8") == "x\nx\nx\n"


def test_edit_is_idempotent_on_rerun(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text("Status: DRAFT\n", encoding="utf-8")
    first = fs_edit(tmp_path, "a.md", "DRAFT", "FINAL", permission="allow")
    assert first["ok"] and first["replacements"] == 1
    second = fs_edit(tmp_path, "a.md", "DRAFT", "FINAL", permission="allow")
    assert second["ok"] and second.get("already_applied") is True
    assert p.read_text(encoding="utf-8") == "Status: FINAL\n"


def test_edit_missing_file_and_empty_old_text(tmp_path: Path):
    assert fs_edit(tmp_path, "ghost.md", "a", "b", permission="allow")["error"] == "wrong_file"
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    assert fs_edit(tmp_path, "a.md", "", "b", permission="allow")["error"] == "bad_args"


# ------------------------------------------------------------------- fs_mkdir


def test_mkdir_creates_and_is_idempotent(tmp_path: Path):
    first = fs_mkdir(tmp_path, "out/reports", permission="allow")
    assert first["ok"] and first["created"]
    assert (tmp_path / "out" / "reports").is_dir()
    second = fs_mkdir(tmp_path, "out/reports", permission="allow")
    assert second["ok"] and second["created"] is False


def test_mkdir_blocked_and_escaping(tmp_path: Path):
    assert fs_mkdir(tmp_path, "../evil", permission="allow")["error"] == "path_escape"
    assert fs_mkdir(tmp_path, ".git/hooks", permission="allow")["error"] == "blocked_path"


# ----------------------------------------------------------------- ask / diff


def test_ask_mode_stages_without_writing(tmp_path: Path):
    out = fs_write(tmp_path, "s.md", "# staged\n", permission="ask")
    assert out["needs_confirm"] is True
    assert out["token"] and "+# staged" in out["diff"]
    assert not (tmp_path / "s.md").exists(), "ask mode must not touch disk"

    applied = confirm_pending(out["token"], approve=True)
    assert applied["ok"] and applied["applied"]
    assert (tmp_path / "s.md").read_text(encoding="utf-8") == "# staged\n"


def test_ask_mode_reject_discards(tmp_path: Path):
    out = fs_write(tmp_path, "s.md", "nope", permission="ask")
    res = confirm_pending(out["token"], approve=False)
    assert res["ok"] and res["applied"] is False
    assert not (tmp_path / "s.md").exists()


def test_token_cannot_be_replayed(tmp_path: Path):
    out = fs_write(tmp_path, "s.md", "one", permission="ask")
    assert confirm_pending(out["token"])["applied"]
    again = confirm_pending(out["token"])
    assert again["error"] == "unknown_token"


def test_ask_mode_edit_produces_a_real_diff(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text("Status: DRAFT\n", encoding="utf-8")
    out = fs_edit(tmp_path, "a.md", "DRAFT", "FINAL", permission="ask")
    assert out["needs_confirm"] is True
    assert "-Status: DRAFT" in out["diff"] and "+Status: FINAL" in out["diff"]
    assert p.read_text(encoding="utf-8") == "Status: DRAFT\n"
    confirm_pending(out["token"])
    assert p.read_text(encoding="utf-8") == "Status: FINAL\n"


# ----------------------------------------------------------------- audit log


def test_audit_log_records_every_write(tmp_path: Path):
    fs_write(tmp_path, "a.md", "one", permission="allow")
    fs_write(tmp_path, "a.md", "two", mode="overwrite", permission="allow")
    fs_edit(tmp_path, "a.md", "two", "three", permission="allow")
    fs_mkdir(tmp_path, "sub", permission="allow")

    log = tmp_path / ".agentforge" / "write_audit.jsonl"
    assert log.exists()
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]
    actions = [r["action"] for r in rows]
    assert actions == ["write", "write", "edit", "mkdir"]
    assert all(r.get("ts") for r in rows)
    assert read_audit(tmp_path)[-1]["action"] == "mkdir"


def test_audit_not_written_for_refused_writes(tmp_path: Path):
    fs_write(tmp_path, "../escape.txt", "x", permission="allow")
    fs_write(tmp_path, "x.pdf", "x", permission="allow")
    assert not (tmp_path / ".agentforge" / "write_audit.jsonl").exists()
