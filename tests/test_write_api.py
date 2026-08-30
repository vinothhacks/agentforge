"""End-to-end through FastAPI: permission toggle, staged confirm, audit, files."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from gateway.app import make_app
from gateway.db import Store
from gateway.packs.files_write import fs_write


def client_for(tmp_path: Path) -> TestClient:
    return TestClient(make_app(tmp_path, Store(tmp_path / "app.sqlite")))


def test_permissions_roundtrip(tmp_path: Path):
    c = client_for(tmp_path)
    body = c.get("/api/permissions").json()
    assert body["fs_write"] in {"deny", "ask", "allow"}
    assert ".md" in body["allowed_extensions"]
    assert ".pdf" not in body["allowed_extensions"]

    assert c.post("/api/permissions", json={"fs_write": "ask"}).json()["fs_write"] == "ask"
    assert c.get("/api/permissions").json()["fs_write"] == "ask"
    assert c.get("/api/health").json()["fs_write"] == "ask"

    assert c.post("/api/permissions", json={"fs_write": "rm -rf"}).status_code == 400


def test_confirm_endpoint_applies_staged_write(tmp_path: Path):
    c = client_for(tmp_path)
    staged = fs_write(tmp_path, "plan.md", "# Plan\n", permission="ask")
    assert not (tmp_path / "plan.md").exists()

    res = c.post("/api/confirm", json={"token": staged["token"], "approve": True}).json()
    assert res["applied"] is True
    assert (tmp_path / "plan.md").read_text(encoding="utf-8") == "# Plan\n"

    # Replay must fail, not write twice.
    assert c.post("/api/confirm", json={"token": staged["token"]}).status_code == 409


def test_confirm_unknown_token(tmp_path: Path):
    c = client_for(tmp_path)
    assert c.post("/api/confirm", json={"token": "deadbeef"}).status_code == 409


def test_audit_endpoint(tmp_path: Path):
    c = client_for(tmp_path)
    fs_write(tmp_path, "a.md", "x", permission="allow")
    entries = c.get("/api/audit").json()["entries"]
    assert entries and entries[-1]["action"] == "write"


def test_files_endpoint_marks_writable_and_shows_new_files(tmp_path: Path):
    (tmp_path / "resume.pdf").write_bytes(b"%PDF-1.4")
    c = client_for(tmp_path)
    before = {f["path"] for f in c.get("/api/files").json()["files"]}
    assert "resume.pdf" in before

    fs_write(tmp_path, "summary.md", "# Summary\n", permission="allow")
    rows = {f["path"]: f for f in c.get("/api/files").json()["files"]}
    assert "summary.md" in rows, "a freshly written file must appear without a restart"
    assert rows["summary.md"]["writable"] is True
    assert rows["resume.pdf"]["writable"] is False


def test_files_endpoint_hides_backups_and_internals(tmp_path: Path):
    fs_write(tmp_path, "a.md", "one", permission="allow")
    fs_write(tmp_path, "a.md", "two", mode="overwrite", permission="allow")
    c = client_for(tmp_path)
    paths = [f["path"] for f in c.get("/api/files").json()["files"]]
    assert paths == ["a.md"], f"backups leaked into the file list: {paths}"
