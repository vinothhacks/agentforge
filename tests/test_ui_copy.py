from pathlib import Path

from fastapi.testclient import TestClient

from gateway.app import make_app
from gateway.db import Store


def test_ui_has_no_pda_and_shows_llmfit() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "gateway" / "static" / "index.html").read_text(encoding="utf-8")
    skill = (root / "templates" / "document-query" / "SKILL.md").read_text(encoding="utf-8")
    assert "PDA" not in html
    assert "PDA" not in skill
    assert "llmfit" in html
    assert "modelPick" in html
    assert "/api/models" in html
    assert "/api/download/file" in html
    assert "/api/export/chat" in html


def test_scan_files_and_downloads(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello workspace", encoding="utf-8")
    app = make_app(tmp_path, Store(tmp_path / "app.sqlite"))
    client = TestClient(app)
    scan = client.get("/api/scan").json()
    assert "llmfit" in scan
    assert scan["llmfit_pypi"].startswith("https://")
    assert "ollama.com/download" in scan["ollama_install"]
    files = client.get("/api/files").json()["files"]
    assert any(f["path"] == "note.txt" for f in files)
    dl = client.get("/api/download/file", params={"path": "note.txt"})
    assert dl.status_code == 200
    assert dl.content == b"hello workspace"
    escaped = client.get("/api/download/file", params={"path": "..\\secret.txt"})
    assert escaped.status_code in {400, 404}
    chat = client.get("/api/export/chat")
    assert chat.status_code == 200
    assert "attachment" in chat.headers.get("content-disposition", "").lower()
