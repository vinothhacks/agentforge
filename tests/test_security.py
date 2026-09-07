"""Phase 0 regression tests: the three routes that leaked the OpenRouter key.

Before this, `GET /api/download/file?path=.agentforge/app.sqlite` returned the
store -- which holds the key in plaintext -- and `allow_origins=["*"]` meant any
page in the user's browser could ask for it.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from gateway.app import make_app
from gateway.db import Store
from gateway.packs.files import fs_list, fs_read
from gateway.packs.files_write import confirm_pending, fs_write
from gateway.paths import blocked_component

EVIL = "https://evil.example"


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / "note.txt").write_text("hello workspace", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-or-secret", encoding="utf-8")
    home = tmp_path / ".agentforge"
    home.mkdir(exist_ok=True)
    (home / "app.sqlite").write_text("pretend-sqlite-with-a-key", encoding="utf-8")
    return tmp_path


# ------------------------------------------------------------------ read tools


def test_fs_read_refuses_the_store(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    out = fs_read(ws, ".agentforge/app.sqlite")
    assert out["error"] == "blocked_path"
    assert "sk-or-secret" not in str(out)


def test_fs_read_refuses_dotfiles(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    for rel in (".env", ".git/config", ".venv/pyvenv.cfg"):
        assert fs_read(ws, rel)["error"] == "blocked_path", rel


def test_fs_read_still_reads_ordinary_files(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    assert fs_read(ws, "note.txt")["text"] == "hello workspace"


def test_fs_list_refuses_a_protected_target(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    assert fs_list(ws, ".agentforge")["error"] == "blocked_path"


def test_fs_list_hides_protected_children(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    paths = fs_list(ws, ".", recursive=True)["paths"]
    assert "note.txt" in paths
    assert not any(p.startswith(".") for p in paths)


def test_blocked_component_write_side_still_allows_dotfiles() -> None:
    # The write side passes hidden=False, so an explicit dotfile write is legal
    # while .git / .agentforge stay protected.
    assert blocked_component((".env",)) is None
    assert blocked_component((".env",), hidden=True) == ".env"
    assert blocked_component(("docs", ".git", "config")) == ".git"


# -------------------------------------------------------------------- download


def test_download_refuses_the_store(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    client = TestClient(make_app(ws, Store(ws / ".agentforge" / "state.sqlite")))
    r = client.get("/api/download/file", params={"path": ".agentforge/app.sqlite"})
    assert r.status_code == 400
    assert b"pretend-sqlite" not in r.content


def test_download_refuses_dotfiles_but_serves_real_ones(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    client = TestClient(make_app(ws, Store(ws / ".agentforge" / "state.sqlite")))
    assert client.get("/api/download/file", params={"path": ".env"}).status_code == 400
    ok = client.get("/api/download/file", params={"path": "note.txt"})
    assert ok.status_code == 200
    assert ok.content == b"hello workspace"


# ------------------------------------------------------------------------ CORS


def test_cross_origin_request_gets_no_allow_header(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    client = TestClient(make_app(ws, Store(ws / ".agentforge" / "state.sqlite"), port=8788))
    r = client.get("/api/health", headers={"Origin": EVIL})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_cross_origin_preflight_is_rejected(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    client = TestClient(make_app(ws, Store(ws / ".agentforge" / "state.sqlite"), port=8788))
    r = client.options(
        "/api/permissions",
        headers={
            "Origin": EVIL,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.headers.get("access-control-allow-origin") != EVIL


def test_own_origin_is_still_allowed(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    client = TestClient(make_app(ws, Store(ws / ".agentforge" / "state.sqlite"), port=9999))
    r = client.get("/api/health", headers={"Origin": "http://127.0.0.1:9999"})
    assert r.headers.get("access-control-allow-origin") == "http://127.0.0.1:9999"


# ----------------------------------------------------------------- confirm token


def test_confirm_token_is_bound_to_its_workspace(tmp_path: Path) -> None:
    mine = tmp_path / "mine"
    theirs = tmp_path / "theirs"
    mine.mkdir()
    theirs.mkdir()
    staged = fs_write(mine, "notes.md", "secret plan", permission="ask")
    assert staged["needs_confirm"]

    denied = confirm_pending(staged["token"], approve=True, workspace=theirs)
    assert denied["error"] == "wrong_workspace"
    assert not (mine / "notes.md").exists()


def test_confirm_token_applies_in_its_own_workspace(tmp_path: Path) -> None:
    staged = fs_write(tmp_path, "notes.md", "secret plan", permission="ask")
    applied = confirm_pending(staged["token"], approve=True, workspace=tmp_path)
    assert applied["applied"] is True
    assert (tmp_path / "notes.md").read_text(encoding="utf-8") == "secret plan"


# ---------------------------------------------------------------------- the key


def test_key_can_be_cleared_and_health_reports_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    ws = _workspace(tmp_path)
    client = TestClient(make_app(ws, Store(ws / ".agentforge" / "state.sqlite")))

    assert client.get("/api/health").json()["has_key"] is False
    assert client.post("/api/key", json={"key": "sk-or-abc"}).json()["has_key"] is True
    assert client.get("/api/health").json()["has_key"] is True
    assert client.post("/api/key", json={"key": "  "}).json()["cleared"] is True
    assert client.get("/api/health").json()["has_key"] is False
