"""SQLite app DB: sessions, traces, cards, settings. FTS5 lives in the rag pack.

The OpenRouter key is stored here in plaintext -- there is no encryption and
this docstring used to claim otherwise. The file therefore never leaves the
machine: it lives under .agentforge, which the read tools and the download
endpoint both refuse (see gateway/paths.py::BLOCKED_PARTS).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    messages_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cards (
    digest TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    name TEXT NOT NULL,
    json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS usage (
    session_id TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    usd REAL NOT NULL DEFAULT 0
);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def get_setting(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return None if row is None else row["value"]

    def set_setting(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def delete_setting(self, key: str) -> None:
        self.conn.execute("DELETE FROM settings WHERE key=?", (key,))
        self.conn.commit()

    def save_session(self, session_id: str, workspace: str, messages: list[dict[str, Any]]) -> None:
        now = time.time()
        payload = json.dumps(messages)
        self.conn.execute(
            """INSERT INTO sessions(id, workspace, created_at, updated_at, messages_json)
               VALUES(?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at, messages_json=excluded.messages_json""",
            (session_id, workspace, now, now, payload),
        )
        self.conn.commit()

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        row = self.conn.execute("SELECT messages_json FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            return []
        return json.loads(row["messages_json"])

    def add_trace(self, session_id: str, kind: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO traces(session_id, created_at, kind, payload_json) VALUES(?,?,?,?)",
            (session_id, time.time(), kind, json.dumps(payload)),
        )
        self.conn.commit()

    def traces(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT created_at, kind, payload_json FROM traces WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [
            {"created_at": r["created_at"], "kind": r["kind"], "payload": json.loads(r["payload_json"])}
            for r in rows
        ]

    def save_card(self, card: dict[str, Any]) -> None:
        digest = card.get("model_pin", {}).get("digest") or card.get("digest") or card["model_pin"]["name"]
        self.conn.execute(
            """INSERT INTO cards(digest, provider, name, json, created_at) VALUES(?,?,?,?,?)
               ON CONFLICT(digest) DO UPDATE SET json=excluded.json""",
            (
                digest,
                card["model_pin"]["provider"],
                card["model_pin"]["name"],
                json.dumps(card),
                time.time(),
            ),
        )
        self.conn.commit()

    def load_card(self, digest: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT json FROM cards WHERE digest=?", (digest,)).fetchone()
        return None if row is None else json.loads(row["json"])

    def add_usage(self, session_id: str, prompt: int, completion: int, usd: float) -> dict[str, float | int]:
        row = self.conn.execute("SELECT * FROM usage WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO usage(session_id, prompt_tokens, completion_tokens, usd) VALUES(?,?,?,?)",
                (session_id, prompt, completion, usd),
            )
        else:
            self.conn.execute(
                """UPDATE usage SET prompt_tokens=prompt_tokens+?, completion_tokens=completion_tokens+?, usd=usd+?
                   WHERE session_id=?""",
                (prompt, completion, usd, session_id),
            )
        self.conn.commit()
        return self.get_usage(session_id)

    def get_usage(self, session_id: str) -> dict[str, float | int]:
        row = self.conn.execute("SELECT * FROM usage WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
        return {
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "usd": row["usd"],
        }
