"""Live OpenRouter chat against the advanced corpus. Key from env only."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from gateway.app import make_app
from gateway.db import Store

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "advanced"
QUESTIONS = [
        "List every generated file that mentions demurrage. Cite file names.",
    "What is the TOTAL estimate and berth window for vessel MSC AURORA?",
    "Which documents mention Paldiski South Harbour?",
]


def main() -> int:
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key.startswith("sk-or-"):
        print("skip live chat: no OPENROUTER_API_KEY")
        return 0
    store = Store(ROOT / ".agentforge" / "app.sqlite")
    app = make_app(ROOT, store)
    client = TestClient(app)
    rows = []
    client.post("/api/key", json={"key": key})
    for q in QUESTIONS:
        r = client.post("/api/chat", json={"message": q, "session_id": "advanced-live"})
        data = r.json() if "application/json" in r.headers.get("content-type", "") else {}
        tools = [t.get("tool") for t in (data.get("traces") or []) if t.get("kind") == "tool"]
        enum = (data.get("eval") or {}).get("enumeration")
        rows.append(
            {
                "status": r.status_code,
                "question": q,
                "tools": tools,
                "eval": enum,
                "text": (data.get("text") or data.get("detail") or "")[:600],
            }
        )
        print(f"status={r.status_code} tools={tools}")
        print((data.get("text") or str(data)[:200])[:400])
        print("---")
    dest = ROOT / "LIVE_CHAT.json"
    dest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    bad = [x for x in rows if x["status"] != 200 or not x["text"]]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
