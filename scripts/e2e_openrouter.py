"""Live OpenRouter E2E: start-to-end against a running AgentForge gateway.

Reads OPENROUTER_API_KEY from the environment. Does not print the key.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import httpx

BASE = os.environ.get("AGENTFORGE_URL", "http://127.0.0.1:8788")
MODEL = os.environ.get("AGENTFORGE_MODEL", "openai/gpt-4o-mini")
SESSION = os.environ.get("AGENTFORGE_SESSION", "e2e-live")


def _ok(name: str, cond: bool, detail: str = "") -> dict[str, Any]:
    row = {"step": name, "pass": bool(cond), "detail": detail[:800]}
    flag = "PASS" if cond else "FAIL"
    print(f"[{flag}] {name}: {detail[:240]}")
    return row


def main() -> int:
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key.startswith("sk-or-"):
        print("OPENROUTER_API_KEY is missing or not an OpenRouter key", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    timeout = httpx.Timeout(180.0, connect=15.0)
    with httpx.Client(base_url=BASE, timeout=timeout) as c:
        t0 = time.perf_counter()
        health = c.get("/api/health")
        rows.append(
            _ok(
                "health",
                health.status_code == 200 and health.json().get("ok") is True,
                json.dumps(health.json()),
            )
        )

        ui = c.get("/")
        rows.append(_ok("ui", ui.status_code == 200 and "AgentForge" in ui.text, f"status={ui.status_code}"))

        later = c.get("/api/later")
        later_json = later.json()
        rows.append(
            _ok(
                "later_list",
                later.status_code == 200 and later_json.get("v1_tools") == ["rag_search", "fs_list", "fs_read"],
                json.dumps(later_json)[:400],
            )
        )
        stub = c.get("/api/later/excel_write")
        rows.append(_ok("later_excel_501", stub.status_code == 501, f"status={stub.status_code}"))

        saved = c.post("/api/key", json={"key": key})
        rows.append(_ok("save_key", saved.status_code == 200 and saved.json().get("ok") is True, saved.text))

        ingest = c.post("/api/ingest")
        inj = ingest.json()
        rows.append(
            _ok(
                "ingest",
                ingest.status_code == 200 and inj.get("files_scanned", 0) >= 12,
                json.dumps(inj),
            )
        )

        print("probing openai/gpt-4o-mini (fast, n=3)...")
        probe = c.post("/api/probe", json={"provider": "openrouter", "name": MODEL, "mode": "fast"})
        pj = probe.json() if probe.headers.get("content-type", "").startswith("application/json") else {"raw": probe.text}
        measured = pj.get("measured") or {}
        rows.append(
            _ok(
                "probe",
                probe.status_code == 200 and pj.get("verdict") == "agent",
                json.dumps({k: pj.get(k) for k in ("verdict", "cause", "measured", "model_pin")}),
            )
        )

        q1 = "List every file mentioning demurrage"
        print(f"chat: {q1}")
        chat1 = c.post("/api/chat", json={"message": q1, "session_id": SESSION})
        c1 = chat1.json() if chat1.status_code == 200 else {"error": chat1.text, "status": chat1.status_code}
        tools1 = [t.get("tool") for t in (c1.get("traces") or []) if t.get("kind") == "tool"]
        enum = (c1.get("eval") or {}).get("enumeration") or {}
        text1 = c1.get("text") or ""
        demurrage_ok = (
            chat1.status_code == 200
            and "rag_search" in tools1
            and enum.get("recall", 0) >= 0.90
            and ("DOC-001" in text1 or "NEPTUNE" in text1 or "demurrage" in text1.lower())
        )
        rows.append(
            _ok(
                "chat_enumeration",
                demurrage_ok,
                json.dumps(
                    {
                        "status": chat1.status_code,
                        "tools": tools1,
                        "eval": enum,
                        "stop": (c1.get("usage") or {}).get("stop"),
                        "text": text1[:500],
                    }
                ),
            )
        )

        q2 = "What is the account total and berth window in file DOC-001-NEPTUNE.txt?"
        print(f"chat: {q2}")
        chat2 = c.post("/api/chat", json={"message": q2, "session_id": SESSION})
        c2 = chat2.json() if chat2.status_code == 200 else {"error": chat2.text, "status": chat2.status_code}
        tools2 = [t.get("tool") for t in (c2.get("traces") or []) if t.get("kind") == "tool"]
        text2 = c2.get("text") or ""
        amount_ok = chat2.status_code == 200 and ("15000" in text2 or "15,000" in text2)
        rows.append(
            _ok(
                "chat_doc001_amount",
                amount_ok,
                json.dumps({"status": chat2.status_code, "tools": tools2, "text": text2[:500]}),
            )
        )

        q3 = "List the files in the workspace folder."
        print(f"chat: {q3}")
        chat3 = c.post("/api/chat", json={"message": q3, "session_id": SESSION})
        c3 = chat3.json() if chat3.status_code == 200 else {"error": chat3.text, "status": chat3.status_code}
        tools3 = [t.get("tool") for t in (c3.get("traces") or []) if t.get("kind") == "tool"]
        text3 = c3.get("text") or ""
        list_ok = chat3.status_code == 200 and ("fs_list" in tools3 or "DOC-" in text3)
        rows.append(
            _ok(
                "chat_fs_list",
                list_ok,
                json.dumps({"status": chat3.status_code, "tools": tools3, "text": text3[:500]}),
            )
        )

        usage = c.get("/api/usage", params={"session_id": SESSION})
        uj = usage.json()
        rows.append(
            _ok(
                "usage_meter",
                usage.status_code == 200 and (uj.get("prompt_tokens") or 0) > 0,
                json.dumps(uj),
            )
        )

        traces = c.get("/api/traces", params={"session_id": SESSION})
        tj = traces.json()
        kinds = [t.get("kind") for t in tj] if isinstance(tj, list) else []
        rows.append(
            _ok(
                "traces",
                traces.status_code == 200 and "eval" in kinds and "tool" in kinds,
                f"n={len(tj) if isinstance(tj, list) else 0} kinds={sorted(set(kinds))}",
            )
        )

        scan = c.get("/api/scan")
        rows.append(_ok("scan_optional", scan.status_code == 200, json.dumps(scan.json())[:300]))

        elapsed = round(time.perf_counter() - t0, 1)

    failed = [r for r in rows if not r["pass"]]
    print()
    print(f"elapsed_s={elapsed}  passed={len(rows) - len(failed)}/{len(rows)}")
    out = Path_results(rows, elapsed, MODEL)
    print(f"wrote {out}")
    return 1 if failed else 0


def Path_results(rows: list[dict[str, Any]], elapsed: float, model: str) -> str:
    from pathlib import Path

    dest = Path("scripts") / "e2e_openrouter_last.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(
        json.dumps({"model": model, "elapsed_s": elapsed, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    return str(dest)


if __name__ == "__main__":
    raise SystemExit(main())
