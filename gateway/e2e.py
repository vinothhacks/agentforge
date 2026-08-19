"""Live start-to-end path: ingest → probe → chat → eval, against a real model."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from httpx import ASGITransport, Client

from gateway.app import make_app
from gateway.db import Store
from gateway.evals import write_sample_pdas
from gateway.packs.rag import ingest_workspace

PREFERRED_LOCAL = ("ollama", "gemma4:cloud")
PREFERRED_CLOUD = ("openrouter", "openai/gpt-4o-mini")

QUESTIONS = [
    {
        "id": "enumeration",
        "message": "List every PDA mentioning demurrage",
        "require_tool": "rag_search",
    },
    {
        "id": "cite_amount",
        "message": "What is the PDA amount and berth window in file PDA-001-NEPTUNE.txt?",
        "require_any_tool": ("rag_search", "fs_read"),
        "needle": "15000",
    },
]


def installed_ollama() -> set[str]:
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    names: set[str] = set()
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            names.add(parts[0])
    return names


def pick_model(provider: str | None = None, name: str | None = None) -> tuple[str, str, str]:
    if provider and name:
        return provider, name, "cli"
    if os.environ.get("OPENROUTER_API_KEY"):
        return (*PREFERRED_CLOUD, "openrouter_key")
    tags = installed_ollama()
    if PREFERRED_LOCAL[1] in tags:
        return (*PREFERRED_LOCAL, "p1_agent_pass")
    if "qwen3:4b" in tags:
        return "ollama", "qwen3:4b", "installed_fallback"
    if tags:
        first = sorted(tags)[0]
        return "ollama", first, "first_installed"
    raise RuntimeError(
        "No model available. Set OPENROUTER_API_KEY or install Ollama "
        f"(preferred local tag: {PREFERRED_LOCAL[1]})."
    )


def _tools(traces: list[dict[str, Any]]) -> list[str]:
    return [t.get("tool") for t in traces if t.get("kind") == "tool" and t.get("tool")]


def run_e2e(
    workspace: Path,
    *,
    provider: str,
    name: str,
    probe_mode: str = "fast",
) -> dict[str, Any]:
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    write_sample_pdas(workspace)
    ingest_workspace(workspace)

    home = workspace / ".agentforge"
    home.mkdir(exist_ok=True)
    store = Store(home / "app.sqlite")
    app = make_app(workspace, store)
    steps: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    with Client(transport=ASGITransport(app=app), base_url="http://e2e.local", timeout=180.0) as client:
        health = client.get("/api/health")
        health.raise_for_status()
        body = health.json()
        steps.append(
            {
                "id": "health",
                "ok": bool(body.get("ok") and body.get("ingested")),
                "detail": body,
            }
        )

        later = client.get("/api/later/excel_write")
        steps.append(
            {
                "id": "later_stub",
                "ok": later.status_code == 501,
                "detail": {"status": later.status_code, "body": later.json()},
            }
        )

        ingest = client.post("/api/ingest")
        ingest.raise_for_status()
        ingested = ingest.json()
        steps.append(
            {
                "id": "ingest",
                "ok": ingested.get("files_scanned", 0) >= 12 and ingested.get("no_text_layer_count", 1) == 0,
                "detail": ingested,
            }
        )

        print(f"e2e probe {provider}:{name} mode={probe_mode} ...", flush=True)
        probe = client.post(
            "/api/probe",
            json={"provider": provider, "name": name, "mode": probe_mode},
        )
        probe_json = probe.json() if probe.headers.get("content-type", "").startswith("application/json") else {
            "detail": probe.text
        }
        probe_ok = probe.status_code == 200 and probe_json.get("verdict") == "agent"
        steps.append(
            {
                "id": "probe",
                "ok": probe_ok,
                "detail": {
                    "status": probe.status_code,
                    "verdict": probe_json.get("verdict"),
                    "cause": probe_json.get("cause"),
                    "measured": probe_json.get("measured"),
                    "model_pin": probe_json.get("model_pin"),
                },
            }
        )
        if not probe_ok:
            return _finish(workspace, provider, name, steps, t0, chats=[])

        chats: list[dict[str, Any]] = []
        for q in QUESTIONS:
            print(f"e2e chat {q['id']} ...", flush=True)
            resp = client.post(
                "/api/chat",
                json={"message": q["message"], "session_id": "e2e-live"},
            )
            payload = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {
                "detail": resp.text
            }
            tools = _tools(payload.get("traces") or [])
            text = payload.get("text") or ""
            enum = (payload.get("eval") or {}).get("enumeration")
            checks = {
                "http_ok": resp.status_code == 200,
                "has_text": bool(text.strip()),
            }
            if q.get("require_tool"):
                checks["called_" + q["require_tool"]] = q["require_tool"] in tools
            if q.get("require_any_tool"):
                checks["called_a_tool"] = any(t in tools for t in q["require_any_tool"])
            if q.get("needle"):
                checks["needle"] = q["needle"] in text.replace(",", "")
            if q["id"] == "enumeration":
                checks["enum_recall"] = bool(enum and enum.get("recall", 0) >= 0.90 and enum.get("cause") is None)
            ok = resp.status_code == 200 and all(checks.values())
            row = {
                "id": q["id"],
                "ok": ok,
                "checks": checks,
                "tools": tools,
                "text": text[:1200],
                "eval": payload.get("eval"),
                "usage": payload.get("usage"),
                "status": resp.status_code,
                "error": payload.get("detail") if resp.status_code >= 400 else None,
            }
            chats.append(row)
            steps.append({"id": "chat_" + q["id"], "ok": ok, "detail": row})

        usage = client.get("/api/usage", params={"session_id": "e2e-live"})
        usage_json = usage.json()
        traces = client.get("/api/traces", params={"session_id": "e2e-live"})
        steps.append(
            {
                "id": "session",
                "ok": usage.status_code == 200 and traces.status_code == 200 and len(traces.json()) > 0,
                "detail": {"usage": usage_json, "trace_count": len(traces.json()) if traces.status_code == 200 else 0},
            }
        )

    return _finish(workspace, provider, name, steps, t0, chats=chats)


def _finish(
    workspace: Path,
    provider: str,
    name: str,
    steps: list[dict[str, Any]],
    t0: float,
    chats: list[dict[str, Any]],
) -> dict[str, Any]:
    ok = bool(steps) and all(s["ok"] for s in steps)
    report = {
        "ok": ok,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "workspace": str(workspace),
        "model": {"provider": provider, "name": name},
        "steps": [{"id": s["id"], "ok": s["ok"]} for s in steps],
        "detail": steps,
        "chats": chats,
    }
    out = workspace / ".agentforge" / "e2e_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(out)
    return report
