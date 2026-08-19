"""FastAPI gateway: chat, probe, ingest, scan, usage, later stubs."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gateway.db import Store
from gateway.evals import score_run
from gateway.later import later_status
from gateway.local_scan import install_llmfit, scan as local_scan, try_local
from gateway.packs.files import fs_list, fs_read
from gateway.packs.rag import HybridIndex, ingest_workspace
from gateway.paths import PathEscapeError, safe_join
from gateway.probe.runner import run_probe
from gateway.providers import ProviderError
from gateway.runtime.loop import run_loop, sanitize_messages
from gateway.spec import AgentSpec, Budgets, ModelPin, load_spec
from gateway.stats import max_steps_from_lo95

STATIC = Path(__file__).parent / "static"
TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


class ChatIn(BaseModel):
    message: str
    session_id: str | None = None


class KeyIn(BaseModel):
    key: str


class ProbeIn(BaseModel):
    provider: str = "openrouter"
    name: str = "openai/gpt-4o-mini"
    mode: str = "fast"


def default_spec(workspace: Path, store: Store) -> AgentSpec:
    spec_path = TEMPLATES / "document-query" / "agentspec.yaml"
    spec = load_spec(spec_path) if spec_path.exists() else AgentSpec()
    spec.workspace_root = str(workspace)
    pin_json = store.get_setting("model_pin")
    if pin_json:
        spec.model_pin = ModelPin.model_validate(json.loads(pin_json))
    card_raw = store.get_setting("card")
    if card_raw:
        card = json.loads(card_raw)
        lo = card.get("measured", {}).get("per_step_success_lo95", 0.72)
        spec.budgets = Budgets(
            max_steps=max_steps_from_lo95(lo),
            max_tools=max(3, card.get("measured", {}).get("max_tools") or 3),
            max_tokens=32000,
            max_wall_s=90,
            max_usd=0.50,
        )
        spec.card_ref = card.get("model_pin", {}).get("digest")
    return spec


def make_app(workspace: Path, store: Store) -> FastAPI:
    app = FastAPI(title="AgentForge", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    index = HybridIndex(workspace)
    skill_path = TEMPLATES / "document-query" / "SKILL.md"
    skill_text = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""

    def executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "fs_list":
            return fs_list(workspace, args.get("path") or ".")
        if name == "fs_read":
            return fs_read(workspace, args["path"], int(args.get("max_chars") or 4000))
        if name == "rag_search":
            if not index.ready():
                return {"error": "not_ingested", "hint": "run: agentforge --ingest"}
            try:
                return index.search(args["query"], int(args.get("limit") or 12))
            except Exception as exc:  # noqa: BLE001
                return {"error": "rag_fail", "detail": str(exc)}
        return {"error": "bad_tool", "tool": name}

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "workspace": str(workspace),
            "ingested": index.ready(),
        }

    @app.get("/api/later")
    def later():
        return later_status()

    @app.get("/api/later/{pack}")
    def later_pack(pack: str):
        data = later_status()["later"]
        if pack not in data:
            raise HTTPException(404, f"unknown pack {pack}")
        raise HTTPException(status_code=501, detail=data[pack])

    @app.post("/api/key")
    def set_key(body: KeyIn):
        store.set_setting("openrouter_key", body.key.strip())
        return {"ok": True}

    @app.get("/api/usage")
    def usage(session_id: str = "default"):
        spec = default_spec(workspace, store)
        u = store.get_usage(session_id)
        u["max_tokens"] = spec.budgets.max_tokens
        u["max_usd"] = spec.budgets.max_usd
        return u

    @app.post("/api/ingest")
    def ingest():
        warning = ingest_workspace(workspace)
        index._load()
        return warning

    @app.get("/api/ingest/warnings")
    def ingest_warnings():
        p = workspace / ".agentforge" / "ingest_warnings.json"
        if not p.exists():
            return {"files_scanned": 0, "no_text_layer_count": 0, "no_text_layer": []}
        return json.loads(p.read_text(encoding="utf-8"))

    @app.post("/api/probe")
    def probe(body: ProbeIn):
        key = store.get_setting("openrouter_key") or os.environ.get("OPENROUTER_API_KEY")
        card = run_probe(provider=body.provider, name=body.name, mode=body.mode, n=3, api_key=key)
        store.set_setting("card", json.dumps(card))
        store.set_setting("model_pin", json.dumps(card["model_pin"]))
        store.save_card(card)
        return {k: card[k] for k in ("model_pin", "measured", "verdict", "cause") if k in card}

    @app.get("/api/scan")
    def scan(quick: bool = True):
        return local_scan(quick=quick)

    @app.post("/api/install/llmfit")
    def install_local_llmfit():
        result = install_llmfit()
        if not result.get("ok"):
            raise HTTPException(status_code=500, detail=result)
        return result

    @app.get("/api/files")
    def files():
        skip = {".agentforge", ".git", ".venv"}
        exts = {".pdf", ".txt", ".md", ".docx"}
        rows = []
        for p in workspace.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(workspace)
            if any(part in skip for part in rel.parts):
                continue
            if p.suffix.lower() not in exts:
                continue
            rows.append({"path": rel.as_posix(), "size": p.stat().st_size})
        rows.sort(key=lambda r: r["path"])
        return {"files": rows}

    @app.get("/api/download/file")
    def download_file(path: str = Query(..., min_length=1)):
        try:
            target = safe_join(workspace, path)
        except PathEscapeError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not target.is_file():
            raise HTTPException(404, "file not found")
        return FileResponse(target, filename=target.name)

    @app.get("/api/export/chat")
    def export_chat(session_id: str = "default"):
        msgs = store.load_session(session_id)
        lines = ["# AgentForge chat", ""]
        for m in msgs:
            role = m.get("role") or "unknown"
            if role == "tool":
                continue
            content = (m.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"## {role}")
            lines.append(content)
            lines.append("")
        body = "\n".join(lines).encode("utf-8")
        return Response(
            content=body,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="agentforge-chat.md"'},
        )

    @app.post("/api/scan/run")
    def scan_run():
        key = store.get_setting("openrouter_key") or os.environ.get("OPENROUTER_API_KEY")
        result = try_local(api_key=key)
        if result.get("ok") and result.get("card"):
            card = result["card"]
            store.set_setting("card", json.dumps(card))
            store.set_setting("model_pin", json.dumps(card["model_pin"]))
        return {
            "ok": result.get("ok"),
            "reason": result.get("reason"),
            "english": result.get("english"),
            "card": {k: result.get("card", {}).get(k) for k in ("model_pin", "measured", "verdict", "cause")},
        }

    @app.post("/api/chat")
    def chat(body: ChatIn):
        key = store.get_setting("openrouter_key") or os.environ.get("OPENROUTER_API_KEY")
        spec = default_spec(workspace, store)
        if spec.model_pin.provider == "openrouter" and not key:
            raise HTTPException(400, "paste an OpenRouter key first")
        card_raw = store.get_setting("card")
        if card_raw:
            card = json.loads(card_raw)
            if card.get("verdict") == "chat_only":
                raise HTTPException(409, f"probe verdict chat_only: {card.get('cause')}")
        session_id = body.session_id or "default"
        history = sanitize_messages(store.load_session(session_id))
        try:
            out = run_loop(
                spec,
                body.message,
                history=history,
                executor=executor,
                api_key=key,
                skill_text=skill_text,
            )
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)[:500]) from exc
        store.save_session(session_id, str(workspace), sanitize_messages(out["messages"]))
        for t in out["traces"]:
            store.add_trace(session_id, t.get("kind", "event"), t)
        # usage: approximate split
        turn = int(out["usage"].get("turn_tokens") or 0)
        usd = float(out["usage"].get("usd") or 0)
        totals = store.add_usage(session_id, turn, 0, usd)
        totals["turn_tokens"] = turn
        totals["prompt_tokens"] = int(totals.get("prompt_tokens") or 0)
        totals["completion_tokens"] = int(totals.get("completion_tokens") or 0)
        totals["turn_usd"] = usd
        totals["max_tokens"] = spec.budgets.max_tokens
        totals["max_usd"] = spec.budgets.max_usd
        totals["stop"] = out.get("stop")
        tool_paths: list[str] = []
        for t in out["traces"]:
            if t.get("kind") == "tool" and t.get("tool") == "rag_search":
                pass
        # pull paths from last rag result in messages
        for m in reversed(out["messages"]):
            if m.get("role") == "tool":
                try:
                    payload = json.loads(m.get("content") or "{}")
                    tool_paths.extend(payload.get("paths") or [])
                except json.JSONDecodeError:
                    continue
        lo95 = 0.0
        if card_raw:
            lo95 = json.loads(card_raw).get("measured", {}).get("per_step_success_lo95", 0.0)
        scored = score_run(
            index=index,
            question=body.message,
            answer=out["text"],
            tool_paths=tool_paths,
            usage=out["usage"],
            lo95=lo95,
        )
        store.add_trace(session_id, "eval", scored)
        return {
            "session_id": session_id,
            "text": out["text"],
            "traces": out["traces"],
            "usage": totals,
            "eval": scored,
        }

    @app.get("/api/traces")
    def traces(session_id: str = "default"):
        return store.traces(session_id)

    if STATIC.exists():
        app.mount("/assets", StaticFiles(directory=STATIC), name="assets")

        @app.get("/")
        def root():
            return FileResponse(STATIC / "index.html")

    return app
