"""FastAPI gateway: chat, probe, ingest, scan, usage, later stubs."""

from __future__ import annotations

import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gateway.db import Store
from gateway.evals import score_run
from gateway.catalog import merge_catalog, ollama_show, chat_template_ok, browse_llmfit
from gateway.later import later_status
from gateway.llmfit_client import LlmfitMissing, bench as llmfit_bench, find_llmfit
from gateway.llmfit_download import DownloadRejected, cancel as llmfit_dl_cancel
from gateway.llmfit_download import progress as llmfit_dl_progress
from gateway.llmfit_download import start_download as llmfit_dl_start
from gateway.local_scan import install_llmfit, scan as local_scan, try_local
from gateway.runtime.llamacpp import status as llamacpp_status
from gateway.runtime.ollama_runtime import (
    OllamaRuntime,
    PullRejected,
    normalize_tag,
    ollama_on_path,
)
from gateway.packs.files import fs_list, fs_read
from gateway.packs.files_write import (
    ALLOWED_EXTS,
    MAX_WRITE_BYTES,
    confirm_pending,
    fs_edit,
    fs_mkdir,
    fs_write,
    read_audit,
)
from gateway.packs.rag import HybridIndex, ingest_workspace
from gateway.paths import PathEscapeError, blocked_component, safe_join
from gateway.probe.runner import run_probe
from gateway.providers import ProviderError
from gateway.runtime.loop import run_loop, sanitize_messages
from gateway.runtime.tools import ALL_TOOLS
from gateway.spec import AgentSpec, Budgets, ModelPin, load_spec
from gateway.stats import effective_lo95, max_steps_from_lo95

# A pin selected without a probe has no measurement. `measured: False` keeps a
# conservative default from ever being read back as evidence.
UNMEASURED: dict[str, object] = {
    "per_step_success_lo95": None,
    "max_tools": 3,
    "max_steps": 3,
    "measured": False,
}

DEFAULT_PORT = 8788
STATIC = Path(__file__).parent / "static"
TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


class ChatIn(BaseModel):
    message: str
    session_id: str | None = None


class KeyIn(BaseModel):
    key: str


class ProbeIn(BaseModel):
    # Empty means "probe whatever is pinned right now". A probe measures the
    # pin; it must never silently select a different model for the user.
    provider: str = ""
    name: str = ""
    mode: str = "fast"


class PinIn(BaseModel):
    provider: str = "ollama"
    name: str
    probe: bool = False


class PullIn(BaseModel):
    name: str


class PermissionsIn(BaseModel):
    fs_write: str


class ConfirmIn(BaseModel):
    token: str
    approve: bool = True
    session_id: str | None = None


class ScanRunIn(BaseModel):
    name: str | None = None


class CatalogSettings(BaseModel):
    memory: str | None = None
    max_context: int | None = None


class CatalogPullIn(BaseModel):
    name: str
    ollama_name: str | None = None
    gguf_sources: list | None = None


class SessionResetIn(BaseModel):
    session_id: str = "default"


class CatalogMeasureIn(BaseModel):
    name: str


class LlmfitDownloadIn(BaseModel):
    name: str
    quant: str | None = None
    gguf_sources: list | None = None


def _prefer_local_ollama(ollama_rt: OllamaRuntime) -> str | None:
    """Pick a small installed Ollama tag. Prefer AgentForge GGUF imports (af-*)."""
    models = ollama_rt.list()
    names = [str(m.get("name") or m.get("model") or "").strip() for m in models]
    names = [n for n in names if n and not n.startswith("nomic-embed")]
    if not names:
        return None
    af = [n for n in names if n.startswith("af-")]
    if af:
        return sorted(af, key=len)[0]
    preferred = (
        "qwen2.5:0.5b",
        "tinyllama",
        "granite3.1-moe:latest",
        "qwen3:4b",
        "llama3.2:1b",
        "llama3.2:3b",
    )
    for want in preferred:
        for n in names:
            if n == want or n.startswith(want.split(":")[0] + ":"):
                return n
    return sorted(names, key=lambda n: (len(n), n))[0]


def _pin_ollama(store: Store, name: str) -> ModelPin:
    pin = ModelPin(provider="ollama", name=name, digest=f"ollama:{name}")
    store.set_setting("model_pin", json.dumps(pin.model_dump()))
    store.set_setting(
        "card",
        json.dumps(
            {
                "verdict": "agent",
                "cause": None,
                "model_pin": pin.model_dump(),
                "measured": UNMEASURED,
            }
        ),
    )
    return pin


def default_spec(workspace: Path, store: Store, *, ollama_rt: OllamaRuntime | None = None) -> AgentSpec:
    spec_path = TEMPLATES / "document-query" / "agentspec.yaml"
    spec = load_spec(spec_path) if spec_path.exists() else AgentSpec()
    spec.workspace_root = str(workspace)
    pin_json = store.get_setting("model_pin")
    if pin_json:
        spec.model_pin = ModelPin.model_validate(json.loads(pin_json))
    key = store.get_setting("openrouter_key") or os.environ.get("OPENROUTER_API_KEY")
    # Default template is OpenRouter; without a key, use an installed local model.
    if spec.model_pin.provider == "openrouter" and not key and ollama_rt is not None:
        local = _prefer_local_ollama(ollama_rt)
        if local:
            spec.model_pin = _pin_ollama(store, local)
    saved_perm = store.get_setting("fs_write_permission")
    if saved_perm in {"deny", "ask", "allow"}:
        spec.permissions.fs_write = saved_perm  # type: ignore[assignment]
    card_raw = store.get_setting("card")
    if card_raw:
        card = json.loads(card_raw)
        lo = effective_lo95(card.get("measured", {}).get("per_step_success_lo95"))
        spec.budgets = Budgets(
            max_steps=max_steps_from_lo95(lo),
            max_tools=max(len(ALL_TOOLS), card.get("measured", {}).get("max_tools") or 3),
            max_tokens=32000,
            max_wall_s=180,
            max_usd=0.50,
        )
        spec.card_ref = card.get("model_pin", {}).get("digest")
    return spec


def local_origins(port: int) -> list[str]:
    """The only origins allowed to drive this gateway.

    The UI is served from the same origin as the API, so cross-origin access buys
    the user nothing and costs them everything: with allow_origins=["*"] any page
    in their browser could POST /api/permissions to flip fs_write to allow, then
    drive /api/chat to write files, or GET /api/download/file to read the store.
    """
    return [f"http://127.0.0.1:{port}", f"http://localhost:{port}"]


def make_app(workspace: Path, store: Store, *, port: int = DEFAULT_PORT) -> FastAPI:
    app = FastAPI(title="AgentForge", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=local_origins(port),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    index = HybridIndex(workspace)
    ollama_rt = OllamaRuntime()
    skill_path = TEMPLATES / "document-query" / "SKILL.md"
    skill_text = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""

    def _spec() -> AgentSpec:
        return default_spec(workspace, store, ollama_rt=ollama_rt)

    def make_executor(permission: str, session: str):
        def executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
            if name == "fs_list":
                return fs_list(
                    workspace,
                    args.get("path") or ".",
                    glob=args.get("glob"),
                    recursive=bool(args.get("recursive")),
                )
            if name == "fs_read":
                return fs_read(workspace, args["path"], int(args.get("max_chars") or 4000))
            if name == "rag_search":
                if not index.ready():
                    return {"error": "not_ingested", "hint": "run: agentforge --ingest"}
                try:
                    return index.search(args["query"], int(args.get("limit") or 12))
                except Exception as exc:  # noqa: BLE001
                    return {"error": "rag_fail", "detail": str(exc)}
            if name == "fs_write":
                return fs_write(
                    workspace,
                    args["path"],
                    args.get("content") or "",
                    args.get("mode") or "create",
                    permission=permission,
                    session=session,
                )
            if name == "fs_edit":
                return fs_edit(
                    workspace,
                    args["path"],
                    args.get("old_text") or "",
                    args.get("new_text") or "",
                    permission=permission,
                    session=session,
                )
            if name == "fs_mkdir":
                return fs_mkdir(workspace, args["path"], permission=permission, session=session)
            return {"error": "bad_tool", "tool": name}

        return executor

    # Read-only executor for callers that have no session/permission context.
    executor = make_executor("deny", "default")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        # 1x1 transparent GIF. Cheaper than shipping a file, and stops the
        # 404 that was the only console error in the whole QA pass.
        return Response(
            content=base64.b64decode(
                "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
            ),
            media_type="image/gif",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.post("/api/session/reset")
    def session_reset(body: SessionResetIn):
        """Start a fresh chat. Without this the only reset was deleting the db."""
        sid = (body.session_id or "default").strip() or "default"
        store.save_session(sid, str(workspace), [])
        store.reset_usage(sid)
        return {"ok": True, "session_id": sid}

    @app.get("/api/health")
    def health():
        spec = _spec()
        return {
            "ok": True,
            "workspace": str(workspace),
            "ingested": index.ready(),
            "model": spec.model_pin.model_dump(),
            "fs_write": spec.permissions.fs_write,
            "has_key": bool(
                store.get_setting("openrouter_key") or os.environ.get("OPENROUTER_API_KEY")
            ),
        }

    @app.get("/api/permissions")
    def get_permissions():
        return {
            "fs_write": _spec().permissions.fs_write,
            "allowed_extensions": sorted(ALLOWED_EXTS),
            "max_write_bytes": MAX_WRITE_BYTES,
        }

    @app.post("/api/permissions")
    def set_permissions(body: PermissionsIn):
        value = (body.fs_write or "").strip().lower()
        if value not in {"deny", "ask", "allow"}:
            raise HTTPException(400, "fs_write must be deny, ask or allow")
        store.set_setting("fs_write_permission", value)
        return {"ok": True, "fs_write": value}

    @app.post("/api/confirm")
    def confirm_write(body: ConfirmIn):
        result = confirm_pending(
            body.token,
            approve=bool(body.approve),
            session=body.session_id or "default",
            workspace=workspace,
        )
        if result.get("error"):
            raise HTTPException(409, result.get("detail") or result["error"])
        return result

    @app.get("/api/audit")
    def audit(limit: int = 100):
        return {"entries": read_audit(workspace, limit=max(1, min(limit, 1000)))}

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
        value = (body.key or "").strip()
        if not value:
            # An empty submit is the only way to take a key back out again.
            store.delete_setting("openrouter_key")
            return {"ok": True, "has_key": False, "cleared": True}
        store.set_setting("openrouter_key", value)
        return {"ok": True, "has_key": True, "cleared": False}

    @app.get("/api/usage")
    def usage(session_id: str = "default"):
        spec = _spec()
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
        pin = _spec().model_pin
        provider = (body.provider or "").strip() or pin.provider
        name = (body.name or "").strip() or pin.name
        card = run_probe(provider=provider, name=name, mode=body.mode, n=3, api_key=key)
        # A probe measures the pinned model. It does not select one: writing
        # `model_pin` here used to replace the user's choice with the probe's
        # own target and, on failure, 409 every subsequent chat with no way out.
        store.set_setting("card", json.dumps(card))
        store.save_card(card)
        out = {k: card[k] for k in ("model_pin", "measured", "verdict", "cause") if k in card}
        out["probed"] = {"provider": provider, "name": name}
        return out

    @app.post("/api/probe/clear")
    def probe_clear():
        """Drop a failed verdict so a bad probe is not a dead end."""
        pin = _spec().model_pin
        card = {
            "verdict": "agent",
            "cause": None,
            "model_pin": pin.model_dump(),
            "measured": UNMEASURED,
        }
        store.set_setting("card", json.dumps(card))
        return {"ok": True, "verdict": "agent", "model_pin": pin.model_dump()}

    @app.get("/api/scan")
    def scan(quick: bool = True):
        return local_scan(quick=quick)

    @app.get("/api/catalog")
    def catalog(tools: bool | None = None, state: str | None = None, source: str | None = None, limit: int = 80):
        mem = store.get_setting("catalog_memory")
        ctx_raw = store.get_setting("catalog_max_context")
        max_ctx = int(ctx_raw) if ctx_raw else None
        measured_raw = store.get_setting("catalog_measured")
        measured = json.loads(measured_raw) if measured_raw else {}
        data = merge_catalog(memory=mem, max_context=max_ctx, live=True, limit=max(5, min(limit, 200)))
        try:
            data["llmfit_bin"] = str(find_llmfit())
            data["llmfit"] = True
            data["llmfit_error"] = None
        except LlmfitMissing as exc:
            data["llmfit"] = False
            data["llmfit_error"] = str(exc)
            data["llmfit_bin"] = None
        for row in data["rows"]:
            key = row.get("ollama_name") or row.get("name")
            if key in measured:
                row["measured_tok_s"] = measured[key]
        if tools is True:
            data["rows"] = [r for r in data["rows"] if r.get("supports_tools")]
        if state:
            data["rows"] = [r for r in data["rows"] if r.get("state") == state]
        if source:
            data["rows"] = [r for r in data["rows"] if r.get("source") == source]
        data["current"] = _spec().model_pin.model_dump()
        data["settings"] = {"memory": mem, "max_context": max_ctx}
        data["llamacpp"] = llamacpp_status()
        return data

    @app.get("/api/catalog/ollama")
    def catalog_ollama(limit: int = 80):
        mem = store.get_setting("catalog_memory")
        ctx_raw = store.get_setting("catalog_max_context")
        max_ctx = int(ctx_raw) if ctx_raw else None
        data = merge_catalog(memory=mem, max_context=max_ctx, live=True, limit=max(5, min(limit, 200)))
        rows: list[dict[str, Any]] = []
        for row in data.get("rows") or []:
            if row.get("source") == "ollama":
                rows.append(row)
            elif row.get("ollama_name") and row.get("state") == "PULLABLE":
                rows.append(row)
        return {
            "system": data.get("system") or {},
            "rows": rows,
            "total": len(rows),
            "ollama": ollama_on_path(),
        }

    @app.get("/api/catalog/llmfit")
    def catalog_llmfit(
        q: str = "",
        provider: str = "",
        use_case: str = "",
        capability: str = "",
        fit: str = "",
        sort: str = "score",
        tools: bool | None = None,
        downloadable: bool | None = None,
        offset: int = 0,
        limit: int = 80,
        refresh: bool = False,
    ):
        mem = store.get_setting("catalog_memory")
        return browse_llmfit(
            q=q,
            provider=provider,
            use_case=use_case,
            capability=capability,
            fit=fit,
            sort=sort,
            tools=tools,
            downloadable=downloadable,
            offset=offset,
            limit=limit,
            memory=mem,
            refresh=refresh,
        )

    @app.post("/api/catalog/settings")
    def catalog_settings(body: CatalogSettings):
        if body.memory is not None:
            store.set_setting("catalog_memory", body.memory)
        if body.max_context is not None:
            store.set_setting("catalog_max_context", str(body.max_context))
        return {"ok": True}

    def _row_for(name: str, ollama_name: str | None = None) -> dict[str, Any] | None:
        mem = store.get_setting("catalog_memory")
        ctx_raw = store.get_setting("catalog_max_context")
        max_ctx = int(ctx_raw) if ctx_raw else None
        data = merge_catalog(memory=mem, max_context=max_ctx, live=True)
        want = {normalize_tag(t) for t in ((ollama_name or "").strip(), name.strip()) if t}
        for row in data.get("rows") or []:
            have = {
                normalize_tag(str(row.get(k) or ""))
                for k in ("name", "ollama_name")
                if row.get(k)
            }
            if have & want:
                return row
        return None

    @app.post("/api/catalog/pull")
    def catalog_pull(body: CatalogPullIn):
        tag = (body.ollama_name or body.name).strip()
        row = _row_for(body.name, body.ollama_name)
        if row and row.get("state") in {"NO_RUNTIME", "UNRESOLVABLE"}:
            raise HTTPException(status_code=409, detail=row.get("reason") or f"{row['state']}: Chat and Pull disabled")
        sources = body.gguf_sources if body.gguf_sources is not None else (row or {}).get("gguf_sources")
        try:
            return ollama_rt.pull_start(tag, gguf_sources=sources)
        except PullRejected as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/catalog/pull/{job_id}")
    def catalog_pull_progress(job_id: str):
        return ollama_rt.pull_progress(job_id)

    @app.post("/api/catalog/pull/{job_id}/cancel")
    def catalog_pull_cancel(job_id: str):
        return ollama_rt.pull_cancel(job_id)

    @app.post("/api/catalog/pull/resume")
    def catalog_pull_resume(body: CatalogPullIn):
        tag = (body.ollama_name or body.name).strip()
        try:
            return ollama_rt.pull_resume(tag)
        except PullRejected as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/catalog/llmfit/download")
    def catalog_llmfit_download(body: LlmfitDownloadIn):
        row = _row_for(body.name)
        sources = body.gguf_sources if body.gguf_sources is not None else (row or {}).get("gguf_sources")
        quant = body.quant or (row or {}).get("quant")
        try:
            return llmfit_dl_start(body.name.strip(), quant=quant, gguf_sources=sources)
        except DownloadRejected as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/catalog/llmfit/download/{job_id}")
    def catalog_llmfit_download_progress(job_id: str):
        return llmfit_dl_progress(job_id)

    @app.post("/api/catalog/llmfit/download/{job_id}/cancel")
    def catalog_llmfit_download_cancel(job_id: str):
        return llmfit_dl_cancel(job_id)

    @app.post("/api/catalog/measure")
    def catalog_measure(body: CatalogMeasureIn):
        result = llmfit_bench(body.name, provider="ollama")
        tps = result.get("tokens_per_second") or result.get("tps") or result.get("estimated_tps")
        raw = store.get_setting("catalog_measured")
        measured = json.loads(raw) if raw else {}
        measured[body.name] = tps
        store.set_setting("catalog_measured", json.dumps(measured))
        return {"ok": True, "measured_tok_s": tps, "raw": result}

    @app.get("/api/llamacpp")
    def llamacpp():
        return llamacpp_status()

    @app.get("/api/models")
    def models():
        data = local_scan(quick=False)
        data["current"] = _spec().model_pin.model_dump()
        return data

    @app.post("/api/model")
    def pin_model(body: PinIn):
        if body.provider not in {"openrouter", "ollama"}:
            raise HTTPException(400, "provider must be openrouter or ollama")
        if not body.name.strip():
            raise HTTPException(400, "name required")
        key = store.get_setting("openrouter_key") or os.environ.get("OPENROUTER_API_KEY")
        if body.probe:
            card = run_probe(provider=body.provider, name=body.name.strip(), mode="fast", n=3, api_key=key)
            store.set_setting("card", json.dumps(card))
            store.set_setting("model_pin", json.dumps(card["model_pin"]))
            store.save_card(card)
            return {
                "ok": card.get("verdict") == "agent",
                "probed": True,
                "model_pin": card.get("model_pin"),
                "verdict": card.get("verdict"),
                "cause": card.get("cause"),
            }
        pin = ModelPin(provider=body.provider, name=body.name.strip())
        pin.digest = f"{pin.provider}:{pin.name}"
        store.set_setting("model_pin", json.dumps(pin.model_dump()))
        store.set_setting(
            "card",
            json.dumps(
                {
                    "verdict": "agent",
                    "cause": None,
                    "model_pin": pin.model_dump(),
                    "measured": UNMEASURED,
                }
            ),
        )
        return {"ok": True, "probed": False, "model_pin": pin.model_dump()}

    @app.post("/api/ollama/pull")
    def pull_ollama(body: PullIn):
        try:
            return ollama_rt.pull_start(body.name.strip())
        except PullRejected as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/install/llmfit")
    def install_local_llmfit():
        result = install_llmfit()
        if not result.get("ok"):
            raise HTTPException(status_code=500, detail=result)
        return result

    @app.get("/api/files")
    def files():
        skip = {".agentforge", ".git", ".venv", "node_modules", "__pycache__"}
        # Include the writable text formats so a file the agent just created
        # shows up in the sidebar without a restart.
        exts = {".pdf", ".txt", ".md", ".docx", ".json", ".csv", ".yaml", ".yml"}
        rows = []
        for p in workspace.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(workspace)
            if any(part in skip for part in rel.parts):
                continue
            if p.suffix.lower() not in exts or p.name.startswith("~$"):
                continue
            indexed = index.indexed_paths()
            rows.append(
                {
                    "path": rel.as_posix(),
                    "size": p.stat().st_size,
                    "indexed": rel.as_posix() in indexed,
                    "writable": p.suffix.lower() in ALLOWED_EXTS,
                }
            )
        rows.sort(key=lambda r: r["path"])
        return {"files": rows}

    @app.get("/api/download/file")
    def download_file(path: str = Query(..., min_length=1)):
        try:
            target = safe_join(workspace, path)
        except PathEscapeError as exc:
            raise HTTPException(400, str(exc)) from exc
        # Without this, ?path=.agentforge/app.sqlite hands out the store -- and
        # the OpenRouter key it holds in plaintext.
        blocked = blocked_component(target.relative_to(workspace.resolve()).parts, hidden=True)
        if blocked:
            raise HTTPException(400, f"{blocked} is protected and cannot be downloaded")
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
    def scan_run(body: ScanRunIn | None = None):
        key = store.get_setting("openrouter_key") or os.environ.get("OPENROUTER_API_KEY")
        name = body.name.strip() if body and body.name else None
        result = try_local(api_key=key, name=name)
        if result.get("ok") and result.get("card"):
            card = result["card"]
            store.set_setting("card", json.dumps(card))
            store.set_setting("model_pin", json.dumps(card["model_pin"]))
        return {
            "ok": result.get("ok"),
            "reason": result.get("reason"),
            "english": result.get("english"),
            "picked": result.get("picked"),
            "card": {k: result.get("card", {}).get(k) for k in ("model_pin", "measured", "verdict", "cause")},
        }

    @app.post("/api/chat")
    def chat(body: ChatIn):
        key = store.get_setting("openrouter_key") or os.environ.get("OPENROUTER_API_KEY")
        spec = _spec()
        if spec.model_pin.provider == "openrouter" and not key:
            local = _prefer_local_ollama(ollama_rt)
            if local:
                spec.model_pin = _pin_ollama(store, local)
            else:
                raise HTTPException(
                    400,
                    "No OpenRouter key and no local Ollama model. "
                    "Use Browse Ollama models / Browse llmfit models, or paste an OpenRouter key.",
                )
        if spec.model_pin.provider == "ollama":
            row = _row_for(spec.model_pin.name, spec.model_pin.name)
            if row and row.get("state") in {"NO_RUNTIME", "UNRESOLVABLE"}:
                raise HTTPException(
                    409,
                    row.get("reason") or f"{row['state']}: Chat and Pull disabled",
                )
            show = ollama_show(spec.model_pin.name)
            if chat_template_ok(show) is False:
                raise HTTPException(409, "missing or invalid chat template; chat refused")
            ollama_rt.ensure_available(spec.model_pin.name)
        complete_fn = ollama_rt.chat if spec.model_pin.provider == "ollama" else None
        grounded_skill = skill_text
        if index.ready():
            try:
                pack = index.search(body.message, limit=8)
                hits = pack.get("hits") or []
                if hits:
                    excerpts = "\n".join(
                        f"- {h.get('path')}: {(h.get('text') or '')[:240]}" for h in hits[:8]
                    )
                    grounded_skill = (
                        (skill_text or "")
                        + "\n\nRetrieved from this folder (cite these paths):\n"
                        + excerpts
                    )
            except Exception:  # noqa: BLE001
                grounded_skill = skill_text
        card_raw = store.get_setting("card")
        if card_raw:
            card = json.loads(card_raw)
            pinned = spec.model_pin.name
            same = (card.get("model_pin") or {}).get("name") == pinned
            if same and card.get("verdict") == "chat_only":
                raise HTTPException(409, f"probe verdict chat_only: {card.get('cause')}")
        session_id = body.session_id or "default"
        history = sanitize_messages(store.load_session(session_id))
        turn_executor = make_executor(spec.permissions.fs_write, session_id)
        try:
            out = run_loop(
                spec,
                body.message,
                history=history,
                executor=turn_executor,
                api_key=key,
                skill_text=grounded_skill,
                complete_fn=complete_fn,
            )
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)[:500]) from exc
        store.save_session(session_id, str(workspace), sanitize_messages(out["messages"]))
        for t in out["traces"]:
            store.add_trace(session_id, t.get("kind", "event"), t)
        # usage: approximate split
        turn = int(out["usage"].get("turn_tokens") or 0)
        usd = float(out["usage"].get("usd") or 0)
        prompt_t = int(out["usage"].get("prompt_tokens") or 0)
        completion_t = int(out["usage"].get("completion_tokens") or turn - prompt_t)
        totals = store.add_usage(session_id, prompt_t, completion_t, usd)
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
        lo95 = effective_lo95(None)
        if card_raw:
            lo95 = effective_lo95(
                json.loads(card_raw).get("measured", {}).get("per_step_success_lo95")
            )
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
            # The UI needs these to show a diff modal and refresh the file list.
            "needs_confirm": out.get("needs_confirm") or [],
            "writes": out.get("writes") or [],
            "fs_write": spec.permissions.fs_write,
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
