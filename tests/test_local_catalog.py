"""Named local-catalog cases (llmfit + Ollama). Do not edit these to force a pass."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.app import make_app
from gateway.catalog import (
    already_downloaded_names,
    fit_label,
    in_download_cache,
    is_sharded_gguf,
    merge_catalog,
)
from gateway.db import Store
from gateway.llmfit_client import LlmfitMissing, LlmfitSchemaError, find_llmfit, fit as llmfit_fit
from gateway.runtime.llamacpp import LLAMACPP_ENABLED, status as llamacpp_status
from gateway.runtime.ollama_runtime import OllamaRuntime, PullRejected

FIXTURE_BIN = Path(__file__).resolve().parent / "fixtures" / "fake_llmfit.py"
NEEDLE = "ORCHID-TOKEN-7712"

# Official plan names (hyphenated) → this module's test_* functions.
PLAN_NAMES = {
    "test_llmfit_missing_binary": "llmfit-missing-binary",
    "test_llmfit_schema_drift": "llmfit-schema-drift",
    "test_catalog_merge": "catalog-merge",
    "test_no_duplicate_download": "no-duplicate-download",
    "test_fits_calc": "fits-calc",
    "test_sharded_rejected": "sharded-rejected",
    "test_no_runtime_row_blocked": "no-runtime-row-blocked",
    "test_pull_cancel_resume": "pull-cancel-resume",
    "test_tools_filter": "tools-filter",
    "test_rag_answer_grounded": "rag-answer-grounded",
}


def _fake_llmfit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, mode: str = "ok") -> Path:
    monkeypatch.setenv("LLMFIT_BIN", str(FIXTURE_BIN))
    monkeypatch.setenv("FAKE_LLMFIT_MODE", mode)
    monkeypatch.setenv("FAKE_LLMFIT_ARGV", str(tmp_path / "llmfit_argv.json"))
    return FIXTURE_BIN


def _stub_ollama(monkeypatch: pytest.MonkeyPatch, models: list[dict[str, Any]] | None = None) -> None:
    tags = models if models is not None else [{"name": "tinyllama:latest", "size_gb": 0.6}]

    def tags_fn(base: str = "http://127.0.0.1:11434") -> list[dict[str, Any]]:
        return list(tags)

    def show_fn(name: str, base: str = "http://127.0.0.1:11434") -> dict[str, Any]:
        return {"template": "{{ .System }}{{ .Prompt }}", "details": {"family": "llama"}}

    monkeypatch.setattr("gateway.catalog._ollama_tags", tags_fn)
    monkeypatch.setattr("gateway.catalog.ollama_show", show_fn)
    monkeypatch.setattr("gateway.app.ollama_show", show_fn)


def _client(tmp_path: Path) -> TestClient:
    app = make_app(tmp_path, Store(tmp_path / "app.sqlite"))
    return TestClient(app)


def test_llmfit_missing_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """llmfit-missing-binary"""
    missing = tmp_path / "no-such-llmfit.exe"
    monkeypatch.setenv("LLMFIT_BIN", str(missing))
    with pytest.raises(LlmfitMissing) as exc:
        find_llmfit()
    msg = str(exc.value)
    assert "llmfit not found" in msg.lower() or "not found" in msg.lower()
    assert "uv tool install llmfit" in msg


def test_llmfit_schema_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """llmfit-schema-drift"""
    _fake_llmfit(monkeypatch, tmp_path, mode="schema-drift")
    with pytest.raises(LlmfitSchemaError) as exc:
        llmfit_fit(limit=5)
    assert "ollama_name" in str(exc.value)


def test_catalog_merge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """catalog-merge"""
    _fake_llmfit(monkeypatch, tmp_path)
    payload = llmfit_fit(limit=8)
    data = merge_catalog(
        live=False,
        ollama_models=[{"name": "tinyllama:latest", "size_gb": 0.6}],
        fit_payload=payload,
        system={"available_ram_gb": 16.0, "total_ram_gb": 32.0},
        downloaded=set(),
    )
    by_state: dict[str, list[str]] = {}
    for row in data["rows"]:
        by_state.setdefault(row["state"], []).append(row["name"])
    assert any(n == "tinyllama:latest" for n in by_state.get("INSTALLED", []))
    assert any("Qwen2.5-3B" in n for n in by_state.get("PULLABLE", []))
    assert any("Voice-Embedding" in n or "Qwen3-Voice" in n for n in by_state.get("NO_RUNTIME", []))
    assert any("sharded" in n.lower() or "70b" in n.lower() for n in by_state.get("UNRESOLVABLE", []))
    cols = {
        "name",
        "source",
        "params",
        "quant",
        "size",
        "est_vram",
        "FITS",
        "ctx",
        "supports_tools",
        "est_tok_s",
        "measured_tok_s",
        "state",
    }
    assert cols <= set(data["rows"][0])


def test_no_duplicate_download(tmp_path: Path) -> None:
    """no-duplicate-download"""
    cache = tmp_path / "llmfit-models"
    cache.mkdir()
    (cache / "cached-local.gguf").write_bytes(b"gguf")
    found = already_downloaded_names(
        {
            "llmfit": cache,
            "huggingface": tmp_path / "hf",
            "ollama": tmp_path / "ollama",
            "llama": tmp_path / "llama",
        }
    )
    assert in_download_cache("org/cached-local", "cached-local:latest", found)
    data = merge_catalog(
        live=False,
        ollama_models=[],
        fit_payload={
            "models": [
                {
                    "name": "org/cached-local",
                    "ollama_name": "cached-local:latest",
                    "fit_level": "Good",
                    "best_quant": "Q4_0",
                    "estimated_tps": 30,
                    "parameter_count": "1B",
                    "gguf_sources": ["cached-local.gguf"],
                    "capability_ids": [],
                }
            ]
        },
        system={"available_ram_gb": 16},
        downloaded=found,
    )
    row = next(r for r in data["rows"] if r["ollama_name"] == "cached-local:latest")
    assert row["state"] == "INSTALLED"
    assert row["state"] != "PULLABLE"


def test_fits_calc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """fits-calc"""
    assert fit_label(2.5, 8.0) == "perfect"
    assert fit_label(4.0, 5.65) == "good"
    assert fit_label(12.0, 5.0) == "wont_fit"
    argv_path = tmp_path / "llmfit_argv.json"
    _fake_llmfit(monkeypatch, tmp_path)
    llmfit_fit(limit=5, memory="8G", max_context=4096)
    argv = json.loads(argv_path.read_text(encoding="utf-8"))
    assert "--memory" in argv and "8G" in argv
    assert "--max-context" in argv and "4096" in argv
    data = merge_catalog(
        live=False,
        ollama_models=[{"name": "tinyllama:latest", "size_gb": 2.0}],
        fit_payload={"models": []},
        system={"available_ram_gb": 32.0},
        memory="8G",
        downloaded=set(),
    )
    inst = next(r for r in data["rows"] if r["name"] == "tinyllama:latest")
    assert inst["FITS"] == "perfect"


def test_sharded_rejected() -> None:
    """sharded-rejected"""
    sources = ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf"]
    assert is_sharded_gguf(sources)
    rt = OllamaRuntime()
    with pytest.raises(PullRejected) as exc:
        rt.pull_start("giant:70b", gguf_sources=sources)
    assert "sharded" in str(exc.value).lower()


def test_no_runtime_row_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """no-runtime-row-blocked"""
    _fake_llmfit(monkeypatch, tmp_path)
    _stub_ollama(monkeypatch, models=[])

    def fake_merge(**kwargs: Any) -> dict[str, Any]:
        return {
            "system": {},
            "rows": [
                {
                    "name": "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B",
                    "ollama_name": None,
                    "state": "NO_RUNTIME",
                    "reason": "no ollama_name; llama.cpp is off",
                    "gguf_sources": ["voice.gguf"],
                    "supports_tools": False,
                    "FITS": "Good",
                    "chat_template_ok": None,
                }
            ],
            "caches": {},
        }

    monkeypatch.setattr("gateway.app.merge_catalog", fake_merge)
    client = _client(tmp_path)
    pull = client.post(
        "/api/catalog/pull",
        json={"name": "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"},
    )
    assert pull.status_code == 409
    pin = client.post(
        "/api/model",
        json={"provider": "ollama", "name": "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"},
    )
    assert pin.status_code == 200
    chat = client.post("/api/chat", json={"message": "hello", "session_id": "t"})
    assert chat.status_code == 409
    body = chat.json()
    detail = body.get("detail") or ""
    assert "NO_RUNTIME" in str(detail) or "llama.cpp" in str(detail) or "disabled" in str(detail).lower() or "ollama_name" in str(detail)


def test_pull_cancel_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """pull-cancel-resume"""
    resume_flag = {"n": 0}
    lock = threading.Lock()

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            with lock:
                resume_flag["n"] += 1
                run = resume_flag["n"]
            for i in range(1, 30):
                time.sleep(0.05)
                yield json.dumps({"status": "pulling", "total": 1000, "completed": i * 20, "run": run})
            yield json.dumps({"status": "success", "total": 1000, "completed": 1000})

    def fake_stream(method: str, url: str, **kwargs: Any) -> FakeResp:
        assert url.endswith("/api/pull")
        return FakeResp()

    monkeypatch.setattr("gateway.runtime.ollama_runtime.httpx.stream", fake_stream)
    rt = OllamaRuntime()
    started = rt.pull_start("tinyllama")
    job_id = started["job_id"]
    deadline = time.time() + 5
    while time.time() < deadline:
        prog = rt.pull_progress(job_id)
        if float(prog.get("percent") or 0) > 0:
            break
        time.sleep(0.05)
    assert float(rt.pull_progress(job_id).get("percent") or 0) > 0
    rt.pull_cancel(job_id)
    deadline = time.time() + 5
    while time.time() < deadline:
        prog = rt.pull_progress(job_id)
        if prog.get("done"):
            break
        time.sleep(0.05)
    assert rt.pull_progress(job_id)["status"] == "cancelled"
    resumed = rt.pull_resume("tinyllama")
    assert resumed.get("resumed") is True
    job2 = resumed["job_id"]
    deadline = time.time() + 8
    while time.time() < deadline:
        prog = rt.pull_progress(job2)
        if prog.get("done"):
            break
        time.sleep(0.05)
    assert rt.pull_progress(job2)["status"] == "success"
    assert resume_flag["n"] >= 2


def test_tools_filter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """tools-filter"""
    _fake_llmfit(monkeypatch, tmp_path)
    _stub_ollama(monkeypatch, models=[])
    payload = llmfit_fit(limit=8)
    data = merge_catalog(
        live=False,
        ollama_models=[],
        fit_payload=payload,
        system={"available_ram_gb": 16},
        downloaded=set(),
    )
    with_tools = [r for r in data["rows"] if r.get("supports_tools")]
    without = [r for r in data["rows"] if not r.get("supports_tools")]
    assert with_tools
    assert without
    monkeypatch.setattr(
        "gateway.app.merge_catalog",
        lambda **kwargs: {"system": {}, "rows": data["rows"], "caches": {}},
    )
    client = _client(tmp_path)
    filtered = client.get("/api/catalog", params={"tools": "true"}).json()["rows"]
    assert filtered
    assert all(r.get("supports_tools") for r in filtered)


def _ollama_up() -> bool:
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _small_installed() -> str | None:
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5.0)
        r.raise_for_status()
    except Exception:  # noqa: BLE001
        return None
    for m in r.json().get("models") or []:
        size = float(m.get("size") or 0)
        name = m.get("name") or m.get("model")
        if name and size and size < 1024**3:
            return str(name)
    return None


def test_rag_answer_grounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """rag-answer-grounded — one live <1GB pull+RAG path when Ollama is up; otherwise stubbed pull+index+chat."""
    (tmp_path / "alpha.md").write_text(
        f"Project code for the orchard run is {NEEDLE}. Do not mention this in other files.\n",
        encoding="utf-8",
    )
    (tmp_path / "beta.md").write_text("Beta file talks about weather and tea leaves only.\n", encoding="utf-8")
    (tmp_path / "gamma.md").write_text("Gamma file lists crate sizes 12 and 18. No project code.\n", encoding="utf-8")

    live = _ollama_up() and os.environ.get("AGENTFORGE_LIVE_RAG", "0") == "1"
    if live:
        tag = _small_installed()
        rt = OllamaRuntime()
        if not tag:
            tag = "tinyllama"
            job = rt.pull_start(tag)
            deadline = time.time() + 180
            while time.time() < deadline:
                p = rt.pull_progress(job["job_id"])
                if p.get("done"):
                    break
                time.sleep(1)
            assert rt.pull_progress(job["job_id"]).get("status") == "success"
        client = _client(tmp_path)
        assert client.post("/api/ingest").status_code == 200
        assert client.post("/api/model", json={"provider": "ollama", "name": tag}).status_code == 200
        asked = client.post(
            "/api/chat",
            json={"message": "What is the project code token in alpha.md? Quote it exactly.", "session_id": "rag"},
        )
        assert asked.status_code == 200, asked.text
        assert NEEDLE in (asked.json().get("text") or "")
        return

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            yield json.dumps({"status": "success", "total": 100, "completed": 100})

    monkeypatch.setattr("gateway.runtime.ollama_runtime.httpx.stream", lambda *a, **k: FakeResp())
    monkeypatch.setattr(
        "gateway.runtime.ollama_runtime.OllamaRuntime.list",
        lambda self: [{"name": "tinyllama"}],
    )
    monkeypatch.setattr(
        "gateway.runtime.ollama_runtime.OllamaRuntime.ensure_available",
        lambda self, name, **k: {"ok": True, "already": True, "name": name},
    )

    def fake_chat(**kwargs: Any) -> dict[str, Any]:
        blob = json.dumps(kwargs.get("messages") or [])
        assert NEEDLE in blob, "RAG excerpts must reach the model"
        return {
            "content": f"The project code in alpha.md is {NEEDLE}.",
            "tool_calls": [],
            "prompt_tokens": 40,
            "completion_tokens": 12,
            "elapsed_s": 0.01,
            "finish_reason": "stop",
        }

    monkeypatch.setattr("gateway.runtime.ollama_runtime.complete", fake_chat)
    _stub_ollama(monkeypatch, models=[{"name": "tinyllama", "size_gb": 0.6}])
    _fake_llmfit(monkeypatch, tmp_path)

    def fake_merge(**kwargs: Any) -> dict[str, Any]:
        return {
            "system": {},
            "rows": [
                {
                    "name": "tinyllama",
                    "ollama_name": "tinyllama",
                    "state": "INSTALLED",
                    "gguf_sources": [],
                    "supports_tools": True,
                    "FITS": "perfect",
                    "chat_template_ok": True,
                }
            ],
            "caches": {},
        }

    monkeypatch.setattr("gateway.app.merge_catalog", fake_merge)
    client = _client(tmp_path)
    pull = client.post("/api/catalog/pull", json={"name": "tinyllama", "ollama_name": "tinyllama"})
    assert pull.status_code == 200
    job_id = pull.json()["job_id"]
    deadline = time.time() + 5
    while time.time() < deadline:
        p = client.get(f"/api/catalog/pull/{job_id}").json()
        if p.get("done"):
            break
        time.sleep(0.05)
    assert client.get(f"/api/catalog/pull/{job_id}").json().get("status") == "success"
    assert client.post("/api/ingest").status_code == 200
    assert client.post("/api/model", json={"provider": "ollama", "name": "tinyllama"}).status_code == 200
    asked = client.post(
        "/api/chat",
        json={"message": "What is the project code token in alpha.md? Quote it exactly.", "session_id": "rag"},
    )
    assert asked.status_code == 200, asked.text
    assert NEEDLE in (asked.json().get("text") or "")


def test_llamacpp_flag_default_off() -> None:
    assert LLAMACPP_ENABLED is False
    st = llamacpp_status()
    assert st["flag"] is False
    assert "docs" in st
    assert "compile" in (st.get("note") or "").lower() or st.get("present") is not None
