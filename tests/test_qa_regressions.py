"""Regressions for the defects in docs/qa-report-2026-08-31.md.

Every one of these reproduced on a tree where `pytest` was 142 green, so each
test pins behaviour the suite previously had no opinion about. Section numbers
refer to the report.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gateway.app import make_app
from gateway.catalog import browse_llmfit
from gateway.db import Store
from gateway.evals import enumeration_recall, score_run
from gateway.llmfit_download import strip_ansi
from gateway.packs.rag import HybridIndex
from gateway.runtime.ollama_runtime import OllamaRuntime, normalize_tag
from gateway.runtime.tools import dispatch
from gateway.stats import UNMEASURED_LO95, effective_lo95, max_steps_from_lo95


def _client(tmp_path: Path) -> TestClient:
    return TestClient(make_app(tmp_path, Store(tmp_path / "app.sqlite")))


def _card(tmp_path: Path) -> dict[str, Any]:
    return json.loads(Store(tmp_path / "app.sqlite").get_setting("card") or "{}")


def _pin(tmp_path: Path) -> dict[str, Any]:
    return json.loads(Store(tmp_path / "app.sqlite").get_setting("model_pin") or "{}")


# --- 1: a probe measures the pin, it never replaces it ---------------------


def test_probe_does_not_overwrite_the_users_pin(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path)
    client.post("/api/model", json={"provider": "ollama", "name": "my-local-model"})
    assert _pin(tmp_path)["name"] == "my-local-model"

    monkeypatch.setattr(
        "gateway.app.run_probe",
        lambda **kw: {
            "verdict": "chat_only",
            "cause": "timeout",
            "model_pin": {"provider": "openrouter", "name": "openai/gpt-4o-mini"},
            "measured": {"per_step_success_lo95": 0.0},
        },
    )
    client.post("/api/probe", json={"provider": "openrouter", "name": "openai/gpt-4o-mini"})

    # The pin the user chose survives a probe aimed at something else.
    assert _pin(tmp_path)["name"] == "my-local-model"


def test_probe_defaults_to_the_active_pin(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path)
    client.post("/api/model", json={"provider": "ollama", "name": "pinned-model"})
    seen: dict[str, Any] = {}

    def fake_probe(**kw: Any) -> dict[str, Any]:
        seen.update(kw)
        pin = {"provider": kw["provider"], "name": kw["name"], "digest": "d"}
        return {"verdict": "agent", "cause": None, "model_pin": pin, "measured": {}}

    monkeypatch.setattr("gateway.app.run_probe", fake_probe)
    r = client.post("/api/probe", json={})
    assert r.status_code == 200
    assert seen["name"] == "pinned-model"
    assert seen["provider"] == "ollama"
    assert r.json()["probed"] == {"provider": "ollama", "name": "pinned-model"}


def test_chat_only_verdict_is_recoverable(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path)
    client.post("/api/model", json={"provider": "ollama", "name": "pinned-model"})
    monkeypatch.setattr(
        "gateway.app.run_probe",
        lambda **kw: {
            "verdict": "chat_only",
            "cause": "timeout",
            "model_pin": {"provider": "ollama", "name": "pinned-model"},
            "measured": {"per_step_success_lo95": 0.0},
        },
    )
    client.post("/api/probe", json={})
    assert _card(tmp_path)["verdict"] == "chat_only"

    assert client.post("/api/probe/clear").status_code == 200
    assert _card(tmp_path)["verdict"] == "agent"


# --- 2: a failed pull must not report success ------------------------------


class _FakeStream:
    def __init__(self, events: list[dict[str, Any]]):
        self.events = events

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        for ev in self.events:
            yield json.dumps(ev)


def _drain(rt: OllamaRuntime, job_id: str, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        prog = rt.pull_progress(job_id)
        if prog.get("done"):
            return prog
        time.sleep(0.02)
    raise AssertionError("pull job never finished")


def test_error_inside_the_pull_stream_fails_the_job(monkeypatch) -> None:
    # Ollama answers HTTP 200 and reports the failure in the body.
    events = [
        {"status": "pulling manifest"},
        {"error": "pull model manifest: file does not exist"},
    ]
    monkeypatch.setattr(
        "gateway.runtime.ollama_runtime.httpx.stream",
        lambda *a, **k: _FakeStream(events),
    )
    monkeypatch.setattr(OllamaRuntime, "list_or_none", lambda self: [])
    rt = OllamaRuntime()
    prog = _drain(rt, rt.pull_start("definitely-not-a-real-model-xyz")["job_id"])
    assert prog["status"] == "error"
    assert "does not exist" in (prog["error"] or "")
    assert prog["percent"] != 100.0


def test_pull_that_never_produced_the_tag_fails(monkeypatch) -> None:
    # A stream cut short exits the loop cleanly; the tag is the real evidence.
    monkeypatch.setattr(
        "gateway.runtime.ollama_runtime.httpx.stream",
        lambda *a, **k: _FakeStream([{"status": "pulling", "total": 10, "completed": 4}]),
    )
    monkeypatch.setattr(OllamaRuntime, "list_or_none", lambda self: [{"name": "other:latest"}])
    rt = OllamaRuntime()
    prog = _drain(rt, rt.pull_start("ghost-model")["job_id"])
    assert prog["status"] == "error"
    assert "ghost-model" in (prog["error"] or "")


def test_unreachable_ollama_does_not_fail_a_good_pull(monkeypatch) -> None:
    monkeypatch.setattr(
        "gateway.runtime.ollama_runtime.httpx.stream",
        lambda *a, **k: _FakeStream([{"status": "success", "total": 10, "completed": 10}]),
    )
    monkeypatch.setattr(OllamaRuntime, "list_or_none", lambda self: None)
    rt = OllamaRuntime()
    prog = _drain(rt, rt.pull_start("some-model")["job_id"])
    assert prog["status"] == "success"


# --- 3: the grader must not invent a term ----------------------------------


class _StubIndex:
    """grep_paths answers for `demurrage` only, like the sample corpus."""

    def grep_paths(self, term: str) -> list[str]:
        return ["DOC-001.txt", "DOC-004.txt"] if term.lower() == "demurrage" else []


@pytest.mark.parametrize(
    "question",
    [
        "List the files in this folder.",
        "Give me the checklist for berth windows",
        "Can you listen to my request and summarise DOC-002?",
        "Show me a playlist",
    ],
)
def test_questions_containing_list_are_not_graded_on_a_fixture_word(question: str) -> None:
    out = score_run(
        index=_StubIndex(),
        question=question,
        answer="Here is the answer.",
        tool_paths=[],
        usage={},
        lo95=0.5,
    )
    assert out["enumeration"] is None
    assert out["pass"] is True


def test_an_explicit_term_is_still_graded() -> None:
    out = score_run(
        index=_StubIndex(),
        question="Which files mention demurrage?",
        answer="DOC-001.txt and DOC-004.txt",
        tool_paths=[],
        usage={},
        lo95=0.5,
    )
    assert out["enumeration"] is not None
    assert out["enumeration"]["term"] == "demurrage"
    assert out["enumeration"]["recall"] == 1.0


def test_empty_ground_truth_still_reports_precision() -> None:
    out = enumeration_recall(_StubIndex(), "nothingmatchesthis", ["a.txt"])
    assert "precision" in out


# --- 5: the llmfit browser must be able to hide dead rows ------------------


def _llmfit_rows(monkeypatch, rows: list[dict[str, Any]]) -> None:
    monkeypatch.setattr("gateway.catalog.llmfit_client.system", lambda: {"total_ram_gb": 32.0})
    monkeypatch.setattr("gateway.catalog.llmfit_client.list_models", lambda refresh=False: rows)
    monkeypatch.setattr("gateway.catalog.already_downloaded_names", lambda: set())


def test_downloadable_filter_hides_unresolvable_rows(monkeypatch) -> None:
    _llmfit_rows(
        monkeypatch,
        [
            {"name": "org/plain-repo", "gguf_sources": [], "min_ram_gb": 2},
            {"name": "org/has-gguf-GGUF", "gguf_sources": ["model.Q4_K_M.gguf"], "min_ram_gb": 2},
        ],
    )
    every = browse_llmfit(limit=50)
    only = browse_llmfit(downloadable=True, limit=50)
    assert every["total"] > only["total"]
    assert all(r["state"] != "UNRESOLVABLE" for r in only["rows"])
    assert only["filters"]["downloadable"] == only["total"]


def test_a_cache_hit_with_nothing_runnable_is_not_installed(monkeypatch) -> None:
    # The cache name match is fuzzy, so a safetensors-only repo could collide
    # with a downloaded GGUF and render as READY with no usable action.
    _llmfit_rows(monkeypatch, [{"name": "meta-llama/Llama-3.2-1B-Instruct", "gguf_sources": []}])
    monkeypatch.setattr(
        "gateway.catalog.already_downloaded_names", lambda: {"llama-3.2-1b-instruct"}
    )
    row = browse_llmfit(limit=5)["rows"][0]
    assert row["state"] == "UNRESOLVABLE"
    assert "GGUF" in (row["reason"] or "")


def test_score_sort_puts_actionable_rows_first(monkeypatch) -> None:
    _llmfit_rows(
        monkeypatch,
        [
            # Better fit, but nothing the user can do with it.
            {"name": "org/dead-perfect", "gguf_sources": [], "min_ram_gb": 1},
            {"name": "org/live-GGUF", "gguf_sources": ["m.Q4_K_M.gguf"], "min_ram_gb": 8},
        ],
    )
    rows = browse_llmfit(sort="score", limit=50)["rows"]
    assert rows[0]["state"] != "UNRESOLVABLE"


# --- 6: bare tags and :latest are the same model ---------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("af-model", "af-model:latest"),
        ("llama3.2:1b", "llama3.2:1b"),
        ("", ""),
    ],
)
def test_normalize_tag_matches_latest(a: str, b: str) -> None:
    assert normalize_tag(a) == normalize_tag(b)


def test_has_tag_matches_a_bare_pin_against_latest(monkeypatch) -> None:
    monkeypatch.setattr(
        OllamaRuntime, "list_or_none", lambda self: [{"name": "af-unsloth-llama:latest"}]
    )
    assert OllamaRuntime().has_tag("af-unsloth-llama") is True
    assert OllamaRuntime().has_tag("something-else") is False


# --- 7: no control characters in the progress line -------------------------


def test_ansi_escapes_are_stripped() -> None:
    dirty = "\x1b[K  66.2% - Downloading 0.5/0.8 GB"
    assert strip_ansi(dirty) == "  66.2% - Downloading 0.5/0.8 GB"
    assert strip_ansi("plain") == "plain"


# --- 9: an unprobed pin is unmeasured, not 0.5 -----------------------------


def test_unprobed_pin_is_not_reported_as_measured(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.post("/api/model", json={"provider": "ollama", "name": "never-probed"})
    measured = _card(tmp_path)["measured"]
    assert measured["per_step_success_lo95"] is None
    assert measured["measured"] is False


def test_unmeasured_lo95_still_yields_a_working_budget() -> None:
    assert effective_lo95(None) == UNMEASURED_LO95
    assert max_steps_from_lo95(effective_lo95(None)) >= 3
    assert effective_lo95("not a number") == UNMEASURED_LO95
    assert effective_lo95(0.72) == 0.72


# --- 10: a misaddressed tool call recovers, and says so in English ---------


def test_args_that_fit_a_sibling_tool_are_rerouted() -> None:
    called: list[str] = []

    def exec_(name: str, args: dict[str, Any]) -> dict[str, Any]:
        called.append(name)
        return {"hits": []}

    # rag_search's arguments, addressed to fs_list -- the observed 1B failure.
    out = dispatch("fs_list", {"query": "demurrage", "limit": 80}, exec_)
    assert called == ["rag_search"]
    assert out["rerouted_from"] == "fs_list"


def test_missing_arguments_are_not_rerouted() -> None:
    called: list[str] = []

    def exec_(name: str, args: dict[str, Any]) -> dict[str, Any]:
        called.append(name)
        return {}

    out = dispatch("rag_search", {}, exec_)
    assert called == []
    assert out["error"] == "schema_reject"


def test_a_term_bearing_question_falls_back_to_retrieval_not_a_listing() -> None:
    from gateway.runtime.loop import enumeration_term, wants_file_list

    # Both must be true for the fallback to reach the retrieval branch.
    assert wants_file_list("Which files mention demurrage?") is True
    assert enumeration_term("Which files mention demurrage?") == "demurrage"
    # A plain listing question names no term and still gets fs_list.
    assert enumeration_term("what files are in this folder") is None


def test_a_repeated_identical_tool_call_is_rendered_once() -> None:
    # Observed live on granite3.1-moe: the model called rag_search twice with
    # identical args across two steps and the whole answer printed twice.
    from gateway.runtime.loop import summarize_results

    call = {
        "tool": "rag_search",
        "args": {"query": "demurrage", "limit": 5},
        "result": {"hits": [{"path": "DOC-001.txt", "page": 1, "text": "Demurrage ..."}]},
    }
    once = summarize_results([call])
    twice = summarize_results([call, dict(call)])
    assert twice == once, "an identical repeated call must not duplicate the answer"

    # A genuinely different call still contributes.
    other = dict(call, args={"query": "berth", "limit": 5})
    assert summarize_results([call, other]) != once


def test_schema_reject_carries_plain_english() -> None:
    out = dispatch("fs_read", {"nonsense": 1, "also": 2}, lambda n, a: {})
    assert out["error"] == "schema_reject"
    assert "`fs_read`" in out["detail"]
    assert "`path`" in out["detail"]


# --- 4: files the agent cannot see are marked, and re-index fixes it -------


def test_files_added_after_launch_are_flagged_until_reindexed(tmp_path: Path) -> None:
    (tmp_path / "alpha.md").write_text("Unique marker: quintessorial", encoding="utf-8")
    client = _client(tmp_path)
    client.post("/api/ingest")

    (tmp_path / "beta.md").write_text("A second marker: zephyrine", encoding="utf-8")
    rows = {f["path"]: f for f in client.get("/api/files").json()["files"]}
    assert rows["alpha.md"]["indexed"] is True
    assert rows["beta.md"]["indexed"] is False, "a file the agent cannot search must say so"

    assert client.post("/api/ingest").status_code == 200
    rows = {f["path"]: f for f in client.get("/api/files").json()["files"]}
    assert rows["beta.md"]["indexed"] is True
    assert HybridIndex(tmp_path).grep_paths("zephyrine") == ["beta.md"]


# --- low: session reset and favicon ----------------------------------------


def test_session_reset_clears_history_and_usage(tmp_path: Path) -> None:
    client = _client(tmp_path)
    store = Store(tmp_path / "app.sqlite")
    store.save_session("default", str(tmp_path), [{"role": "user", "content": "hi"}])
    store.add_usage("default", 100, 20, 0.01)

    assert client.post("/api/session/reset", json={"session_id": "default"}).status_code == 200
    fresh = Store(tmp_path / "app.sqlite")
    assert fresh.load_session("default") == []
    assert fresh.get_usage("default")["prompt_tokens"] == 0


def test_a_fresh_workspace_does_not_write_without_asking(tmp_path: Path) -> None:
    # The shipped template overrode the code default of `deny` with `allow`,
    # so a first run could overwrite files in the user's document folder with
    # no confirmation at all.
    perms = _client(tmp_path).get("/api/permissions").json()
    assert perms["fs_write"] in {"deny", "ask"}, "a fresh workspace must not write unprompted"


def test_favicon_does_not_404(tmp_path: Path) -> None:
    r = _client(tmp_path).get("/favicon.ico")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")
