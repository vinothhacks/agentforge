"""Ollama runtime: list / pull / chat / unload. Progress + cancel + resume."""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any

import httpx

from gateway.bins import which_tool
from gateway.catalog import is_sharded_gguf
from gateway.providers import complete

OLLAMA_BASE = "http://127.0.0.1:11434"

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def normalize_tag(name: str) -> str:
    """`llama3.2` and `llama3.2:latest` are the same model to Ollama.

    llmfit imports pin the bare form while /api/tags reports `:latest`, so an
    exact string compare silently misses. Normalise both sides.
    """
    n = (name or "").strip()
    if not n or ":" in n:
        return n
    return n + ":latest"


class PullRejected(ValueError):
    pass


class OllamaRuntime:
    def __init__(self, base: str = OLLAMA_BASE):
        self.base = base.rstrip("/")

    def list(self) -> list[dict[str, Any]]:
        return self.list_or_none() or []

    def list_or_none(self) -> list[dict[str, Any]] | None:
        """None when Ollama could not be reached, as opposed to an empty list."""
        try:
            r = httpx.get(f"{self.base}/api/tags", timeout=5.0)
            r.raise_for_status()
            return list((r.json() or {}).get("models") or [])
        except Exception:  # noqa: BLE001
            return None

    def has_tag(self, name: str) -> bool | None:
        """True/False when Ollama answers, None when it cannot be reached."""
        models = self.list_or_none()
        if models is None:
            return None
        want = normalize_tag(name)
        for m in models:
            have = str(m.get("name") or m.get("model") or "")
            if have and normalize_tag(have) == want:
                return True
        return False

    def ensure_available(self, name: str, *, gguf_sources: Any = None) -> dict[str, Any]:
        if is_sharded_gguf(gguf_sources):
            raise PullRejected("sharded GGUF repo; pull refused")
        names = {m.get("name") or m.get("model") for m in self.list()}
        if name in names:
            return {"ok": True, "already": True, "name": name}
        return self.pull_start(name)

    def pull_start(self, name: str, *, gguf_sources: Any = None) -> dict[str, Any]:
        if is_sharded_gguf(gguf_sources):
            raise PullRejected("sharded GGUF repo; pull refused")
        job_id = uuid.uuid4().hex[:12]
        cancel = threading.Event()
        job = {
            "id": job_id,
            "name": name,
            "status": "starting",
            "percent": 0.0,
            "log": "",
            "error": None,
            "done": False,
            "cancel": cancel,
        }
        with _JOBS_LOCK:
            _JOBS[job_id] = job
        threading.Thread(target=self._pull_worker, args=(job_id, name, cancel), daemon=True).start()
        return {"ok": True, "job_id": job_id, "name": name, "resumed": False}

    def pull_progress(self, job_id: str) -> dict[str, Any]:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if not job:
                return {"ok": False, "reason": "unknown_job"}
            return {
                "ok": True,
                "job_id": job_id,
                "name": job["name"],
                "status": job["status"],
                "percent": job["percent"],
                "log": job["log"][-2000:],
                "error": job["error"],
                "done": job["done"],
            }

    def pull_cancel(self, job_id: str) -> dict[str, Any]:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if not job:
                return {"ok": False, "reason": "unknown_job"}
            job["cancel"].set()
            job["status"] = "cancelling"
        return {"ok": True, "job_id": job_id}

    def pull_resume(self, name: str) -> dict[str, Any]:
        """Ollama pull is resumable; start again on the same tag."""
        out = self.pull_start(name)
        out["resumed"] = True
        return out

    def _pull_worker(self, job_id: str, name: str, cancel: threading.Event) -> None:
        try:
            with httpx.stream(
                "POST",
                f"{self.base}/api/pull",
                json={"name": name, "stream": True},
                timeout=600.0,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if cancel.is_set():
                        with _JOBS_LOCK:
                            _JOBS[job_id]["status"] = "cancelled"
                            _JOBS[job_id]["done"] = True
                        return
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Ollama answers 200 and reports failure inside the stream:
                    # {"error":"pull model manifest: file does not exist"}.
                    # raise_for_status() never sees it.
                    if ev.get("error"):
                        with _JOBS_LOCK:
                            _JOBS[job_id]["status"] = "error"
                            _JOBS[job_id]["error"] = str(ev["error"])
                            _JOBS[job_id]["done"] = True
                        return
                    total = float(ev.get("total") or 0)
                    completed = float(ev.get("completed") or 0)
                    pct = (completed / total * 100.0) if total else 0.0
                    with _JOBS_LOCK:
                        _JOBS[job_id]["status"] = ev.get("status") or "pulling"
                        _JOBS[job_id]["percent"] = round(pct, 1)
                        _JOBS[job_id]["log"] += (ev.get("status") or "") + "\n"
            # A stream can also end early (network drop) having reported
            # nothing. Success is the tag actually resolving, not the loop
            # exiting: confirm against /api/tags before declaring victory.
            # `None` means we could not ask; only a definite "absent" fails
            # the job, so an unreachable Ollama does not mask a good pull.
            if self.has_tag(name) is False:
                with _JOBS_LOCK:
                    _JOBS[job_id]["status"] = "error"
                    _JOBS[job_id]["error"] = (
                        f"pull finished but {name!r} is not in Ollama's model list"
                    )
                    _JOBS[job_id]["done"] = True
                return
            with _JOBS_LOCK:
                _JOBS[job_id]["status"] = "success"
                _JOBS[job_id]["percent"] = 100.0
                _JOBS[job_id]["done"] = True
        except Exception as exc:  # noqa: BLE001
            with _JOBS_LOCK:
                _JOBS[job_id]["status"] = "error"
                _JOBS[job_id]["error"] = str(exc)
                _JOBS[job_id]["done"] = True

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        kwargs = dict(kwargs)
        kwargs["provider"] = "ollama"
        return complete(**kwargs)

    def unload(self, name: str) -> dict[str, Any]:
        try:
            r = httpx.post(
                f"{self.base}/api/generate",
                json={"model": name, "keep_alive": 0, "prompt": ""},
                timeout=30.0,
            )
            return {"ok": r.status_code < 400, "status": r.status_code}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}


def ollama_on_path() -> bool:
    return which_tool("ollama") is not None
