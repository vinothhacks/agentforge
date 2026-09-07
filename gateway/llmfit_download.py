"""llmfit download jobs: GGUF to cache, then ollama create (local tag, not a guessed hub name)."""

from __future__ import annotations

import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from gateway.bins import which_tool
from gateway.catalog import cache_dirs, gguf_downloadable, is_sharded_gguf
from gateway.llmfit_client import _invoke, find_llmfit

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
GGUF_LINE_RE = re.compile(r"([\w.\-]+\.gguf)", re.I)
IDLE_TIMEOUT_S = 120.0


def precheck_gguf_repo(name: str, *, gguf_sources: Any = None, timeout: float = 90.0) -> tuple[bool, str]:
    """Run llmfit download --list --no-dashboard; fail fast when repo has no GGUF."""
    if not gguf_downloadable(name, gguf_sources):
        return False, f"Repository '{name}' does not look like a GGUF repo (try a *-GGUF HuggingFace repo)"
    exe = find_llmfit()
    cmd = _invoke(exe, ["--no-dashboard", "download", "--list", name.strip()])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return False, f"Timed out checking GGUF files for '{name}'"
    blob = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    lower = blob.lower()
    if proc.returncode != 0:
        if "no gguf" in lower or "not found" in lower or proc.returncode == 1:
            return False, f"No GGUF files found in repository '{name}'"
        return False, blob[-800:] or f"llmfit --list exit {proc.returncode}"
    files = GGUF_LINE_RE.findall(blob)
    if not files:
        return False, f"No GGUF files found in repository '{name}'"
    return True, f"{len(files)} GGUF file(s) available"


class DownloadRejected(ValueError):
    pass


def local_ollama_tag(hf_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (hf_name or "model").lower()).strip("-")
    return ("af-" + slug)[:80].strip("-") or "af-model"


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def strip_ansi(text: str) -> str:
    """llmfit's progress line carries erase-line escapes; they render as boxes."""
    return ANSI_RE.sub("", text or "")


def _append(job_id: str, line: str, percent: float | None = None) -> None:
    clean = strip_ansi(line)
    with _LOCK:
        job = _JOBS[job_id]
        job["log"] += clean + "\n"
        if percent is not None:
            job["percent"] = max(float(job.get("percent") or 0), min(99.0, percent))
        job["status"] = clean.strip()[:80] or job["status"]


def _newest_gguf(root: Path, since: float) -> Path | None:
    files = [p for p in root.rglob("*.gguf") if p.is_file() and p.stat().st_mtime >= since - 1]
    if not files:
        files = [p for p in root.rglob("*.gguf") if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def ollama_create_from_gguf(gguf: Path, tag: str) -> None:
    exe = which_tool("ollama")
    if not exe:
        raise DownloadRejected("Ollama is not installed; downloaded the GGUF but cannot import it.")
    modelfile = gguf.parent / f"Modelfile.{tag.replace('/', '-')}"
    src = str(gguf.resolve()).replace("\\", "/")
    modelfile.write_text(f"FROM {src}\n", encoding="utf-8")
    proc = subprocess.run(
        [exe, "create", tag, "-f", str(modelfile)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or f"ollama create exit {proc.returncode}")[-1500:]
        raise DownloadRejected(err)


def start_download(name: str, *, quant: str | None = None, gguf_sources: Any = None) -> dict[str, Any]:
    if is_sharded_gguf(gguf_sources):
        raise DownloadRejected("sharded GGUF repo; pull refused")
    ok, detail = precheck_gguf_repo(name, gguf_sources=gguf_sources)
    if not ok:
        raise DownloadRejected(detail)
    job_id = uuid.uuid4().hex[:12]
    cancel = threading.Event()
    tag = local_ollama_tag(name)
    job = {
        "id": job_id,
        "name": name,
        "ollama_tag": tag,
        "status": "starting",
        "percent": 0.0,
        "log": "",
        "error": None,
        "done": False,
        "gguf": None,
        "cancel": cancel,
        "proc": None,
    }
    with _LOCK:
        _JOBS[job_id] = job
    threading.Thread(target=_worker, args=(job_id, name, quant, tag, cancel), daemon=True).start()
    return {"ok": True, "job_id": job_id, "name": name, "ollama_tag": tag}


def progress(job_id: str) -> dict[str, Any]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return {"ok": False, "reason": "unknown_job"}
        return {
            "ok": True,
            "job_id": job_id,
            "name": job["name"],
            "ollama_tag": job["ollama_tag"],
            "status": job["status"],
            "percent": job["percent"],
            "log": job["log"][-2000:],
            "error": job["error"],
            "done": job["done"],
            "gguf": job["gguf"],
        }


def cancel(job_id: str) -> dict[str, Any]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return {"ok": False, "reason": "unknown_job"}
        job["cancel"].set()
        proc = job.get("proc")
        job["status"] = "cancelling"
    if proc and proc.poll() is None:
        proc.kill()
    return {"ok": True, "job_id": job_id}


def _worker(job_id: str, name: str, quant: str | None, tag: str, cancel: threading.Event) -> None:
    out_dir = cache_dirs()["llmfit"]
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    last_output_time = time.time()
    captured_logs: list[str] = []
    try:
        exe = find_llmfit()
        args = ["--no-dashboard", "download", name]
        if quant:
            args.extend(["-q", quant])
        args.extend(["--output-dir", str(out_dir)])
        cmd = _invoke(exe, args)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        with _LOCK:
            _JOBS[job_id]["proc"] = proc

        def _idle_watch() -> None:
            while proc.poll() is None and not cancel.is_set():
                time.sleep(5.0)
                with _LOCK:
                    job = _JOBS.get(job_id) or {}
                if job.get("done") or cancel.is_set():
                    return
                pct = float(job.get("percent") or 0)
                if pct <= 0 and time.time() - started > IDLE_TIMEOUT_S:
                    proc.kill()
                    return

        threading.Thread(target=_idle_watch, daemon=True).start()
        assert proc.stdout is not None
        for line in proc.stdout:
            last_output_time = time.time()
            if cancel.is_set():
                proc.kill()
                with _LOCK:
                    _JOBS[job_id]["status"] = "cancelled"
                    _JOBS[job_id]["done"] = True
                return
            text = (line or "").rstrip()
            if text:
                captured_logs.append(text)
            m = PCT_RE.search(text)
            pct = float(m.group(1)) if m else None
            _append(job_id, text, pct)
            if "no gguf files found" in text.lower() or "make sure this is a valid gguf repository" in text.lower():
                proc.kill()
                raise DownloadRejected(f"No GGUF files found in repository '{name}'")
        rc = proc.wait(timeout=10)
        if cancel.is_set():
            with _LOCK:
                _JOBS[job_id]["status"] = "cancelled"
                _JOBS[job_id]["done"] = True
            return
        with _LOCK:
            pct = float(_JOBS[job_id].get("percent") or 0)
        if rc != 0 and pct <= 0 and not captured_logs:
            raise DownloadRejected(f"No GGUF files found in repository '{name}'")
        if rc != 0:
            err_summary = "\n".join(captured_logs[-5:]) if captured_logs else f"llmfit download exit {rc}"
            raise DownloadRejected(err_summary)
        gguf = _newest_gguf(out_dir, started)
        if not gguf:
            raise DownloadRejected("llmfit download finished but no .gguf file was found")
        _append(job_id, f"importing into Ollama as {tag}", 99.0)
        ollama_create_from_gguf(gguf, tag)
        with _LOCK:
            _JOBS[job_id]["gguf"] = str(gguf)
            _JOBS[job_id]["status"] = "success"
            _JOBS[job_id]["percent"] = 100.0
            _JOBS[job_id]["done"] = True
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            if _JOBS[job_id]["done"]:
                return
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["error"] = str(exc)
            _JOBS[job_id]["done"] = True
