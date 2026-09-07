"""llmfit subprocess adapter: --json only. Never scrape the TUI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
import time

INSTALL_HINT = (
    "llmfit not found. Install: uv tool install llmfit "
    "or place llmfit.exe on PATH / %USERPROFILE%\\.cargo\\bin"
)

FIT_ROW_REQUIRED = (
    "name",
    "ollama_name",
    "fit_level",
    "best_quant",
    "estimated_tps",
    "parameter_count",
)


class LlmfitMissing(FileNotFoundError):
    pass


class LlmfitSchemaError(ValueError):
    pass


class LlmfitError(RuntimeError):
    pass


def find_llmfit() -> Path:
    env = os.environ.get("LLMFIT_BIN")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise LlmfitMissing(INSTALL_HINT)
    which = shutil.which("llmfit")
    if which:
        return Path(which)
    home = Path.home()
    candidates = [
        home / ".cargo" / "bin" / "llmfit.exe",
        home / ".cargo" / "bin" / "llmfit",
        home / ".local" / "bin" / "llmfit.exe",
        home / ".local" / "bin" / "llmfit",
        Path("target") / "release" / "llmfit.exe",
        Path("target") / "release" / "llmfit",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise LlmfitMissing(INSTALL_HINT)


def parse_json_blob(text: str) -> Any:
    raw = text or ""
    brace = raw.find("{")
    brack = raw.find("[")
    starts = [i for i in (brace, brack) if i >= 0]
    if not starts:
        raise LlmfitSchemaError("llmfit produced no JSON")
    try:
        return json.loads(raw[min(starts) :])
    except json.JSONDecodeError as exc:
        raise LlmfitSchemaError(f"llmfit JSON parse failed: {exc}") from exc


def _invoke(exe: Path, rest: list[str]) -> list[str]:
    if exe.suffix.lower() == ".py":
        return [sys.executable, str(exe), *rest]
    return [str(exe), *rest]


def run_json(args: list[str], *, timeout: float = 90.0) -> Any:
    exe = find_llmfit()
    cmd = _invoke(exe, ["--json", "--no-dashboard", *args])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise LlmfitError(f"llmfit timed out: {args}") from exc
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        if "no-dashboard" in blob.lower():
            cmd = _invoke(exe, ["--json", *args])
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
            blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            raise LlmfitError(blob[-1500:] or f"llmfit exit {proc.returncode}")
    return parse_json_blob(blob)


def validate_fit_payload(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or "models" not in data:
        raise LlmfitSchemaError("fit JSON missing models[]")
    models = data.get("models")
    if not isinstance(models, list):
        raise LlmfitSchemaError("fit JSON models is not a list")
    for row in models:
        if not isinstance(row, dict) or "name" not in row:
            raise LlmfitSchemaError("fit row missing name")
        if "ollama_name" not in row:
            raise LlmfitSchemaError("fit row missing ollama_name (do not invent tags)")
    return data


def system(timeout: float = 30.0) -> dict[str, Any]:
    data = run_json(["system"], timeout=timeout)
    if not isinstance(data, dict):
        raise LlmfitSchemaError("system JSON is not an object")
    return data.get("system") or data


def fit(*, limit: int = 40, memory: str | None = None, max_context: int | None = None) -> dict[str, Any]:
    # --memory / --max-context are global flags (llmfit 1.1.x). -n is the probed fit limit.
    extra: list[str] = []
    if memory:
        extra.extend(["--memory", str(memory)])
    if max_context:
        extra.extend(["--max-context", str(int(max_context))])
    args: list[str] = [*extra, "fit", "-n", str(limit)]
    data = run_json(args, timeout=120.0)
    return validate_fit_payload(data)


_LIST_CACHE: dict[str, Any] = {"t": 0.0, "rows": None}


def list_models(*, timeout: float = 180.0, refresh: bool = False) -> list[dict[str, Any]]:
    """Full embedded catalog. JSON array. No ollama_name — do not invent tags."""
    now = time.time()
    cached = _LIST_CACHE.get("rows")
    if cached is not None and not refresh and now - float(_LIST_CACHE["t"] or 0) < 600:
        return cached
    data = run_json(["list"], timeout=timeout)
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        rows = data["models"]
    elif isinstance(data, list):
        rows = data
    else:
        raise LlmfitSchemaError("list JSON is not an array")
    rows = [r for r in rows if isinstance(r, dict) and r.get("name")]
    _LIST_CACHE["t"] = now
    _LIST_CACHE["rows"] = rows
    return rows


def bench(model: str, *, provider: str = "ollama", timeout: float = 180.0) -> dict[str, Any]:
    data = run_json(["bench", model, "--provider", provider], timeout=timeout)
    if not isinstance(data, dict):
        raise LlmfitSchemaError("bench JSON is not an object")
    return data
