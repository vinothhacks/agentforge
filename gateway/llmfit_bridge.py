"""llmfit + Ollama: hardware-fit models the user can actually pin."""

from __future__ import annotations

import json
import re
import subprocess
import time
from typing import Any

import httpx

from gateway.catalog import merge_catalog
from gateway.bins import which_tool

SKIP_CATEGORIES = {"embedding", "embeddings", "tts", "audio", "rerank", "reranker"}
MIN_PARAMS_B = 0.5

OLLAMA_SUGGESTIONS: list[dict[str, Any]] = []  # replaced by merge_catalog / ollama_name rows


def parse_json_blob(text: str) -> dict[str, Any] | None:
    start = (text or "").find("{")
    if start < 0:
        return None
    try:
        data = json.loads(text[start:])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def run_llmfit(args: list[str], timeout: float = 90.0) -> dict[str, Any] | None:
    exe = which_tool("llmfit")
    if not exe:
        return None
    cmd = [exe, "--json", "--no-dashboard", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    err = (proc.stderr or "") + (proc.stdout or "")
    if proc.returncode != 0 and "no-dashboard" in err.lower():
        try:
            proc = subprocess.run([exe, "--json", *args], capture_output=True, text=True, timeout=timeout, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
    blob = parse_json_blob((proc.stdout or "") + "\n" + (proc.stderr or ""))
    if proc.returncode != 0 and blob is None:
        return None
    return blob


def system_specs() -> dict[str, Any] | None:
    data = run_llmfit(["system"], timeout=30)
    if not data:
        return None
    return data.get("system") or data


def available_ram_gb(system: dict[str, Any] | None) -> float:
    if not system:
        return 0.0
    return float(system.get("available_ram_gb") or system.get("total_ram_gb") or 0.0)


def fit_label(size_gb: float, available_gb: float) -> str:
    if size_gb <= 0 or available_gb <= 0:
        return "unknown"
    ratio = size_gb / available_gb
    if ratio <= 0.45:
        return "perfect"
    if ratio <= 0.75:
        return "good"
    if ratio <= 0.95:
        return "marginal"
    if ratio <= 1.15:
        return "tight"
    return "wont_fit"


def _params_b(row: dict[str, Any]) -> float:
    raw = row.get("params_b")
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(row.get("parameter_count") or "")
    m = re.search(r"([\d.]+)\s*([MBT])", text, re.I)
    if not m:
        return 0.0
    n = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "M":
        return n / 1000.0
    if unit == "T":
        return n * 1000.0
    return n


def usable_recommend_row(row: dict[str, Any]) -> bool:
    cat = str(row.get("category") or row.get("use_case") or "").lower()
    if any(s in cat for s in SKIP_CATEGORIES):
        return False
    name = str(row.get("name") or "").lower()
    if "embed" in name or "rerank" in name:
        return False
    if str(row.get("runtime") or "").lower() == "mlx":
        return False
    return _params_b(row) >= MIN_PARAMS_B


def summarize_model(row: dict[str, Any]) -> dict[str, Any]:
    caps = row.get("capability_ids") or []
    if not caps and isinstance(row.get("capabilities"), list):
        caps = [str(c).lower().replace(" ", "_") for c in row["capabilities"]]
    return {
        "name": row.get("name"),
        "ollama_name": row.get("ollama_name"),
        "provider": row.get("provider"),
        "category": row.get("category"),
        "fit_level": row.get("fit_level") or row.get("fit_label"),
        "score": row.get("score"),
        "estimated_tps": row.get("estimated_tps"),
        "best_quant": row.get("best_quant"),
        "memory_required_gb": row.get("memory_required_gb"),
        "parameter_count": row.get("parameter_count"),
        "params_b": _params_b(row),
        "runtime": row.get("runtime"),
        "run_mode": row.get("run_mode"),
        "capabilities": caps,
        "capability_ids": [str(c).lower().replace(" ", "_") for c in (row.get("capability_ids") or caps)],
        "tool_use": "tool_use" in {str(c).lower().replace(" ", "_") for c in (row.get("capability_ids") or caps)},
        "notes": row.get("notes") or [],
    }


def recommend_chat(limit: int = 16) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    queries = [
        ["recommend", "-n", str(limit), "--use-case", "chat", "--min-fit", "marginal"],
        ["recommend", "-n", str(limit), "--use-case", "general", "--capability", "tool_use", "--min-fit", "marginal"],
    ]
    for args in queries:
        data = run_llmfit(args, timeout=90)
        for row in (data or {}).get("models") or []:
            if not isinstance(row, dict) or not usable_recommend_row(row):
                continue
            rec = summarize_model(row)
            key = str(rec.get("name") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(rec)
    out.sort(key=lambda r: float(r.get("score") or 0), reverse=True)
    return out[:limit]


def ollama_installed_models() -> list[dict[str, Any]]:
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5.0)
        r.raise_for_status()
        payload = r.json()
    except Exception:  # noqa: BLE001
        return _ollama_list_cli()
    rows = []
    for m in payload.get("models") or []:
        size = float(m.get("size") or 0)
        rows.append(
            {
                "name": m.get("name") or m.get("model"),
                "size_gb": round(size / (1024**3), 2) if size else 0.0,
                "digest": (m.get("digest") or "")[:12],
            }
        )
    return [r for r in rows if r.get("name")]


def _ollama_list_cli() -> list[dict[str, Any]]:
    exe = which_tool("ollama")
    if not exe:
        return []
    try:
        proc = subprocess.run([exe, "list"], capture_output=True, text=True, timeout=15, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    rows: list[dict[str, Any]] = []
    for line in (proc.stdout or "").splitlines()[1:]:
        parts = line.split()
        if not parts:
            continue
        size_gb = 0.0
        m = re.search(r"([\d.]+)\s*(GB|MB|TB)", line, re.I)
        if m:
            n = float(m.group(1))
            unit = m.group(2).upper()
            size_gb = n / 1024 if unit == "MB" else n * 1024 if unit == "TB" else n
        rows.append({"name": parts[0], "size_gb": round(size_gb, 2), "digest": ""})
    return rows


def ollama_pull(name: str) -> dict[str, Any]:
    exe = which_tool("ollama")
    if not exe:
        return {"ok": False, "reason": "ollama_missing"}
    try:
        proc = subprocess.run(
            [exe, "pull", name],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "timeout"}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-2000:],
        "stderr": (proc.stderr or "")[-2000:],
        "name": name,
    }


_CACHE: dict[str, Any] = {"t": 0.0, "data": None}


def hardware_fit_catalog() -> dict[str, Any]:
    now = time.time()
    if _CACHE["data"] and now - float(_CACHE["t"]) < 45:
        return _CACHE["data"]
    merged = merge_catalog(live=True)
    ram = available_ram_gb(merged.get("system"))
    installed: list[dict[str, Any]] = []
    suggested: list[dict[str, Any]] = []
    recommended: list[dict[str, Any]] = []
    for row in merged.get("rows") or []:
        item = {
            "name": row.get("ollama_name") or row.get("name"),
            "provider": "ollama" if row.get("ollama_name") else row.get("source"),
            "installed": row.get("state") == "INSTALLED",
            "size_gb": row.get("size"),
            "fit_level": str(row.get("FITS") or "unknown").lower(),
            "notes": row.get("reason"),
            "source": row.get("source"),
            "ollama_name": row.get("ollama_name"),
            "can_pin_ollama": bool(row.get("ollama_name")) and row.get("state") in {"INSTALLED", "PULLABLE"},
            "pin_name": row.get("ollama_name") or row.get("name"),
            "state": row.get("state"),
        }
        if row.get("state") == "INSTALLED" and row.get("source") == "ollama":
            installed.append(item)
        elif row.get("state") == "PULLABLE":
            suggested.append(item)
        if row.get("source") == "llmfit":
            recommended.append(item)
    data = {
        "system": merged.get("system"),
        "available_ram_gb": ram,
        "installed": installed,
        "suggested": suggested,
        "recommended": recommended,
    }
    _CACHE["t"] = now
    _CACHE["data"] = data
    return data
