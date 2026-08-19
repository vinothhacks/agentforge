"""Read llmfit catalog flags. REST has no use_case=tool_use — we filter client-side."""

from __future__ import annotations

import json
import subprocess
from typing import Any


def catalog_flag(model_name: str, llmfit_json: dict[str, Any] | None = None) -> str:
    """Return yes / no / unknown."""
    data = llmfit_json if llmfit_json is not None else load_llmfit_fit()
    if data is None:
        return "unknown"
    needle = model_name.lower()
    for row in data.get("models", []):
        names = [
            str(row.get("name") or ""),
            str(row.get("ollama_name") or ""),
        ]
        if any(needle in n.lower() or n.lower() in needle for n in names if n):
            ids = row.get("capability_ids") or []
            if "tool_use" in ids or "Tool Use" in (row.get("capabilities") or []):
                return "yes"
            return "no"
    return "unknown"


def load_llmfit_fit(limit: int = 400) -> dict[str, Any] | None:
    try:
        proc = subprocess.run(
            ["llmfit", "recommend", "--json", "--use-case", "general", "--limit", str(limit)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def system_specs() -> dict[str, Any] | None:
    try:
        proc = subprocess.run(
            ["llmfit", "--json", "system"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            # some builds: llmfit system --json
            proc = subprocess.run(
                ["llmfit", "system", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def nominate(limit: int = 8) -> list[dict[str, Any]]:
    data = load_llmfit_fit(limit=80) or {"models": []}
    out = []
    for row in data.get("models", []):
        flag = "yes" if "tool_use" in (row.get("capability_ids") or []) else "no"
        out.append(
            {
                "name": row.get("name"),
                "ollama_name": row.get("ollama_name"),
                "catalog": flag,
                "fit_level": row.get("fit_level"),
                "estimated_tps": row.get("estimated_tps"),
                "best_quant": row.get("best_quant"),
                "usable_context": row.get("usable_context"),
            }
        )
        if len(out) >= limit:
            break
    return out
