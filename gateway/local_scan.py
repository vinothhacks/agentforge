"""Optional local scan via llmfit. Never blocks the OpenRouter 2-minute path."""

from __future__ import annotations

import subprocess
from typing import Any

from gateway.bins import which_tool
from gateway.llmfit_bridge import hardware_fit_catalog
from gateway.probe.ladder import fail_forward

LLMFIT_PYPI = "https://pypi.org/project/llmfit/"
OLLAMA_DOWNLOAD = "https://ollama.com/download"


def ollama_installed() -> bool:
    return which_tool("ollama") is not None


def llmfit_installed() -> bool:
    return which_tool("llmfit") is not None


def scan(*, quick: bool = False) -> dict[str, Any]:
    llm = llmfit_installed()
    ollama = ollama_installed()
    info: dict[str, Any] = {
        "llmfit": llm,
        "ollama": ollama,
        "llmfit_pypi": LLMFIT_PYPI,
        "llmfit_install": "uv tool install llmfit",
        "ollama_install": OLLAMA_DOWNLOAD,
        "llmfit_hint": None if llm else "uv tool install llmfit",
        "system": None,
        "available_ram_gb": 0,
        "installed": [],
        "suggested": [],
        "recommended": [],
        "nominees": [],
    }
    if quick:
        return info
    catalog = hardware_fit_catalog()
    info.update(catalog)
    info["nominees"] = [
        n
        for n in (catalog.get("installed") or []) + (catalog.get("suggested") or [])
        if n.get("fit_level") not in {"wont_fit"}
    ]
    return info


def install_llmfit() -> dict[str, Any]:
    if llmfit_installed():
        return {"ok": True, "llmfit": True, "already": True, "llmfit_pypi": LLMFIT_PYPI}
    uv = which_tool("uv")
    if not uv:
        return {"ok": False, "reason": "uv_missing", "hint": "install uv from https://docs.astral.sh/uv/"}
    try:
        proc = subprocess.run(
            [uv, "tool", "install", "llmfit"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "timeout"}
    ok = proc.returncode == 0 or llmfit_installed()
    return {
        "ok": ok,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-2000:],
        "stderr": (proc.stderr or "")[-2000:],
        "llmfit": llmfit_installed(),
        "llmfit_pypi": LLMFIT_PYPI,
    }


def try_local(api_key: str | None = None, name: str | None = None) -> dict[str, Any]:
    info = scan(quick=False)
    if not info["ollama"]:
        return {"ok": False, "reason": "ollama_missing", "scan": info, "english": "Download Ollama first."}
    installed = info.get("installed") or []
    suggested = [s for s in (info.get("suggested") or []) if s.get("fit_level") in {"perfect", "good", "marginal"}]
    pick = name
    if not pick and installed:
        ranked = sorted(
            installed,
            key=lambda r: {"perfect": 0, "good": 1, "marginal": 2, "tight": 3}.get(str(r.get("fit_level")), 9),
        )
        pick = ranked[0]["name"]
    if not pick and suggested:
        pick = suggested[0]["name"]
    if not pick:
        return {
            "ok": False,
            "reason": "no_local_model",
            "scan": info,
            "english": "No local model that fits. Pick one in the sidebar and pull it with Ollama.",
        }
    alts = [("ollama", m["name"]) for m in installed if m.get("name") != pick][:3]
    alts.append(("openrouter", "openai/gpt-4o-mini"))
    result = fail_forward(
        provider="ollama",
        name=pick,
        alternate_tags=alts,
        api_key=api_key,
        mode="fast",
        n=3,
    )
    result["scan"] = info
    result["picked"] = pick
    if result.get("ok"):
        result["english"] = f"Pinned local model {pick}."
    return result
