"""Optional local scan via llmfit. Never blocks the OpenRouter 2-minute path."""

from __future__ import annotations

import subprocess
from typing import Any

from gateway.bins import which_tool
from gateway.probe.catalog import nominate, system_specs

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
        "nominees": [],
    }
    if quick:
        return info
    info["system"] = system_specs()
    info["nominees"] = nominate(limit=8) if llm else []
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


def try_local(api_key: str | None = None) -> dict[str, Any]:
    info = scan()
    if not info["ollama"]:
        return {"ok": False, "reason": "ollama_missing", "scan": info, "english": "Download Ollama first."}
    if not info["llmfit"]:
        return {
            "ok": False,
            "reason": "llmfit_missing",
            "scan": info,
            "english": "Install llmfit (sidebar button or: uv tool install llmfit).",
        }
    nominees = [
        ("ollama", n["ollama_name"] or n["name"])
        for n in info["nominees"]
        if n.get("ollama_name") or n.get("name")
    ]
    if not nominees:
        nominees = [("ollama", "qwen3:4b")]
    first = nominees[0]
    alts = nominees[1:4] + [("openrouter", "openai/gpt-4o-mini")]
    from gateway.probe.ladder import fail_forward

    result = fail_forward(
        provider=first[0],
        name=first[1],
        alternate_tags=alts,
        api_key=api_key,
        mode="fast",
        n=3,
    )
    result["scan"] = info
    return result
