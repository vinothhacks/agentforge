"""Optional local scan via llmfit. Never blocks the OpenRouter 2-minute path."""

from __future__ import annotations

import subprocess
from typing import Any

from gateway.probe.catalog import nominate, system_specs
from gateway.probe.ladder import fail_forward


def ollama_installed() -> bool:
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, timeout=10, check=False)
        return True
    except FileNotFoundError:
        return False


def llmfit_installed() -> bool:
    try:
        subprocess.run(["llmfit", "--version"], capture_output=True, timeout=10, check=False)
        return True
    except FileNotFoundError:
        return False


def scan() -> dict[str, Any]:
    return {
        "llmfit": llmfit_installed(),
        "ollama": ollama_installed(),
        "system": system_specs(),
        "nominees": nominate(limit=8) if llmfit_installed() else [],
        "ollama_install": "https://ollama.com/download" if not ollama_installed() else None,
        "llmfit_hint": "uvx llmfit" if not llmfit_installed() else None,
    }


def try_local(api_key: str | None = None) -> dict[str, Any]:
    info = scan()
    if not info["ollama"]:
        return {"ok": False, "reason": "ollama_missing", "scan": info}
    nominees = [( "ollama", n["ollama_name"] or n["name"]) for n in info["nominees"] if n.get("ollama_name") or n.get("name")]
    if not nominees:
        nominees = [("ollama", "qwen3:4b")]
    first = nominees[0]
    alts = nominees[1:4] + [("openrouter", "openai/gpt-4o-mini")]
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
