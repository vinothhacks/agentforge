"""Read llmfit catalog flags. REST has no use_case=tool_use — we filter client-side."""

from __future__ import annotations

from typing import Any

from gateway.llmfit_bridge import recommend_chat, run_llmfit, system_specs as hardware_system


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
    models = recommend_chat(limit=min(limit, 24))
    if models:
        return {"models": models}
    return run_llmfit(["recommend", "-n", str(min(limit, 80)), "--use-case", "general"])


def system_specs() -> dict[str, Any] | None:
    return hardware_system()


def nominate(limit: int = 8) -> list[dict[str, Any]]:
    out = []
    for row in recommend_chat(limit=max(limit, 8)):
        flag = "yes" if row.get("tool_use") else "no"
        out.append(
            {
                "name": row.get("name"),
                "ollama_name": row.get("ollama_name"),
                "catalog": flag,
                "fit_level": row.get("fit_level"),
                "estimated_tps": row.get("estimated_tps"),
                "best_quant": row.get("best_quant"),
                "usable_context": row.get("effective_context_length") or row.get("usable_context"),
            }
        )
        if len(out) >= limit:
            break
    return out
