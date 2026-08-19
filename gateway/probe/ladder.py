"""Fail-forward ladder. Dead ends only after every rung."""

from __future__ import annotations

from typing import Any, Callable

from gateway.probe.runner import run_probe


CTX_LADDER = [8192, 4096, 2048]
QUANT_LADDER = [None, "Q4_K_M", "Q3_K_M"]


def fail_forward(
    *,
    provider: str,
    name: str,
    alternate_tags: list[tuple[str, str]] | None = None,
    api_key: str | None = None,
    complete_fn: Callable | None = None,
    mode: str = "fast",
    n: int = 3,
    probe_fn: Callable | None = None,
) -> dict[str, Any]:
    """Try lower num_ctx → alternate quant → alternate tag → cloud.

    alternate_tags is a list of (provider, name) including a cloud fallback last.
    """
    probe = probe_fn or run_probe
    attempts: list[dict[str, Any]] = []
    tags: list[tuple[str, str, str | None]] = [(provider, name, None)]
    for q in QUANT_LADDER[1:]:
        tags.append((provider, name, q))
    for p, nme in alternate_tags or []:
        tags.append((p, nme, None))

    seen: set[tuple[str, str, int, str | None]] = set()
    for prov, tag, quant in tags:
        for ctx in CTX_LADDER:
            key = (prov, tag, ctx, quant)
            if key in seen:
                continue
            seen.add(key)
            kwargs: dict[str, Any] = dict(
                provider=prov,
                name=tag,
                num_ctx=ctx,
                mode=mode,
                n=n,
                api_key=api_key,
                quant=quant,
            )
            if complete_fn is not None:
                kwargs["complete_fn"] = complete_fn
            card = probe(**kwargs)
            attempts.append({"provider": prov, "name": tag, "num_ctx": ctx, "quant": quant, "card": card})
            if card["verdict"] == "agent" and card["measured"]["max_tools"] >= 3:
                return {"ok": True, "card": card, "attempts": attempts, "rung": len(attempts)}
    last = attempts[-1]["card"] if attempts else {
        "verdict": "chat_only",
        "cause": "ladder_exhausted",
        "model_pin": {"provider": provider, "name": name, "num_ctx": 8192},
        "measured": {"max_tools": 0, "max_steps": 1, "per_step_success_lo95": 0.0},
    }
    last["cause"] = last.get("cause") or "ladder_exhausted"
    last["verdict"] = "chat_only"
    return {
        "ok": False,
        "card": last,
        "attempts": attempts,
        "rung": len(attempts),
        "english": _english(last),
    }


def _english(card: dict[str, Any]) -> str:
    cause = card.get("cause") or "unknown"
    name = card.get("model_pin", {}).get("name", "the model")
    mapping = {
        "no_tool_call": f"{name} did not call tools on this runtime.",
        "schema_fail": f"{name} called a tool with invalid arguments.",
        "template_mismatch": f"{name} has a chat-template mismatch for tool calling.",
        "probe_timeout": f"{name} did not finish the probe in time.",
        "oom": f"{name} ran out of memory.",
        "ctx_overflow": f"{name} overflowed the context window.",
        "ladder_exhausted": "Every fallback failed. Chat-only on this machine; use OpenRouter.",
        "timeout": f"{name} timed out.",
        "bad_tool": f"{name} called the wrong tool.",
    }
    return mapping.get(cause, f"{name} failed the probe ({cause}).")
