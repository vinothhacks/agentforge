"""LiteLLM adapter. OpenRouter first, Ollama later — same interface."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import litellm


class ProviderError(RuntimeError):
    pass


def litellm_model(provider: str, name: str) -> str:
    if provider == "ollama":
        if name.startswith("ollama/"):
            return name
        return f"ollama/{name}"
    if provider == "openrouter":
        if name.startswith("openrouter/"):
            return name
        return f"openrouter/{name}"
    raise ProviderError(f"unknown provider: {provider}")


def complete(
    *,
    provider: str,
    name: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    num_ctx: int = 8192,
    temperature: float = 0.0,
    api_key: str | None = None,
    timeout: float = 90.0,
) -> dict[str, Any]:
    model = litellm_model(provider, name)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if provider == "openrouter":
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ProviderError("OPENROUTER_API_KEY is not set")
        kwargs["api_key"] = key
    if provider == "ollama":
        kwargs["num_ctx"] = num_ctx
        kwargs["api_base"] = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

    t0 = time.perf_counter()
    # Daemon thread: Ollama/LiteLLM often ignore `timeout`. A non-daemon
    # worker would block process exit after the probe budget fires.
    box: dict[str, Any] = {}

    def _call() -> None:
        try:
            box["resp"] = litellm.completion(**kwargs)
        except Exception as exc:  # noqa: BLE001
            box["exc"] = exc

    worker = threading.Thread(target=_call, daemon=True)
    worker.start()
    worker.join(timeout=timeout)
    if worker.is_alive():
        raise ProviderError(f"timeout after {timeout:.0f}s")
    if "exc" in box:
        raise ProviderError(str(box["exc"])) from box["exc"]
    resp = box["resp"]
    elapsed = time.perf_counter() - t0

    choice = resp.choices[0]
    msg = choice.message
    usage = getattr(resp, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    tool_calls = []
    raw_calls = getattr(msg, "tool_calls", None) or []
    for tc in raw_calls:
        fn = getattr(tc, "function", None)
        tool_calls.append(
            {
                "id": getattr(tc, "id", "") or "",
                "name": getattr(fn, "name", "") if fn else "",
                "arguments": getattr(fn, "arguments", "") if fn else "{}",
            }
        )
    content = msg.content or ""
    return {
        "content": content,
        "tool_calls": tool_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_s": elapsed,
        "finish_reason": choice.finish_reason,
    }


def estimate_usd(provider: str, prompt_tokens: int, completion_tokens: int) -> float:
    if provider != "openrouter":
        return 0.0
    # Conservative default; real rates vary. Meter uses this until we get provider usage.
    return (prompt_tokens * 3e-6) + (completion_tokens * 1.5e-5)
