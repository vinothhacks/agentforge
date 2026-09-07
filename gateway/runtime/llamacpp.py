"""llama.cpp runtime behind a feature flag. Default OFF. Do not compile."""

from __future__ import annotations

import os
from typing import Any

import httpx

from gateway.bins import which_tool

INSTALL_DOCS = "https://github.com/ggml-org/llama.cpp"

LLAMACPP_ENABLED = os.environ.get("AGENTFORGE_LLAMACPP", "").strip() in {"1", "true", "yes"}


def llama_server_path() -> str | None:
    return which_tool("llama-server") or which_tool("llama-server.exe")


def status() -> dict[str, Any]:
    path = llama_server_path()
    return {
        "flag": LLAMACPP_ENABLED,
        "present": bool(path),
        "path": path,
        "docs": INSTALL_DOCS,
        "note": None
        if path
        else "llama-server not on PATH. AgentForge will not compile llama.cpp. See docs.",
    }


class LlamaCppRuntime:
    """OpenAI-compatible llama-server. Only constructed when flag is on and binary exists."""

    def __init__(self, base: str = "http://127.0.0.1:8080"):
        if not LLAMACPP_ENABLED:
            raise RuntimeError("llama.cpp runtime is OFF (AGENTFORGE_LLAMACPP unset)")
        if not llama_server_path():
            raise RuntimeError(status()["note"])
        self.base = base.rstrip("/")

    def list(self) -> list[dict[str, Any]]:
        try:
            r = httpx.get(f"{self.base}/v1/models", timeout=5.0)
            r.raise_for_status()
            return list((r.json() or {}).get("data") or [])
        except Exception:  # noqa: BLE001
            return []

    def ensure_available(self, name: str) -> dict[str, Any]:
        try:
            r = httpx.get(f"{self.base}/v1/models", timeout=5.0)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "llama-server is not reachable. Start it yourself; AgentForge does not compile llama.cpp. "
                f"Docs: {INSTALL_DOCS}"
            ) from exc
        return {"ok": True, "name": name, "runtime": "llama.cpp"}

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        messages = kwargs.get("messages") or []
        tools = kwargs.get("tools")
        payload: dict[str, Any] = {
            "model": kwargs.get("name") or "local",
            "messages": messages,
            "temperature": float(kwargs.get("temperature") or 0),
        }
        if tools:
            payload["tools"] = tools
        timeout = float(kwargs.get("timeout") or 90.0)
        r = httpx.post(f"{self.base}/v1/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            tool_calls.append(
                {
                    "id": tc.get("id") or "",
                    "name": fn.get("name") or "",
                    "arguments": fn.get("arguments") or "{}",
                }
            )
        return {
            "content": msg.get("content") or "",
            "tool_calls": tool_calls,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "elapsed_s": 0.0,
            "finish_reason": choice.get("finish_reason"),
        }

    def unload(self, name: str) -> dict[str, Any]:
        return {"ok": True, "skipped": True}
