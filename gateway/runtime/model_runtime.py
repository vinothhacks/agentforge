"""Single local-model runtime interface. Ollama is the only impl until M3."""

from __future__ import annotations

from typing import Any, Protocol


class ModelRuntime(Protocol):
    def list(self) -> list[dict[str, Any]]: ...

    def ensure_available(self, name: str) -> dict[str, Any]: ...

    def chat(self, **kwargs: Any) -> dict[str, Any]: ...

    def unload(self, name: str) -> dict[str, Any]: ...
