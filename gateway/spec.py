"""AgentSpec: compiled source of truth. Budgets come from the probe card, not constants."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ModelPin(BaseModel):
    provider: Literal["openrouter", "ollama"] = "openrouter"
    name: str = "openai/gpt-4o-mini"
    digest: str | None = None
    quant: str | None = None
    template_hash: str | None = None
    num_ctx: int = 8192


class Persona(BaseModel):
    name: str = "PDA desk"
    language: str = "en"


class MemorySpec(BaseModel):
    short_term: bool = True
    long_term: bool = False


class RagSpec(BaseModel):
    enabled: bool = True
    sources: list[str] = Field(default_factory=lambda: ["."])
    retrieval: Literal["hybrid"] = "hybrid"


class Permissions(BaseModel):
    fs_write: Literal["deny"] = "deny"
    shell: Literal["deny"] = "deny"
    send_email: Literal["deny"] = "deny"


class Budgets(BaseModel):
    max_steps: int = 3
    max_tools: int = 3
    max_tokens: int = 32000
    max_wall_s: float = 90.0
    max_usd: float = 0.50


class AgentSpec(BaseModel):
    id: str = "pda-query"
    template: str = "pda-query"
    model_pin: ModelPin = Field(default_factory=ModelPin)
    card_ref: str | None = None
    persona: Persona = Field(default_factory=Persona)
    memory: MemorySpec = Field(default_factory=MemorySpec)
    workspace_root: str = "."
    rag: RagSpec = Field(default_factory=RagSpec)
    tools: list[str] = Field(default_factory=lambda: ["rag_search", "fs_list", "fs_read"])
    permissions: Permissions = Field(default_factory=Permissions)
    budgets: Budgets = Field(default_factory=Budgets)
    planner: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})

    def workspace_path(self) -> Path:
        return Path(self.workspace_root).expanduser().resolve()


def load_spec(path: Path) -> AgentSpec:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AgentSpec.model_validate(data)


def dump_spec(spec: AgentSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec.model_dump(), sort_keys=False), encoding="utf-8")
