"""Template-first compiler: skip the interview, emit AgentSpec from document-query + card."""

from __future__ import annotations

from pathlib import Path

from gateway.spec import AgentSpec, Budgets, ModelPin, dump_spec, load_spec
from gateway.stats import max_steps_from_lo95, max_tools_from_lo95

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


def compile_workspace(workspace: Path, card: dict | None = None) -> AgentSpec:
    src = TEMPLATES / "document-query" / "agentspec.yaml"
    spec = load_spec(src) if src.exists() else AgentSpec()
    spec.workspace_root = str(workspace.resolve())
    spec.memory.long_term = False
    spec.tools = ["rag_search", "fs_list", "fs_read"]
    spec.planner = {"enabled": False}
    if card:
        pin = card.get("model_pin") or {}
        spec.model_pin = ModelPin.model_validate(pin)
        m = card.get("measured") or {}
        lo = float(m.get("per_step_success_lo95") or 0.0)
        spec.budgets = Budgets(
            max_steps=max_steps_from_lo95(lo),
            max_tools=max(3, int(m.get("max_tools") or max_tools_from_lo95(lo))),
            max_tokens=32000,
            max_wall_s=90,
            max_usd=0.50,
        )
        spec.card_ref = pin.get("digest")
    return spec


def persist_spec(spec: AgentSpec, dest: Path) -> None:
    dump_spec(spec, dest)
