"""Template-first compiler: skip the interview, emit AgentSpec from document-query + card."""

from __future__ import annotations

from pathlib import Path

from gateway.runtime.tools import ALL_TOOLS
from gateway.spec import AgentSpec, Budgets, ModelPin, dump_spec, load_spec
from gateway.stats import effective_lo95, max_steps_from_lo95, max_tools_from_lo95

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


def compile_workspace(workspace: Path, card: dict | None = None) -> AgentSpec:
    src = TEMPLATES / "document-query" / "agentspec.yaml"
    spec = load_spec(src) if src.exists() else AgentSpec()
    spec.workspace_root = str(workspace.resolve())
    spec.memory.long_term = False
    # Keep the full tool list; permissions.fs_write is what gates writing,
    # not the tool roster. Hardcoding the read set here silently dropped the
    # write tools on every CLI start.
    spec.tools = list(ALL_TOOLS)
    spec.planner = {"enabled": False}
    if card:
        pin = card.get("model_pin") or {}
        spec.model_pin = ModelPin.model_validate(pin)
        m = card.get("measured") or {}
        lo = effective_lo95(m.get("per_step_success_lo95"))
        spec.budgets = Budgets(
            max_steps=max_steps_from_lo95(lo),
            max_tools=max(len(ALL_TOOLS), int(m.get("max_tools") or max_tools_from_lo95(lo))),
            max_tokens=32000,
            max_wall_s=90,
            max_usd=0.50,
        )
        spec.card_ref = pin.get("digest")
    return spec


def persist_spec(spec: AgentSpec, dest: Path) -> None:
    dump_spec(spec, dest)
