"""P-1 disagreement experiment: 25 tags × suite × catalog flag, n=5, Wilson CI.

D = P(probe_fail | catalog_yes) among tags with n=5 complete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gateway.probe.catalog import catalog_flag, load_llmfit_fit
from gateway.probe.runner import FULL_BUDGET_S, run_probe


# Mix of: installed-ish Ollama, catalog tool_use yes/no, unknown, OpenRouter, a known
# template-mismatch style tag (abliterated / custom template).
P1_TAGS: list[dict[str, str]] = [
    {"provider": "ollama", "name": "qwen3:4b"},
    {"provider": "ollama", "name": "granite3.1-moe:latest"},
    {"provider": "ollama", "name": "parable/granite4.1-fable:3b"},
    {"provider": "ollama", "name": "richardyoung/qwythos-9b-abliterated:IQ3_M"},
    {"provider": "ollama", "name": "llama3.1:8b"},
    {"provider": "ollama", "name": "llama3.2:3b"},
    {"provider": "ollama", "name": "mistral:7b"},
    {"provider": "ollama", "name": "qwen2.5:7b"},
    {"provider": "ollama", "name": "qwen2.5-coder:7b"},
    {"provider": "ollama", "name": "phi3:mini"},
    {"provider": "ollama", "name": "gemma2:9b"},
    {"provider": "ollama", "name": "gemma3:4b"},
    {"provider": "ollama", "name": "command-r:35b"},
    {"provider": "ollama", "name": "deepseek-r1:7b"},
    {"provider": "ollama", "name": "nomic-embed-text"},
    {"provider": "ollama", "name": "tinyllama"},
    {"provider": "ollama", "name": "codellama:7b"},
    {"provider": "ollama", "name": "vicuna:7b"},
    {"provider": "openrouter", "name": "openai/gpt-4o-mini"},
    {"provider": "openrouter", "name": "anthropic/claude-3.5-sonnet"},
    {"provider": "openrouter", "name": "google/gemini-2.0-flash-001"},
    {"provider": "openrouter", "name": "meta-llama/llama-3.1-8b-instruct"},
    {"provider": "openrouter", "name": "qwen/qwen-2.5-7b-instruct"},
    {"provider": "ollama", "name": "not-a-real-model-xyz"},
    {"provider": "ollama", "name": "gemma4:cloud"},
]


def _installed_ollama() -> set[str]:
    try:
        import subprocess

        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=20)
        names = set()
        for line in proc.stdout.splitlines()[1:]:
            parts = line.split()
            if parts:
                names.add(parts[0])
        return names
    except Exception:  # noqa: BLE001
        return set()


def run_experiment(
    *,
    n: int = 5,
    mode: str = "fast",
    api_key: str | None = None,
    only_installed: bool = True,
    complete_fn=None,
    out_path: Path | None = None,
    llmfit_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = llmfit_json if llmfit_json is not None else load_llmfit_fit(limit=400)
    installed = _installed_ollama() if complete_fn is None else set()
    rows: list[dict[str, Any]] = []

    for tag in P1_TAGS:
        flag = catalog_flag(tag["name"], data)
        skip = False
        skip_reason = None
        if complete_fn is None and tag["provider"] == "ollama" and only_installed:
            names = {tag["name"], tag["name"].replace(":latest", "")}
            if not (names & installed) and tag["name"] not in installed:
                skip = True
                skip_reason = "not_installed"
        if tag["provider"] == "openrouter" and not api_key and complete_fn is None:
            skip = True
            skip_reason = "no_openrouter_key"
        if skip:
            rows.append(
                {
                    "tag": tag["name"],
                    "provider": tag["provider"],
                    "catalog": flag,
                    "pass_n": None,
                    "n": 0,
                    "lo95": None,
                    "hi95": None,
                    "verdict": skip_reason,
                    "cause": skip_reason,
                }
            )
            print(f"P-1 skip {tag['provider']}:{tag['name']} ({skip_reason})", flush=True)
            continue
        print(f"P-1 probing {tag['provider']}:{tag['name']} catalog={flag} ...", flush=True)
        try:
            kwargs: dict[str, Any] = dict(
                provider=tag["provider"],
                name=tag["name"],
                mode=mode,
                n=n,
                api_key=api_key,
                budget_s=FULL_BUDGET_S,
            )
            if complete_fn is not None:
                kwargs["complete_fn"] = complete_fn
            card = run_probe(**kwargs)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "tag": tag["name"],
                    "provider": tag["provider"],
                    "catalog": flag,
                    "pass_n": "0/0",
                    "n": 0,
                    "lo95": None,
                    "hi95": None,
                    "verdict": "error",
                    "cause": str(exc)[:200],
                }
            )
            continue
        m = card["measured"]
        rows.append(
            {
                "tag": tag["name"],
                "provider": tag["provider"],
                "catalog": flag,
                "pass_n": f"{m['successes']}/{m['n']}",
                "n": m["n"],
                "lo95": m["per_step_success_lo95"],
                "hi95": m["per_step_success_hi95"],
                "verdict": card["verdict"],
                "cause": card.get("cause"),
                "max_steps": m["max_steps"],
                "max_tools": m["max_tools"],
            }
        )

    complete_yes = [r for r in rows if r["catalog"] == "yes" and r["n"] == n]
    if complete_yes:
        fails = sum(1 for r in complete_yes if r["verdict"] != "agent")
        d = fails / len(complete_yes)
    else:
        d = None

    if d is None:
        decision = "insufficient_n — no catalog_yes tag completed n=5. Do not treat this as D<0.10."
        gate = "continue_p0_with_caveat"
    elif d < 0.10:
        decision = "D < 0.10 — catalog is good enough; AgentForge is a feature. Plan says stop or shrink."
        gate = "feature"
    elif d >= 0.25:
        decision = "D >= 0.25 — catalog lies often; the probe is the product. Continue P0."
        gate = "product"
    else:
        decision = "0.10 <= D < 0.25 — probe is a gate, not the headline. Continue P0; do not market certification."
        gate = "gate"

    result = {
        "D": d,
        "decision": decision,
        "gate": gate,
        "n_repeats": n,
        "mode": mode,
        "catalog_yes_complete": len(complete_yes),
        "rows": rows,
    }
    if out_path:
        out_path.write_text(_to_markdown(result), encoding="utf-8")
        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _to_markdown(result: dict[str, Any]) -> str:
    probed = [r for r in result["rows"] if (r.get("n") or 0) > 0]
    agents = [r["tag"] for r in probed if r.get("verdict") == "agent"]
    chat = [r["tag"] for r in probed if r.get("verdict") != "agent"]
    lines = [
        "# P-1 disagreement experiment",
        "",
        f"**D = P(probe_fail | catalog_yes)** = `{result['D']}`",
        "",
        f"**Decision:** {result['decision']}",
        "",
        f"Repeats n={result['n_repeats']}, mode={result['mode']}, "
        f"catalog_yes with n complete: {result['catalog_yes_complete']}. "
        f"gate=`{result['gate']}`.",
        "",
        "Catalog flags are read client-side from llmfit `capability_ids` "
        "(REST has no `use_case=tool_use`). Uninstalled tags = `not_installed`. "
        "OpenRouter without a key = `no_openrouter_key`.",
        "",
        "## Findings",
        "",
        f"- Live-probed tags: {len(probed)}. Passed as `agent`: {', '.join(agents) or '(none)'}.",
        f"- Live-probed `chat_only`: {', '.join(chat) or '(none)'}.",
        "- `D` is undefined until at least one **catalog_yes** tag completes n=5. "
        "That is not D<0.10. llmfit did not flag the installed Ollama tags as "
        "`tool_use`, which is the catalog hole this experiment exists to measure.",
        "",
        "| tag | provider | catalog | pass/n | lo95 | verdict | cause |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in result["rows"]:
        lines.append(
            f"| {r['tag']} | {r['provider']} | {r['catalog']} | {r['pass_n']} | "
            f"{r['lo95']} | {r['verdict']} | {r['cause']} |"
        )
    lines += [
        "",
        "## Gate rule (from the plan)",
        "",
        "- `D < 0.10` — feature; stop or shrink to the PDA template.",
        "- `D >= 0.25` — catalog lies often; the probe is the product. Continue P0.",
        "- `0.10 <= D < 0.25` — probe is a gate, not the headline. Continue P0; do not market certification.",
        "",
        "User instruction for this implementation pass: complete P0–P3 regardless. "
        "This file is the interview artifact. Do not market certification unless gate=product.",
        "",
        "## Design (25 tags)",
        "",
        "Mix of installed Ollama instruct tags, catalog tool_use yes/no, unknown, "
        "OpenRouter allowlist ids, one abliterated/custom-template tag, one embedding "
        "tag, one fake name, one `:cloud` tag.",
        "",
    ]
    return "\n".join(lines)
