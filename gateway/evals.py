"""Golden evals for the PDA-folder job. Enumeration recall vs grep ground truth."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from gateway.packs.rag import HybridIndex

FAILURES = (
    "bad_tool",
    "bad_args",
    "schema_reject",
    "budget_stop",
    "injection_ignored",
    "context_overflow",
    "wrong_file",
    "timeout",
    "incomplete_enumeration",
)


def enumeration_recall(index: HybridIndex, term: str, returned_paths: list[str]) -> dict[str, Any]:
    truth = set(index.grep_paths(term))
    got = {p.replace("\\", "/") for p in returned_paths}
    if not truth:
        return {
            "term": term,
            "ground_truth": 0,
            "returned": len(got),
            "hits": 0,
            "recall": 1.0,
            "cause": None,
            "missing": [],
        }
    hits = len(truth & got)
    recall = hits / len(truth)
    precision = hits / len(got) if got else 1.0
    cause = None if recall >= 0.90 else "incomplete_enumeration"
    return {
        "term": term,
        "ground_truth": len(truth),
        "returned": len(got),
        "hits": hits,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "cause": cause,
        "missing": sorted(truth - got),
    }


def extract_paths_from_text(text: str) -> list[str]:
    return re.findall(r"[\w./\\-]+\.(?:pdf|txt|md)", text, flags=re.I)


def score_run(
    *,
    index: HybridIndex,
    question: str,
    answer: str,
    tool_paths: list[str],
    usage: dict[str, Any],
    lo95: float,
) -> dict[str, Any]:
    paths = list(dict.fromkeys(tool_paths + extract_paths_from_text(answer)))
    enum = None
    m = re.search(r"mention(?:ing)?\s+(\w+)", question, flags=re.I)
    if m or "list" in question.lower():
        term = m.group(1) if m else "demurrage"
        enum = enumeration_recall(index, term, paths)
    return {
        "question": question,
        "enumeration": enum,
        "per_step_success_lo95": lo95,
        "tokens": usage.get("turn_tokens"),
        "usd": usage.get("usd"),
        "stop": usage.get("stop"),
        "pass": (enum is None) or (enum["cause"] is None),
    }


def write_sample_pdas(folder: Path, n: int = 12) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    vessels = ["NEPTUNE", "ORION", "VESTA", "ATLAS", "HERA", "JUNO"]
    ports = ["SGSIN", "NLRTM", "DEHAM", "USNYC", "CNSHA", "AEJEA"]
    for i in range(n):
        dem = "Demurrage applicable at USD 12,000/day after 24h." if i % 3 == 0 else "No detention charges."
        body = (
            f"PORT DISBURSEMENT ACCOUNT\n"
            f"Vessel: {vessels[i % len(vessels)]}\n"
            f"Port: {ports[i % len(ports)]}\n"
            f"PDA amount: USD {15000 + i * 1370}\n"
            f"Berth window: 2026-09-0{(i % 8) + 1} 06:00 LT\n"
            f"{dem}\n"
            f"Pilotage: USD 2100\n"
            f"Agency fee: USD 950\n"
        )
        (folder / f"PDA-{i+1:03d}-{vessels[i % len(vessels)]}.txt").write_text(body, encoding="utf-8")


GOLDEN = [
    "What is the PDA amount / berth window in file PDA-001-NEPTUNE.txt?",
    "List PDAs that mention demurrage.",
    "Cite the page for vessel Y.",
    "List every PDA mentioning demurrage",
]


def run_offline_golden(workspace: Path) -> dict[str, Any]:
    """Retrieval-side golden: hybrid index vs grep. No LLM required.

    Always re-ingest so fixture text changes (e.g. demurrage negatives) are scored.
    """
    from gateway.packs.rag import ingest_workspace

    ingest_workspace(workspace)
    idx = HybridIndex(workspace)
    question = "List every PDA mentioning demurrage"
    result = idx.search(question, limit=40)
    scored = score_run(
        index=idx,
        question=question,
        answer="",
        tool_paths=result["paths"],
        usage={"turn_tokens": 0, "usd": 0.0, "stop": None},
        lo95=0.0,
    )
    scored["questions"] = GOLDEN
    scored["taxonomy"] = list(FAILURES)
    return scored
