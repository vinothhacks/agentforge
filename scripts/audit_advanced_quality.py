"""Offline quality audit of fixtures/advanced (downloaded + generated)."""

from __future__ import annotations

import json
from pathlib import Path

from gateway.evals import enumeration_recall
from gateway.packs.rag import HybridIndex, extract_pdf, ingest_workspace

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "advanced"


def main() -> None:
    warning = ingest_workspace(ROOT)
    idx = HybridIndex(ROOT)
    queries = [
        "List every file mentioning demurrage",
        "demurrage",
        "pilotage inwards",
        "MSC AURORA",
        "Paldiski",
        "FONASBA",
        "cash advance",
    ]
    rows = []
    for q in queries:
        result = idx.search(q, limit=20)
        term = result.get("term") or q
        rec = enumeration_recall(idx, term, result["paths"])
        rows.append(
            {
                "query": q,
                "term": term,
                "paths": len(result["paths"]),
                "hits": len(result["hits"]),
                "recall": rec["recall"],
                "precision": rec["precision"],
                "ground_truth": rec["ground_truth"],
                "cause": rec["cause"],
            }
        )

    pdf_quality = []
    for pdf in sorted(ROOT.rglob("*.pdf")):
        if ".agentforge" in pdf.parts:
            continue
        pages, has_text = extract_pdf(pdf)
        chars = sum(len(p["text"]) for p in pages)
        rel = str(pdf.relative_to(ROOT)).replace("\\", "/")
        pdf_quality.append(
            {
                "path": rel,
                "pages": len(pages),
                "has_text": has_text,
                "chars": chars,
            }
        )

    report = {
        "ingest": warning,
        "queries": rows,
        "pdfs": pdf_quality,
        "files_indexed": warning.get("files_scanned"),
        "chunks": warning.get("chunks"),
    }
    dest = ROOT / "QUALITY.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
