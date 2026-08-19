"""PDF-extract benchmark: downloaded source PDFs only. No synthetic PDA text files."""

from __future__ import annotations

import json
import time
from pathlib import Path

from gateway.packs.rag import extract_pdf

SOURCE = Path(__file__).resolve().parents[1] / "fixtures" / "advanced" / "source"
OUT = Path(__file__).resolve().parents[1] / "BENCHMARK.json"


def main() -> None:
    rows = []
    for pdf in sorted(SOURCE.glob("*.pdf")):
        t0 = time.perf_counter()
        pages, has_text = extract_pdf(pdf)
        elapsed = round(time.perf_counter() - t0, 3)
        texts = [p["text"] for p in pages]
        chars = sum(len(t) for t in texts)
        preview = next((t.replace("\n", " ").strip()[:220] for t in texts if t.strip()), "")
        rows.append(
            {
                "file": pdf.name,
                "bytes": pdf.stat().st_size,
                "pages": len(pages),
                "has_text_layer": has_text,
                "chars": chars,
                "chars_per_page": round(chars / max(len(pages), 1), 1),
                "extract_s": elapsed,
                "preview": preview,
            }
        )
    ok = [r for r in rows if r["has_text_layer"]]
    report = {
        "suite": "pdf_extract_only",
        "n_pdfs": len(rows),
        "text_layer_ok": len(ok),
        "text_layer_fail": len(rows) - len(ok),
        "text_layer_rate": round(len(ok) / len(rows), 4) if rows else 0.0,
        "total_pages": sum(r["pages"] for r in rows),
        "total_chars": sum(r["chars"] for r in rows),
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
