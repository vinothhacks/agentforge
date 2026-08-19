from pathlib import Path

from scripts.build_advanced_corpus import write_advanced_docs

from gateway.evals import enumeration_recall
from gateway.packs.rag import HybridIndex, extract_pdf, ingest_workspace


def test_advanced_enumeration_and_precision(tmp_path: Path):
    dem = write_advanced_docs(tmp_path, n=24, with_pdf=True)
    ingest_workspace(tmp_path)
    idx = HybridIndex(tmp_path)
    result = idx.search("List every file mentioning demurrage", limit=12)
    rec = enumeration_recall(idx, "demurrage", result["paths"])
    assert rec["ground_truth"] >= len(dem)
    assert rec["recall"] >= 0.90
    assert rec["precision"] >= 0.90


def test_pdf_text_layer_extracts_amounts(tmp_path: Path):
    write_advanced_docs(tmp_path, n=3, with_pdf=True)
    pdfs = list(tmp_path.glob("*.pdf"))
    assert pdfs
    pages, has_text = extract_pdf(pdfs[0])
    assert has_text
    blob = "\n".join(p["text"] for p in pages)
    assert "Pilotage" in blob or "pilotage" in blob.lower()
    assert "Account total" in blob or "estimate" in blob.lower()
