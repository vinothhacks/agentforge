from pathlib import Path

import pytest

from gateway.packs.rag import extract_pdf

SOURCE = Path(__file__).resolve().parents[1] / "fixtures" / "advanced" / "source"


@pytest.mark.skipif(not SOURCE.exists() or not list(SOURCE.glob("*.pdf")), reason="no downloaded PDFs")
def test_pdf_extract_benchmark_source_only():
    pdfs = sorted(SOURCE.glob("*.pdf"))
    assert len(pdfs) >= 3
    ok = 0
    for pdf in pdfs:
        pages, has_text = extract_pdf(pdf)
        assert pages
        if has_text:
            ok += 1
            assert sum(len(p["text"]) for p in pages) > 100
    # Image-only PDFs are allowed; at least half of the suite must have a text layer.
    assert ok / len(pdfs) >= 0.5
