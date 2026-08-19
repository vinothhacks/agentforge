# PDF extract benchmark

Suite: **pdf_extract_only**. Downloaded public PDFs in `fixtures/advanced/source/`. No synthetic PDA `.txt` files.

Reproduce:

```bash
uv run python scripts/benchmark_pdf_extract.py
```

## Score

| Metric | Value |
| --- | --- |
| PDFs | 4 |
| Pages | 40 |
| Text-layer success | **3 / 4 (75%)** |
| Characters extracted | 113,239 |
| Failures | 1 image-only PDF (OCR out of v1) |

## Per file

| File | Bytes | Pages | Text layer | Chars | Chars/page | Extract s | Preview |
| --- | --- | --- | --- | --- | --- | --- | --- |
| esteve-paldiski-brochure.pdf | 2,562,270 | 7 | yes | 4,990 | 713 | 0.391 | Paldiski South Harbour |
| fonasba-standard-liner-agency-agreement-1993.pdf | 338,523 | 5 | **no** | 0 | 0 | 0.005 | *(empty — scanned)* |
| itic-intermediary-2006-proforma-da.pdf | 1,944,182 | 12 | yes | 70,995 | 5,916 | 0.288 | ITIC professional insurer… |
| uscourts-pda-da-desk-opinion.pdf | 692,028 | 16 | yes | 37,254 | 2,328 | 0.200 | Case 4:11-cv-00938 Southern District of Texas… |

Extractor: `pypdf` layout mode with hyphen-join and table spacing cleanup (`gateway.packs.rag.extract_pdf`). Image-only pages are counted as fail and listed in ingest warnings. OCR is not in v1.

Sources: `fixtures/advanced/source/SOURCES.txt`.
