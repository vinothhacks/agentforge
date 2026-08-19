# AgentForge

OpenRouter-first tool-using agent. `uvx` / `pip`. v1 tools: `rag_search` (hybrid FTS5 + vectors), `fs_list`, `fs_read`.

Point it at a **folder of PDFs**. Ingest extracts the text layer once and builds both indexes.

```bash
uvx agentforge --dir path/to/pdfs
# paste an OpenRouter key, ask questions, get citations
```

Re-extract / re-index:

```bash
agentforge --ingest --dir path/to/pdfs
```

## PDF extract benchmark

Not a synthetic PDA sample set. Benchmark is **pypdf extract** on four downloaded public PDFs (40 pages).

| PDFs | Pages | Text layer | Chars | Fail |
| --- | --- | --- | --- | --- |
| 4 | 40 | **3 / 4 (75%)** | 113,239 | 1 scanned FONASBA PDF (OCR out of v1) |

Full table and previews: [BENCHMARK.md](BENCHMARK.md). Reproduce: `uv run python scripts/benchmark_pdf_extract.py`.

## Probe (optional)

```bash
agentforge --experiment
```

Export to Excel comes next. Mail, MCP, web search, long-term memory, and a shell are out of v1.

Python 3.11+. Optional local path: [Ollama](https://ollama.com/download) and `uvx llmfit`.
