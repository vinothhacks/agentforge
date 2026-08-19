# AgentForge

OpenRouter-first tool-using agent. v1 tools: `rag_search` (hybrid FTS5 + vectors), `fs_list`, `fs_read`.

PyPI already has an unrelated `agentforge` with **no CLI**. Do not run `uvx agentforge` — that is the other package.

Run **this** repo from GitHub:

```bash
uvx --from git+https://github.com/vinothhacks/agentforge.git agentforge --dir "C:\Users\sm2063\Documents\Vinoth_N_Package_v1"
```

Paste an OpenRouter key in the UI, ask questions, get citations. Ingest extracts the PDF text layer once and builds both indexes.

Re-extract / re-index:

```bash
uvx --from git+https://github.com/vinothhacks/agentforge.git agentforge --ingest --dir path\to\pdfs
```

From a clone:

```bash
git clone https://github.com/vinothhacks/agentforge.git
cd agentforge
uv run agentforge --dir path\to\pdfs
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
