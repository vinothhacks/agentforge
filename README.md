# AgentForge

OpenRouter-first tool-using agent. v1 tools: `rag_search` (hybrid FTS5 + vectors), `fs_list`, `fs_read`. Indexes PDF / txt / md (not `.docx`).

The UI lists **llmfit** and **Ollama** on the left, with install / download links. Workspace files and the chat can be downloaded from the same sidebar.

## Run from Downloads (or any folder)

`uv run agentforge` looks for a **local** `pyproject.toml`. There is none in Downloads, so you get `Failed to spawn: agentforge`.

PyPI also has an **unrelated** `agentforge` with no CLI. `uvx agentforge` installs that other package.

**Install once, then run from anywhere:**

```powershell
uv tool install git+https://github.com/vinothhacks/agentforge.git
agentforge --dir "C:\Users\sm2063\Documents\Vinoth_N_Package_v1"
```

Upgrade after a git push:

```powershell
uv tool install --force git+https://github.com/vinothhacks/agentforge.git
```

**One-shot (no install):**

```powershell
uvx --from git+https://github.com/vinothhacks/agentforge.git agentforge --dir "C:\Users\sm2063\Documents\Vinoth_N_Package_v1"
```

Same thing via `uv run` (note `--no-project --with`):

```powershell
uv run --no-project --with git+https://github.com/vinothhacks/agentforge.git agentforge --dir "C:\Users\sm2063\Documents\Vinoth_N_Package_v1"
```

Paste an OpenRouter key in the UI and ask about the folder. First launch ingests PDF text and builds both indexes.

Re-extract / re-index:

```powershell
agentforge --ingest --dir "C:\Users\sm2063\Documents\Vinoth_N_Package_v1"
```

Local models:

- [Download Ollama](https://ollama.com/download)
- [llmfit on PyPI](https://pypi.org/project/llmfit/) or `uv tool install llmfit` (also a button in the UI)

## From a clone

`uv run agentforge` works **only** after `cd` into this repo:

```powershell
git clone https://github.com/vinothhacks/agentforge.git
cd agentforge
uv run agentforge --dir "C:\Users\sm2063\Documents\Vinoth_N_Package_v1"
```

From another folder, point uv at the clone:

```powershell
uv run --directory path\to\agentforge agentforge --dir "C:\Users\sm2063\Documents\Vinoth_N_Package_v1"
```

## PDF extract benchmark

Benchmark is **pypdf extract** on four downloaded public PDFs (40 pages). Not a synthetic sample set.

| PDFs | Pages | Text layer | Chars | Fail |
| --- | --- | --- | --- | --- |
| 4 | 40 | **3 / 4 (75%)** | 113,239 | 1 scanned FONASBA PDF (OCR out of v1) |

Full table and previews: [BENCHMARK.md](BENCHMARK.md). Reproduce: `uv run python scripts/benchmark_pdf_extract.py`.

## Probe (optional)

```powershell
agentforge --experiment
```

Mail, MCP, web search, long-term memory, and a shell are out of v1.

Python 3.11+.
