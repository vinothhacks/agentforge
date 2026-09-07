# Plan: File read + edit for AgentForge workspace chat

Status: **implemented 2026-08-21**. Phases 0-4 built. Read + write + edit are live
behind `permissions.fs_write`. See "Verification status" for what is proven and
what still needs a run on a machine with Ollama.

## Goal

Let the user ask natural questions about a folder (e.g. resumes) and have the agent:

1. **List** files reliably (`fs_list`)
2. **Read** text / extracted PDF + DOCX content (`fs_read`, `rag_search`)
3. **Edit** existing files and **create** new ones under the workspace, with safety rails

OpenRouter stays optional. Local Ollama / downloaded GGUF models are the default path.

---

## Capability matrix (after this change)

| Capability | State | Notes |
|------------|--------|--------|
| List files | `fs_list` | Sandboxed. Now takes `glob` + `recursive`, returns exact `count` |
| Read text/md | `fs_read` | UTF-8 |
| Read PDF | `fs_read` | Page markers `[page N]`, capped at 20k chars |
| Read DOCX | `fs_read` | stdlib zipfile + ElementTree, no new dependency |
| Search content | `rag_search` | Hybrid index. **DOCX now ingested** (previously never was) |
| Create file | `fs_write` | `create` \| `overwrite`, allowlisted extensions |
| Edit file | `fs_edit` | Exact-once string replace, idempotent re-run |
| Make folder | `fs_mkdir` | Sandboxed |
| Chat UI | `:8789` | Never silent, never raw JSON, sidebar edit toggle + diff modal |

---

## Phase 0 - Stabilize chat — DONE

1. Pin a **Ready** installed model before Q&A (Browse Ollama → Use).
2. UI shows **Thinking…** while `/api/chat` runs; 4-minute abort with a clear message.
3. Download errors (GGUF precheck) are separated from chat errors in the status line.
4. Skill forces `fs_list` first for list questions; exact tool names only.
5. `finalize()` guarantees a non-empty, plain-English reply.

### Phase 0 findings (2026-08-20 run) and what fixed each

Workspace: `Vinoth_N_Package_v1` · Model: `granite3.1-moe:latest` · Score was **7/10**

| # | Was | Cause | Fix |
|---|-----|-------|-----|
| 1 list files | PASS | reply was raw JSON | `finalize()` + `summarize_results()` render prose; `looks_like_json()` detects and replaces |
| 2 list PDFs | FAIL | Ollama `502` | provider retries 3x with backoff; per-call timeout now tracks remaining wall budget instead of a fixed 120s cap |
| 3 list DOCX | PASS | — | **plus** DOCX is now actually ingested and searchable |
| 4 README summary | PASS* | invented tool name | `bad_tool` now returns `valid_tools` so the model self-corrects |
| 5 GenAI experience | PASS* | invented tool name → `bad_tool` | same |
| 6 job titles | PASS | — | — |
| 7 Automation Lead | FAIL | 502 | retries |
| 8 GenAI Engineer | PASS* | invented tool name | same |
| 9 quote skills | FAIL | 502 | retries |
| 10 PDF count | PASS | wrong count `10` | `fs_list(glob="*.pdf")` returns an exact `count`; the skill tells the model to quote it rather than tally |

---

## Phase 1 - Stronger read path — DONE

1. `fs_list` takes `glob` (`*.pdf`, `.pdf`, `pdf`, `*Resume*` all work) and `recursive`; returns `count`, `file_count`, `dir_count`, `truncated`, `paths`. Globs never match directories, so a folder cannot inflate a file count.
2. `fs_read` extracts PDF (page markers) and DOCX (tables kept on one line) via `gateway/packs/extract.py`. Scanned PDFs return `no_text_layer` rather than empty text.
3. `rag_search` returns a `paths` list for enumeration.
4. `GET /api/files` lists the writable text formats too and flags each row `writable`, so a file the agent just created appears without a restart.

---

## Phase 2 - Safe write / edit — DONE

### Tools

| Tool | Args | Behavior |
|------|------|----------|
| `fs_write` | `path`, `content`, `mode=create\|overwrite` | Create or replace whole file |
| `fs_edit` | `path`, `old_text`, `new_text` | Exact replace once; fails on 0 or >1 matches |
| `fs_mkdir` | `path` | Create directory under workspace |

### Permissions

`Permissions.fs_write` is `deny` \| `ask` \| `allow`.

- `deny` — write tools are **not advertised** to the model at all, and the gateway refuses them even if called.
- `ask` — nothing touches disk. The tool stages the change, returns `needs_confirm` + a unified diff + a single-use token. `POST /api/confirm` applies it.
- `allow` — the write lands immediately. **A backup is still taken first.**

The shipped template is `allow`. The sidebar control overrides it at runtime and the choice persists in the store.

### Safety — six gates, in order

1. Permission check.
2. `safe_join` — separators normalised so `..\x` and `../x` mean the same thing on Windows and Linux; `..` components, drive letters, UNC prefixes, NUL bytes and Windows device names (`CON`, `NUL`, `COM1`…) are rejected.
3. Blocked directories: `.agentforge`, `.git`, `.venv`, `node_modules`, `__pycache__`.
4. Extension allowlist: `.md` `.txt` `.json` `.csv` `.yaml` `.yml`. **No PDF/DOCX binary rewrite.**
5. Size cap 512 KB, measured in UTF-8 bytes.
6. Backup to `.agentforge/backups/<path>.<ts>.bak` before any replacement, then an atomic temp-file + `os.replace` write.

Every applied write appends to `.agentforge/write_audit.jsonl`. Refused writes are not logged as writes. `GET /api/audit` reads it back.

### Acceptance — all three verified by `tests/test_write_integration.py`

- "Create `summary.md` listing all resume PDFs" → file exists on disk with correct names.
- "Replace X with Y" → exact edit; the second run is idempotent (`already_applied`), not a corruption.
- "Write `../outside.txt`" → `path_escape`, and **no file is created anywhere**, including no literal `..\outside.txt` on Linux.

---

## Phase 3 - UX — DONE

1. Sidebar **Allow file edits** selector (off / ask / allow) with a live state badge.
2. Diff preview modal for `ask` mode, colourised, with Apply / Discard. Queues multiple staged changes.
3. **Open file** links via `/api/download/file`; editable files carry an `editable` tag.
4. File list refreshes automatically after a successful write or an approved diff.

---

## Phase 4 - Evaluation — DONE

`scripts/tenq_resume_chat.py` runs 10 read + 3 write questions and checks the
**disk**, not just the reply text. `scripts/verify_all.py` (and `verify_all.bat`)
drives the whole thing: pytest → temp copy → ingest → boot → pin model → set
permission → harness → verdict.

    scripts\verify_all.bat "C:\path\to\Vinoth_N_Package_v1"

The workspace is always copied to a temp folder first, so no write can reach the
real resumes. OpenRouter is stripped from the environment; local model only.

---

## Fixed 13-question checklist

Read (must return a non-empty reply within 4 minutes; list questions must call
`fs_list` or cite real tool paths):

1. What resume files are in this folder?
2. List every PDF in this folder.
3. List every DOCX in this folder.
4. What does README_resume_usage_notes.md say?
5. Summarize Vinoth's AI / GenAI experience from the resumes.
6. What job titles appear across the resume files?
7. Which resume is aimed at Automation Lead?
8. Which resume is aimed at GenAI Engineer?
9. Quote one skills or tools line from any resume (cite path).
10. How many resume PDF files are there?

Write (must be confirmed against the filesystem):

11. Create `summary.md` listing every resume PDF → file exists and names match.
12. In `notes.md`, replace DRAFT with FINAL → exact edit, re-run is idempotent.
13. Create `../outside.txt` → refused, nothing created outside the workspace.

---

## Verification status

**Executed and green (88 assertions, Linux sandbox, real fixture PDFs and DOCX):**
path sandbox, extension allowlist, blocked dirs, size cap, edit 0/1/many-match,
idempotency, ask-mode staging and token replay, audit log, backups, `fs_list`
glob counts, PDF + DOCX extraction, DOCX ingest and retrieval, tool gating at
deny/ask/allow, `bad_tool` self-correction, never-silent finalizer, and the full
agent loop for all three Phase 2 acceptance criteria.

**Not executed here — needs a machine with Ollama:** the FastAPI HTTP layer
(`tests/test_write_api.py`), live model tool-calling, the 13-question harness,
and the browser UI. The sandbox has no network, no pytest, and Python 3.10.
Run `scripts\verify_all.bat` to close that gap.

---

## Non-goals (later)

- Editing PDF/DOCX binaries in place
- Shell / git commit from the agent
- Cloud sync or multi-user locking
- Scraping llmfit TUI or inventing Ollama tags from HuggingFace names
