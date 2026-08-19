"""agentforge CLI: uvx/pip entry. OpenRouter-first, --dir workspace, --ingest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path

import uvicorn

from gateway.db import Store
from gateway.evals import run_offline_golden, write_sample_docs
from gateway.packs.rag import ingest_workspace
from gateway.paths import resolve_workspace
from gateway.probe.experiment import run_experiment
from gateway.compiler import compile_workspace, persist_spec


def _home(workspace: Path) -> Path:
    d = workspace / ".agentforge"
    d.mkdir(exist_ok=True)
    return d


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="agentforge",
        description="Convert an LLM into a tool-using agent. OpenRouter first, local later.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "From Downloads or any folder (install once, then run):\n"
            "  uv tool install git+https://github.com/vinothhacks/agentforge.git\n"
            "  agentforge --dir path\\to\\pdfs\n\n"
            "These fail on purpose:\n"
            "  uv run agentforge   - only works after clone + cd into this repo\n"
            "  uvx agentforge      - PyPI agentforge is a different package with no CLI"
        ),
    )
    p.add_argument("--dir", default=".", help="Workspace folder (Claude Code pattern).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8788)
    p.add_argument("--ingest", action="store_true", help="Ingest PDFs/txt into hybrid index and exit.")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--samples", action="store_true", help="Write sample .txt files into --dir.")
    p.add_argument("--experiment", action="store_true", help="Run P-1 disagreement experiment.")
    p.add_argument("--eval", action="store_true", dest="run_eval", help="Run offline golden evals.")
    p.add_argument("--eval-fixtures", action="store_true", help="Alias for --samples.")
    p.add_argument("--e2e", action="store_true", help="Live start-to-end test against a real model.")
    p.add_argument("--provider", default=None, help="For --e2e: openrouter | ollama.")
    p.add_argument("--model", default=None, help="For --e2e: model id, e.g. gemma4:cloud.")
    args = p.parse_args(argv)

    workspace = Path(args.dir).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if args.samples or args.eval_fixtures:
        write_sample_docs(workspace)
        print(f"wrote sample files in {workspace}")
        return 0

    if args.run_eval:
        if not any(workspace.glob("DOC-*")) and not any(workspace.glob("*.pdf")):
            write_sample_docs(workspace)
        scored = run_offline_golden(workspace)
        print(json.dumps(scored, indent=2))
        enum = scored.get("enumeration") or {}
        if enum.get("cause") == "incomplete_enumeration":
            return 3
        if enum and enum.get("recall", 0) < 0.90:
            return 3
        return 0

    if args.experiment:
        out = Path("p1_disagreement.md")
        key = os.environ.get("OPENROUTER_API_KEY")
        result = run_experiment(n=5, mode="fast", api_key=key, only_installed=True, out_path=out)
        print(f"D={result['D']} gate={result['gate']}")
        print(f"wrote {out}")
        return 0

    if args.ingest:
        warning = ingest_workspace(workspace)
        print(json.dumps(warning, indent=2))
        if warning.get("no_text_layer_count"):
            print(
                f"WARNING: {warning['no_text_layer_count']} files have no text layer "
                "and will vanish from lexical search. OCR is out of v1.",
                file=sys.stderr,
            )
        return 0

    try:
        workspace = resolve_workspace(workspace)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(exc, file=sys.stderr)
        return 2

    home = _home(workspace)
    store = Store(home / "app.sqlite")
    spec = compile_workspace(workspace)
    persist_spec(spec, home / "agentspec.yaml")

    idx_flag = home / "fts.sqlite"
    if not (workspace / ".agentforge" / "chunks.jsonl").exists():
        print("no index yet — ingesting (decision b: CLI ingest, not a UI)…")
        warning = ingest_workspace(workspace)
        print(
            f"ingested {warning.get('files_scanned', 0)} files, "
            f"{warning.get('chunks', 0)} chunks, "
            f"no-text-layer={warning.get('no_text_layer_count', 0)}"
        )

    from gateway.app import make_app

    app = make_app(workspace, store)
    url = f"http://{args.host}:{args.port}"
    print(f"AgentForge  workspace={workspace}")
    print(f"Open {url}  - paste an OpenRouter key, ask about your files.")
    print("llmfit and Ollama download links are in the sidebar.")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
