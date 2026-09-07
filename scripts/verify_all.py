"""One-command end-to-end verification. Run this on the machine that has Ollama.

    python scripts/verify_all.py --dir "C:\\path\\to\\Vinoth_N_Package_v1"

Steps
  1. pytest                     - the whole unit suite
  2. temp copy of the workspace - so no write can touch the real folder
  3. agentforge --ingest        - build the hybrid index over the copy
  4. boot the gateway           - on the first free port
  5. pin a Ready local model    - smallest installed Ollama tag
  6. fs_write permission        - set from --permission (default allow)
  7. 13-question harness        - 10 read + 3 write, verified against disk
  8. verdict                    - exit 0 only if every stage passed
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP_COPY = {".agentforge", ".git", ".venv", "node_modules", "__pycache__"}


def say(step: str, msg: str) -> None:
    print(f"\n=== {step} : {msg}", flush=True)


def free_port(start: int = 8789, span: int = 40) -> int:
    for port in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port")


def temp_copy(source: Path) -> Path:
    dest = Path(tempfile.mkdtemp(prefix="agentforge-verify-"))
    for item in source.iterdir():
        if item.name in SKIP_COPY:
            continue
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    return dest


def get(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post(url: str, body: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_healthy(base: str, proc: subprocess.Popen, timeout: float = 90.0) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return None
        try:
            return get(base + "/api/health", timeout=3.0)
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(1.0)
    return None


def pick_local_model() -> str | None:
    try:
        data = get("http://127.0.0.1:11434/api/tags", timeout=5.0)
    except OSError:
        return None
    names = [
        str(m.get("name") or m.get("model") or "")
        for m in (data.get("models") or [])
    ]
    names = [n for n in names if n and not n.startswith("nomic-embed")]
    if not names:
        return None
    af = sorted((n for n in names if n.startswith("af-")), key=len)
    return af[0] if af else sorted(names, key=lambda n: (len(n), n))[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="Real resume workspace. It is COPIED, never written.")
    ap.add_argument("--permission", default="allow", choices=["deny", "ask", "allow"])
    ap.add_argument("--model", default=None, help="Ollama tag. Default: smallest installed.")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--keep", action="store_true", help="Keep the temp workspace for inspection.")
    args = ap.parse_args()

    source = Path(args.dir).expanduser().resolve()
    if not source.is_dir():
        print(f"--dir is not a folder: {source}", file=sys.stderr)
        return 2

    stages: dict[str, bool] = {}
    py = sys.executable

    # ---------------------------------------------------------------- 1 pytest
    if args.skip_tests:
        say("1/7", "pytest skipped")
    else:
        say("1/7", "running pytest")
        rc = subprocess.run([py, "-m", "pytest", "-q"], cwd=REPO).returncode
        stages["pytest"] = rc == 0
        if rc != 0:
            print("\npytest FAILED - fix the unit suite before trusting the live run.")
            return 1

    # ------------------------------------------------------------ 2 temp copy
    say("2/7", "copying workspace")
    ws = temp_copy(source)
    print(f"    {source}\n -> {ws}")
    (ws / "notes.md").write_text("# Notes\n\nStatus: DRAFT\n", encoding="utf-8")

    proc = None
    try:
        # ------------------------------------------------------------ 3 ingest
        say("3/7", "ingesting")
        rc = subprocess.run([py, "-m", "gateway.cli", "--dir", str(ws), "--ingest"],
                            cwd=REPO).returncode
        stages["ingest"] = rc == 0
        if rc != 0:
            print("ingest FAILED")
            return 1

        # -------------------------------------------------------------- 4 boot
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        say("4/7", f"starting gateway on {base}")
        env = dict(os.environ)
        env.pop("OPENROUTER_API_KEY", None)  # local model only, per the plan
        proc = subprocess.Popen(
            [py, "-m", "gateway.cli", "--dir", str(ws), "--port", str(port), "--no-browser"],
            cwd=REPO, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
        health = wait_healthy(base, proc)
        stages["server"] = health is not None
        if not health:
            print("server did not become healthy - is another agentforge running?")
            return 1
        print(f"    workspace={health.get('workspace')} ingested={health.get('ingested')}")

        # -------------------------------------------------------- 5 pin a model
        say("5/7", "pinning a local model")
        model = args.model or pick_local_model()
        if not model:
            print("No Ollama model installed and reachable at 127.0.0.1:11434.")
            print("Install one first:  ollama pull qwen2.5:0.5b")
            return 1
        post(base + "/api/model", {"provider": "ollama", "name": model})
        print(f"    pinned {model}")
        stages["model"] = True

        # ------------------------------------------------------- 6 permissions
        say("6/7", f"setting fs_write={args.permission}")
        got = post(base + "/api/permissions", {"fs_write": args.permission})
        stages["permission"] = got.get("fs_write") == args.permission
        print(f"    fs_write={got.get('fs_write')}")

        # ----------------------------------------------------------- 7 harness
        say("7/7", "running 10 read + 3 write questions")
        rc = subprocess.run(
            [py, "scripts/tenq_resume_chat.py", "--base", base, "--workspace", str(ws)],
            cwd=REPO,
        ).returncode
        stages["harness"] = rc == 0
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if args.keep:
            print(f"\ntemp workspace kept at {ws}")
        else:
            shutil.rmtree(ws, ignore_errors=True)

    print("\n" + "=" * 58)
    for name, ok in stages.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 58)
    allok = all(stages.values())
    print("ALL GREEN" if allok else "SOMETHING FAILED - see above")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
