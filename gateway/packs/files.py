"""Read-only filesystem tools, sandboxed to workspace_root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gateway.paths import PathEscapeError, safe_join

SKIP_DIRS = {".agentforge", ".git", "__pycache__", "node_modules"}


def fs_list(root: Path, rel: str = ".") -> dict[str, Any]:
    try:
        target = safe_join(root, rel or ".")
    except PathEscapeError as exc:
        return {"error": "path_escape", "detail": str(exc)}
    if not target.exists():
        return {"error": "wrong_file", "path": rel}
    if not target.is_dir():
        return {"error": "wrong_file", "path": rel, "detail": "not a directory"}
    entries = []
    for child in sorted(target.iterdir()):
        if child.name in SKIP_DIRS or child.name.startswith("."):
            continue
        entries.append(
            {
                "name": child.name,
                "path": str(child.relative_to(root)).replace("\\", "/"),
                "type": "dir" if child.is_dir() else "file",
                "bytes": child.stat().st_size if child.is_file() else None,
            }
        )
    return {"path": rel, "entries": entries[:200]}


def fs_read(root: Path, rel: str, max_chars: int = 4000) -> dict[str, Any]:
    try:
        target = safe_join(root, rel)
    except PathEscapeError as exc:
        return {"error": "path_escape", "detail": str(exc)}
    if not target.is_file():
        return {"error": "wrong_file", "path": rel}
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"error": "wrong_file", "detail": str(exc)}
    truncated = len(text) > max_chars
    return {
        "path": rel.replace("\\", "/"),
        "text": text[:max_chars],
        "truncated": truncated,
    }
