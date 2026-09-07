"""Read-only filesystem tools, sandboxed to workspace_root.

fs_list gained a glob filter, optional recursion and an exact `count` so
"how many PDFs are there" is answered by the filesystem, not by the model
counting a truncated list.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from gateway.packs.extract import DOCX_EXTS, PDF_EXTS, extract_pages
from gateway.paths import PathEscapeError, blocked_component, safe_join

MAX_ENTRIES = 500
MAX_READ_CHARS = 20000


def _skipped(rel_parts: tuple[str, ...]) -> bool:
    return blocked_component(rel_parts, hidden=True) is not None


def _matches(name: str, glob: str | None) -> bool:
    if not glob:
        return True
    pattern = glob.strip()
    if not pattern:
        return True
    # Bare extensions ("pdf", ".pdf") are the common model output; treat as *.pdf
    if pattern.startswith("."):
        pattern = "*" + pattern
    elif "*" not in pattern and "?" not in pattern and "." not in pattern:
        pattern = "*." + pattern
    return fnmatch.fnmatch(name.lower(), pattern.lower())


def _guard(root: Path, rel: str) -> tuple[Path | None, dict[str, Any] | None]:
    """Resolve a relative path and refuse anything protected or outside the root."""
    try:
        target = safe_join(root, rel or ".")
    except PathEscapeError as exc:
        return None, {"error": "path_escape", "path": rel, "detail": str(exc)}
    try:
        rel_parts = target.relative_to(root.resolve()).parts
    except ValueError:
        return None, {"error": "path_escape", "path": rel, "detail": "resolved outside workspace"}
    blocked = blocked_component(rel_parts, hidden=True)
    if blocked:
        return None, {
            "error": "blocked_path",
            "path": rel,
            "detail": f"{blocked} is protected and cannot be read",
        }
    return target, None


def fs_list(
    root: Path,
    rel: str = ".",
    glob: str | None = None,
    recursive: bool = False,
) -> dict[str, Any]:
    target, err = _guard(root, rel or ".")
    if err:
        return err
    assert target is not None
    if not target.exists():
        return {"error": "wrong_file", "path": rel}
    if not target.is_dir():
        return {"error": "wrong_file", "path": rel, "detail": "not a directory"}

    children = sorted(target.rglob("*") if recursive else target.iterdir())
    entries: list[dict[str, Any]] = []
    matched = 0
    file_count = 0
    dir_count = 0
    for child in children:
        try:
            rel_child = child.relative_to(root)
        except ValueError:
            continue
        if _skipped(rel_child.parts):
            continue
        is_dir = child.is_dir()
        # A glob filter is a question about files; never let folders inflate a count.
        if glob and (is_dir or not _matches(child.name, glob)):
            continue
        matched += 1
        if is_dir:
            dir_count += 1
        else:
            file_count += 1
        if len(entries) < MAX_ENTRIES:
            entries.append(
                {
                    "name": child.name,
                    "path": rel_child.as_posix(),
                    "type": "dir" if is_dir else "file",
                    "bytes": child.stat().st_size if not is_dir else None,
                }
            )
    return {
        "path": rel or ".",
        "glob": glob or None,
        "recursive": bool(recursive),
        "count": matched,
        "file_count": file_count,
        "dir_count": dir_count,
        "truncated": matched > len(entries),
        "entries": entries,
        "paths": [e["path"] for e in entries if e["type"] == "file"],
    }


def fs_read(root: Path, rel: str, max_chars: int = 4000) -> dict[str, Any]:
    target, err = _guard(root, rel)
    if err:
        return err
    assert target is not None
    if not target.is_file():
        return {"error": "wrong_file", "path": rel}

    cap = max(200, min(int(max_chars or 4000), MAX_READ_CHARS))
    suffix = target.suffix.lower()
    try:
        if suffix in PDF_EXTS or suffix in DOCX_EXTS:
            pages, has_text, kind = extract_pages(target)
            if not has_text:
                return {
                    "error": "no_text_layer",
                    "path": rel.replace("\\", "/"),
                    "kind": kind,
                    "detail": "scanned or empty document; OCR is not enabled",
                }
            blocks = []
            for page in pages:
                body = (page.get("text") or "").strip()
                if not body:
                    continue
                marker = f"[page {page['page']}]" if kind == "pdf" else "[docx]"
                blocks.append(f"{marker}\n{body}")
            text = "\n\n".join(blocks)
            return {
                "path": rel.replace("\\", "/"),
                "kind": kind,
                "pages": len(pages),
                "text": text[:cap],
                "truncated": len(text) > cap,
            }
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"error": "wrong_file", "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001 - a corrupt PDF must not 500 the chat
        return {"error": "extract_failed", "path": rel, "detail": str(exc)[:200]}

    return {
        "path": rel.replace("\\", "/"),
        "kind": "text",
        "text": text[:cap],
        "truncated": len(text) > cap,
    }
