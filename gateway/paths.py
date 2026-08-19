"""Sandbox: every file path must resolve inside workspace_root. No `..`, no junctions out."""

from __future__ import annotations

from pathlib import Path


class PathEscapeError(ValueError):
    pass


def resolve_workspace(raw: str | Path) -> Path:
    root = Path(raw).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"workspace does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"workspace is not a directory: {root}")
    return root


def safe_join(workspace_root: Path, rel: str) -> Path:
    """Join a user-supplied relative path and refuse anything outside the root."""
    root = workspace_root.resolve()
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PathEscapeError(f"path escapes workspace: {rel}") from exc
    # Windows junctions: resolve() already followed them; re-check.
    if root not in candidate.parents and candidate != root:
        raise PathEscapeError(f"path escapes workspace: {rel}")
    return candidate


def is_within(workspace_root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(workspace_root.resolve())
        return True
    except ValueError:
        return False
