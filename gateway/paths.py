"""Sandbox: every file path must resolve inside workspace_root.

Path separators are normalised before joining so behaviour is identical on
Windows and POSIX. Without this, "..\\outside.txt" escapes on Windows but is a
legal *filename* on Linux, so the same model output means two different things
depending on the host. Traversal is rejected by inspecting the components, not
only by comparing the resolved result.
"""

from __future__ import annotations

import re
from pathlib import Path

# CON, PRN, AUX, NUL, COM1-9, LPT1-9 are device names on Windows: opening one
# for writing does not create a file, it talks to hardware.
_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {
    f"lpt{i}" for i in range(1, 10)
}
_DRIVE = re.compile(r"^[A-Za-z]:")

# Directories that hold our own state, the user's VCS, or a virtualenv. Reading
# .agentforge is what turns a path bug into an API-key leak: app.sqlite lives
# there. Writing any of them corrupts state the user did not ask us to touch.
BLOCKED_PARTS = {
    ".agentforge",
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
}


class PathEscapeError(ValueError):
    pass


def resolve_workspace(raw: str | Path) -> Path:
    root = Path(raw).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"workspace does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"workspace is not a directory: {root}")
    return root


def split_relative(rel: str) -> list[str]:
    """Normalise a user/model supplied path into safe components.

    Raises PathEscapeError for traversal, drive letters, UNC prefixes and
    Windows device names. A leading "/" is read as workspace-relative, since
    models routinely emit "/notes.md" meaning "notes.md in this folder".
    """
    raw = str(rel or "").strip().replace("\\", "/")
    if "\x00" in raw:
        raise PathEscapeError("path contains a NUL byte")
    if _DRIVE.match(raw):
        raise PathEscapeError(f"absolute drive path is not allowed: {rel}")
    if raw.startswith("//"):
        raise PathEscapeError(f"UNC path is not allowed: {rel}")
    parts: list[str] = []
    for part in raw.split("/"):
        part = part.strip()
        if part in ("", "."):
            continue
        if part == "..":
            raise PathEscapeError(f"path escapes workspace: {rel}")
        if part.split(".")[0].lower() in _RESERVED:
            raise PathEscapeError(f"reserved device name: {part}")
        parts.append(part)
    return parts


def safe_join(workspace_root: Path, rel: str) -> Path:
    """Join a user-supplied relative path and refuse anything outside the root."""
    root = workspace_root.resolve()
    parts = split_relative(rel)
    candidate = root.joinpath(*parts) if parts else root
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        # A symlink or junction inside the workspace pointed back out.
        raise PathEscapeError(f"path escapes workspace: {rel}") from exc
    return candidate


def blocked_component(parts: tuple[str, ...] | list[str], *, hidden: bool = False) -> str | None:
    """Return the first protected component of a workspace-relative path.

    `hidden=True` also refuses dot-prefixed names, which is what the read side
    wants: it keeps .env, .git/config and .agentforge/app.sqlite out of reach of
    both the model and the download endpoint. The write side passes hidden=False
    so an explicit dotfile write stays possible.
    """
    for part in parts:
        if part in BLOCKED_PARTS:
            return part
        if hidden and part.startswith("."):
            return part
    return None


def is_within(workspace_root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(workspace_root.resolve())
        return True
    except ValueError:
        return False
