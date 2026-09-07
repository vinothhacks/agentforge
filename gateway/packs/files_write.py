"""Write tools: fs_write, fs_edit, fs_mkdir. Sandboxed, allowlisted, audited.

Every write passes six gates before a byte lands on disk:

1. permission   - deny | ask | allow, from AgentSpec.permissions.fs_write
2. safe_join    - path must resolve inside workspace_root
3. blocked dirs - never .agentforge / .git / .venv / node_modules
4. extension    - allowlist only; no PDF/DOCX binary rewrite in v1
5. size cap     - 512 KB per write
6. backup       - existing bytes copied to .agentforge/backups before replacement

`ask` mode writes nothing: it returns needs_confirm plus a unified diff and a
token. The client shows the diff and calls confirm_pending(token) to apply.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gateway.paths import BLOCKED_PARTS, PathEscapeError, blocked_component, safe_join

ALLOWED_EXTS = {".md", ".txt", ".json", ".csv", ".yaml", ".yml"}
MAX_WRITE_BYTES = 512 * 1024
PENDING_TTL_S = 900.0

_PENDING: dict[str, dict[str, Any]] = {}
_PENDING_LOCK = threading.Lock()


class WriteRejected(ValueError):
    pass


# --------------------------------------------------------------------------- gates


def _permission_error(permission: str, tool: str) -> dict[str, Any] | None:
    if permission == "deny":
        return {
            "error": "permission_denied",
            "tool": tool,
            "detail": "File editing is off. Turn on 'Allow file edits' in the sidebar.",
        }
    if permission not in {"ask", "allow"}:
        return {"error": "permission_denied", "tool": tool, "detail": f"unknown permission {permission!r}"}
    return None


def _check_path(root: Path, rel: str, *, require_ext: bool) -> tuple[Path | None, dict[str, Any] | None]:
    if not rel or not str(rel).strip():
        return None, {"error": "bad_path", "detail": "path is required"}
    try:
        target = safe_join(root, str(rel))
    except PathEscapeError as exc:
        return None, {"error": "path_escape", "path": rel, "detail": str(exc)}
    try:
        rel_parts = target.relative_to(root).parts
    except ValueError:
        return None, {"error": "path_escape", "path": rel, "detail": "resolved outside workspace"}
    if not rel_parts:
        return None, {"error": "bad_path", "path": rel, "detail": "refusing to write the workspace root"}
    blocked = blocked_component(rel_parts)
    if blocked:
        return None, {
            "error": "blocked_path",
            "path": rel,
            "detail": f"{blocked} is protected and cannot be written",
        }
    if require_ext and target.suffix.lower() not in ALLOWED_EXTS:
        return None, {
            "error": "bad_extension",
            "path": rel,
            "allowed": sorted(ALLOWED_EXTS),
            "detail": (
                f"{target.suffix or 'no extension'} is not writable in v1. "
                "PDF and DOCX are read-only."
            ),
        }
    return target, None


def _check_size(content: str) -> dict[str, Any] | None:
    size = len(content.encode("utf-8"))
    if size > MAX_WRITE_BYTES:
        return {
            "error": "too_large",
            "bytes": size,
            "limit": MAX_WRITE_BYTES,
            "detail": f"{size} bytes exceeds the {MAX_WRITE_BYTES} byte per-write limit",
        }
    return None


# ------------------------------------------------------------------ audit + backup


def _audit(root: Path, record: dict[str, Any]) -> None:
    try:
        home = root / ".agentforge"
        home.mkdir(exist_ok=True)
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        with (home / "write_audit.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # An unwritable audit log must not silently block the user's edit,
        # but it must never be the reason a write is lost either.
        pass


def _backup(root: Path, target: Path) -> str | None:
    """Snapshot existing bytes before they are replaced. Makes 'allow' recoverable."""
    if not target.is_file():
        return None
    try:
        rel = target.relative_to(root).as_posix()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", rel)
        dest_dir = root / ".agentforge" / "backups"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{safe_name}.{stamp}.bak"
        n = 1
        while dest.exists():
            dest = dest_dir / f"{safe_name}.{stamp}.{n}.bak"
            n += 1
        shutil.copy2(target, dest)
        return dest.relative_to(root).as_posix()
    except OSError:
        return None


def _atomic_write(target: Path, content: str) -> int:
    """Write via temp file + replace so a crash cannot truncate the original."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
    data = content.encode("utf-8")
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return len(data)


def _diff(old: str, new: str, path: str) -> str:
    lines = difflib.unified_diff(
        old.splitlines(keepends=False),
        new.splitlines(keepends=False),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
        n=3,
    )
    return "\n".join(list(lines)[:400])


# ------------------------------------------------------------------------- pending


def _prune_pending() -> None:
    now = time.time()
    for token in [t for t, p in _PENDING.items() if now - p["created"] > PENDING_TTL_S]:
        _PENDING.pop(token, None)


def _stage(root: Path, action: str, target: Path, rel: str, new_text: str, session: str) -> dict[str, Any]:
    old = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
    token = uuid.uuid4().hex[:16]
    with _PENDING_LOCK:
        _prune_pending()
        _PENDING[token] = {
            "created": time.time(),
            "root": str(root),
            "target": str(target),
            "path": rel,
            "content": new_text,
            "action": action,
            "session": session,
        }
    _audit(root, {"action": action, "path": rel, "mode": "ask", "staged": True, "session": session})
    return {
        "needs_confirm": True,
        "token": token,
        "action": action,
        "path": rel,
        "exists": target.is_file(),
        "bytes": len(new_text.encode("utf-8")),
        "diff": _diff(old, new_text, rel),
        "detail": "Edits require confirmation. Approve the diff to apply this change.",
    }


def confirm_pending(
    token: str,
    *,
    approve: bool = True,
    session: str = "default",
    workspace: Path | None = None,
) -> dict[str, Any]:
    with _PENDING_LOCK:
        _prune_pending()
        pending = _PENDING.pop(token, None)
    if not pending:
        return {"error": "unknown_token", "detail": "This change expired or was already applied."}
    root = Path(pending["root"])
    rel = pending["path"]
    # A token carries its own root. Without this check it is a bearer capability
    # to write into any workspace this process has ever staged an edit for.
    if workspace is not None and root.resolve() != workspace.resolve():
        return {
            "error": "wrong_workspace",
            "detail": "This change belongs to a different workspace and was not applied.",
        }
    if not approve:
        _audit(root, {"action": pending["action"], "path": rel, "mode": "ask", "rejected": True, "session": session})
        return {"ok": True, "applied": False, "path": rel, "detail": "Change discarded."}
    # Re-check the sandbox at apply time and write to the path that check
    # returned -- not the one recorded at stage time, which a swapped symlink
    # could have pointed elsewhere in between.
    target, err = _check_path(root, rel, require_ext=True)
    if err:
        return err
    assert target is not None
    backup = _backup(root, target)
    written = _atomic_write(target, pending["content"])
    _audit(
        root,
        {
            "action": pending["action"],
            "path": rel,
            "mode": "ask",
            "bytes": written,
            "backup": backup,
            "session": session,
            "ok": True,
        },
    )
    return {"ok": True, "applied": True, "path": rel, "bytes": written, "backup": backup}


def pending_count() -> int:
    with _PENDING_LOCK:
        _prune_pending()
        return len(_PENDING)


# --------------------------------------------------------------------------- tools


def fs_write(
    root: Path,
    rel: str,
    content: str,
    mode: str = "create",
    *,
    permission: str = "deny",
    session: str = "default",
) -> dict[str, Any]:
    err = _permission_error(permission, "fs_write")
    if err:
        return err
    target, err = _check_path(root, rel, require_ext=True)
    if err:
        return err
    assert target is not None
    content = content if isinstance(content, str) else str(content)
    err = _check_size(content)
    if err:
        return err
    if mode not in {"create", "overwrite"}:
        return {"error": "bad_mode", "detail": "mode must be create or overwrite"}
    if target.is_dir():
        return {"error": "bad_path", "path": rel, "detail": "path is a directory"}
    if mode == "create" and target.exists():
        return {
            "error": "exists",
            "path": rel,
            "detail": "File already exists. Use mode=overwrite or fs_edit to change it.",
        }

    if permission == "ask":
        return _stage(root, "write", target, rel, content, session)

    backup = _backup(root, target)
    try:
        written = _atomic_write(target, content)
    except OSError as exc:
        _audit(root, {"action": "write", "path": rel, "error": str(exc), "session": session})
        return {"error": "write_failed", "path": rel, "detail": str(exc)[:200]}
    _audit(
        root,
        {
            "action": "write",
            "path": rel,
            "mode": mode,
            "bytes": written,
            "backup": backup,
            "session": session,
            "ok": True,
        },
    )
    return {
        "ok": True,
        "path": target.relative_to(root).as_posix(),
        "bytes": written,
        "mode": mode,
        "backup": backup,
        "created": mode == "create",
    }


def fs_edit(
    root: Path,
    rel: str,
    old_text: str,
    new_text: str,
    *,
    permission: str = "deny",
    session: str = "default",
) -> dict[str, Any]:
    err = _permission_error(permission, "fs_edit")
    if err:
        return err
    target, err = _check_path(root, rel, require_ext=True)
    if err:
        return err
    assert target is not None
    if not target.is_file():
        return {"error": "wrong_file", "path": rel, "detail": "file does not exist"}
    if not isinstance(old_text, str) or old_text == "":
        return {"error": "bad_args", "detail": "old_text must be a non-empty string"}
    new_text = new_text if isinstance(new_text, str) else str(new_text)

    try:
        original = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"error": "wrong_file", "path": rel, "detail": str(exc)[:200]}

    occurrences = original.count(old_text)
    if occurrences == 0:
        # Idempotency: a re-run of an applied edit is a success, not a failure.
        if new_text and new_text in original:
            return {
                "ok": True,
                "path": target.relative_to(root).as_posix(),
                "already_applied": True,
                "replacements": 0,
                "detail": "new_text is already present; nothing to change.",
            }
        return {
            "error": "no_match",
            "path": rel,
            "detail": "old_text was not found. Read the file and copy the exact text, including whitespace.",
        }
    if occurrences > 1:
        return {
            "error": "ambiguous_match",
            "path": rel,
            "count": occurrences,
            "detail": f"old_text appears {occurrences} times. Include surrounding lines to make it unique.",
        }

    updated = original.replace(old_text, new_text, 1)
    err = _check_size(updated)
    if err:
        return err

    if permission == "ask":
        return _stage(root, "edit", target, rel, updated, session)

    backup = _backup(root, target)
    try:
        written = _atomic_write(target, updated)
    except OSError as exc:
        _audit(root, {"action": "edit", "path": rel, "error": str(exc), "session": session})
        return {"error": "write_failed", "path": rel, "detail": str(exc)[:200]}
    _audit(
        root,
        {
            "action": "edit",
            "path": rel,
            "bytes": written,
            "backup": backup,
            "session": session,
            "ok": True,
        },
    )
    return {
        "ok": True,
        "path": target.relative_to(root).as_posix(),
        "replacements": 1,
        "bytes": written,
        "backup": backup,
    }


def fs_mkdir(root: Path, rel: str, *, permission: str = "deny", session: str = "default") -> dict[str, Any]:
    err = _permission_error(permission, "fs_mkdir")
    if err:
        return err
    target, err = _check_path(root, rel, require_ext=False)
    if err:
        return err
    assert target is not None
    if target.is_file():
        return {"error": "exists", "path": rel, "detail": "a file already exists at that path"}
    already = target.is_dir()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {"error": "write_failed", "path": rel, "detail": str(exc)[:200]}
    _audit(root, {"action": "mkdir", "path": rel, "session": session, "ok": True, "already": already})
    return {"ok": True, "path": target.relative_to(root).as_posix(), "created": not already}


def read_audit(root: Path, limit: int = 100) -> list[dict[str, Any]]:
    path = root / ".agentforge" / "write_audit.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]
