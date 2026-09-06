"""v1 tools: rag_search, fs_list, fs_read, fs_write, fs_edit, fs_mkdir.

Hybrid retrieval lives *inside* rag_search. Write tools are only advertised to
the model when permissions.fs_write is ask or allow, so a read-only agent never
sees a name it could hallucinate a call to.
"""

from __future__ import annotations

from typing import Any, Callable

from gateway.runtime.validate import SchemaReject, validate_args

READ_TOOLS = ["rag_search", "fs_list", "fs_read"]
WRITE_TOOLS = ["fs_write", "fs_edit", "fs_mkdir"]
ALL_TOOLS = READ_TOOLS + WRITE_TOOLS

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "rag_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query over workspace files"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 80, "default": 12},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "fs_list": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative directory under the workspace", "default": "."},
            "glob": {
                "type": "string",
                "description": "Optional filename filter, e.g. '*.pdf', '*.docx', '*Resume*'. Filters files only.",
            },
            "recursive": {"type": "boolean", "description": "Include sub-folders", "default": False},
        },
        "required": [],
        "additionalProperties": False,
    },
    "fs_read": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative file path under the workspace"},
            "max_chars": {"type": "integer", "minimum": 200, "maximum": 20000, "default": 4000},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    "fs_write": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative file path under the workspace. Must end in .md .txt .json .csv .yaml or .yml",
            },
            "content": {"type": "string", "description": "Full file contents to write"},
            "mode": {
                "type": "string",
                "enum": ["create", "overwrite"],
                "default": "create",
                "description": "create fails if the file exists; overwrite replaces it",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    "fs_edit": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative file path under the workspace"},
            "old_text": {
                "type": "string",
                "description": "Exact text to replace. Must appear exactly once in the file.",
            },
            "new_text": {"type": "string", "description": "Replacement text"},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    },
    "fs_mkdir": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative directory to create under the workspace"},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "rag_search": (
        "Search the workspace. Hybrid lexical (FTS5) + vectors. For 'list every X' pass the term "
        "or the full question - lexical paths enumerate matching files."
    ),
    "fs_list": (
        "List files in the workspace folder. Pass glob='*.pdf' to filter and read the exact "
        "'count' field for how-many questions. Read-only."
    ),
    "fs_read": (
        "Read a file from the workspace and return its text. Works on .txt, .md, PDF and DOCX. "
        "Read-only."
    ),
    "fs_write": (
        "Create a new file or overwrite an existing one under the workspace. Text formats only "
        "(.md .txt .json .csv .yaml .yml). Use for new files such as summary.md."
    ),
    "fs_edit": (
        "Replace one exact string in an existing workspace file. Fails if old_text is missing or "
        "appears more than once. Prefer this over fs_write for small changes."
    ),
    "fs_mkdir": "Create a directory under the workspace.",
}

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "parameters": TOOL_SCHEMAS[name],
        },
    }
    for name in ALL_TOOLS
]


def tools_for(enabled: list[str], permission: str = "deny") -> list[str]:
    """Resolve the advertised tool names for a spec + permission pair."""
    allowed = [t for t in enabled if t in TOOL_SCHEMAS]
    if permission not in {"ask", "allow"}:
        allowed = [t for t in allowed if t not in WRITE_TOOLS]
    return allowed


def openai_tools(enabled: list[str], permission: str = "deny") -> list[dict[str, Any]]:
    names = set(tools_for(enabled, permission))
    return [t for t in OPENAI_TOOLS if t["function"]["name"] in names]


Executor = Callable[[str, dict[str, Any]], dict[str, Any]]


def _sibling_for(name: str, args: dict[str, Any], available: list[str] | None) -> str | None:
    """The one other read tool these exact args validate against, if any."""
    if not isinstance(args, dict) or not args:
        # No arguments at all is a missing-argument error, not a call aimed at
        # the wrong tool. Rerouting it would run something nobody asked for.
        return None
    pool = [t for t in READ_TOOLS if t != name]
    if available is not None:
        pool = [t for t in pool if t in available]
    matches = []
    for cand in pool:
        try:
            validate_args(TOOL_SCHEMAS[cand], args, cand)
        except SchemaReject:
            continue
        matches.append(cand)
    return matches[0] if len(matches) == 1 else None


def _reject_detail(name: str, errors: Any) -> str:
    """schema_reject is internal vocabulary. Users get a sentence."""
    props = sorted((TOOL_SCHEMAS.get(name) or {}).get("properties") or {})
    allowed = ", ".join(f"`{p}`" for p in props) or "no arguments"
    return (
        f"`{name}` was called with arguments it does not accept "
        f"({errors}). It takes {allowed}."
    )


def dispatch(
    name: str,
    args: dict[str, Any],
    executor: Executor,
    *,
    available: list[str] | None = None,
) -> dict[str, Any]:
    if name not in TOOL_SCHEMAS:
        # Give the model the real names so the next turn can self-correct
        # instead of repeating an invented call.
        return {
            "error": "bad_tool",
            "tool": name,
            "valid_tools": available if available is not None else list(TOOL_SCHEMAS),
            "detail": "That tool does not exist. Call one of valid_tools using its exact name.",
        }
    if available is not None and name not in available:
        return {
            "error": "tool_not_enabled",
            "tool": name,
            "valid_tools": available,
            "detail": "That tool is not enabled for this agent.",
        }
    try:
        clean = validate_args(TOOL_SCHEMAS[name], args, name)
    except SchemaReject as exc:
        # Small tool-calling models routinely address the right arguments to
        # the wrong tool -- `fs_list(query=..., limit=...)` is rag_search's
        # signature. If the args fit exactly one sibling, run that instead of
        # dead-ending the turn.
        sibling = _sibling_for(name, args, available)
        if sibling:
            clean = validate_args(TOOL_SCHEMAS[sibling], args, sibling)
            out = executor(sibling, clean)
            if isinstance(out, dict):
                out = dict(out)
                out["rerouted_from"] = name
            return out
        return {
            "error": "schema_reject",
            "tool": name,
            "errors": exc.errors,
            "detail": _reject_detail(name, exc.errors),
        }
    return executor(name, clean)
