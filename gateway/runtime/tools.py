"""v1 tools: rag_search, fs_list, fs_read. Hybrid retrieval is *inside* rag_search."""

from __future__ import annotations

from typing import Any, Callable

from gateway.runtime.validate import SchemaReject, validate_args

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "rag_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query over PDA PDFs"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 80, "default": 12},
        },
        "required": ["query"],
    },
    "fs_list": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative directory under the workspace", "default": "."},
        },
        "required": [],
    },
    "fs_read": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative file path under the workspace"},
            "max_chars": {"type": "integer", "minimum": 200, "maximum": 8000, "default": 4000},
        },
        "required": ["path"],
    },
}

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": TOOL_SCHEMAS[name],
        },
    }
    for name, desc in [
        ("rag_search", "Search the PDA folder. Hybrid lexical (FTS5) + vectors. For 'list every X' pass the term or the full question — lexical paths enumerate matching files."),
        ("fs_list", "List files in the workspace folder (read-only)."),
        ("fs_read", "Read a file from the workspace (read-only). Use to cite a page."),
    ]
]


def openai_tools(enabled: list[str]) -> list[dict[str, Any]]:
    return [t for t in OPENAI_TOOLS if t["function"]["name"] in enabled]


Executor = Callable[[str, dict[str, Any]], dict[str, Any]]


def dispatch(name: str, args: dict[str, Any], executor: Executor) -> dict[str, Any]:
    if name not in TOOL_SCHEMAS:
        return {"error": "bad_tool", "tool": name}
    try:
        clean = validate_args(TOOL_SCHEMAS[name], args, name)
    except SchemaReject as exc:
        return {"error": "schema_reject", "tool": name, "errors": exc.errors}
    return executor(name, clean)
