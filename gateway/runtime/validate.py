"""JSON Schema validation *before* executing a tool. Invalid args never run."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator


class SchemaReject(ValueError):
    def __init__(self, tool: str, errors: list[str]):
        super().__init__(f"schema_reject:{tool}:{errors}")
        self.tool = tool
        self.errors = errors


def validate_args(schema: dict[str, Any], args: dict[str, Any], tool: str) -> dict[str, Any]:
    validator = Draft202012Validator(schema)
    errors = [e.message for e in validator.iter_errors(args)]
    if errors:
        raise SchemaReject(tool, errors)
    return args
