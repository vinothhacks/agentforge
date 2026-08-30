"""Write tools must be invisible unless permission allows them, and a wrong
tool name must come back with the real names so the model can self-correct."""

from __future__ import annotations

from gateway.runtime.tools import (
    ALL_TOOLS,
    READ_TOOLS,
    WRITE_TOOLS,
    dispatch,
    openai_tools,
    tools_for,
)


def test_deny_hides_write_tools():
    names = tools_for(ALL_TOOLS, "deny")
    assert names == READ_TOOLS
    advertised = {t["function"]["name"] for t in openai_tools(ALL_TOOLS, "deny")}
    assert advertised.isdisjoint(WRITE_TOOLS)


def test_ask_and_allow_expose_write_tools():
    for permission in ("ask", "allow"):
        names = tools_for(ALL_TOOLS, permission)
        assert set(names) == set(ALL_TOOLS), permission


def test_unknown_permission_is_treated_as_deny():
    assert tools_for(ALL_TOOLS, "sure_why_not") == READ_TOOLS


def test_spec_tool_list_still_filters():
    assert tools_for(["fs_list"], "allow") == ["fs_list"]
    assert tools_for(["fs_list", "not_a_tool"], "allow") == ["fs_list"]


def test_bad_tool_returns_valid_names():
    """The Phase 0 failure mode: the model invented a tool and got no way back."""
    out = dispatch("read_the_readme", {}, lambda n, a: {}, available=READ_TOOLS)
    assert out["error"] == "bad_tool"
    assert out["valid_tools"] == READ_TOOLS


def test_disabled_tool_is_refused_even_if_schema_exists():
    called = []
    out = dispatch("fs_write", {"path": "a.md", "content": "x"},
                   lambda n, a: called.append(n), available=READ_TOOLS)
    assert out["error"] == "tool_not_enabled"
    assert called == []


def test_write_tool_runs_when_enabled():
    seen = {}

    def executor(name, args):
        seen.update(name=name, args=args)
        return {"ok": True}

    out = dispatch("fs_write", {"path": "a.md", "content": "x"}, executor, available=ALL_TOOLS)
    assert out == {"ok": True}
    assert seen["name"] == "fs_write"


def test_schema_rejects_unknown_argument():
    out = dispatch("fs_list", {"path": ".", "sudo": True}, lambda n, a: {}, available=ALL_TOOLS)
    assert out["error"] == "schema_reject"


def test_fs_edit_requires_all_three_args():
    out = dispatch("fs_edit", {"path": "a.md"}, lambda n, a: {}, available=ALL_TOOLS)
    assert out["error"] == "schema_reject"
