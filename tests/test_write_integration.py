"""Whole agent loop, fake model, real tools: prove a write actually lands.

Covers the three Phase 2 acceptance criteria from docs/plan-file-read-edit.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from gateway.packs.files import fs_list, fs_read
from gateway.packs.files_write import fs_edit, fs_mkdir, fs_write
from gateway.runtime.loop import run_loop
from gateway.spec import AgentSpec


def make_executor(root: Path, permission: str):
    def executor(name, args):
        if name == "fs_list":
            return fs_list(root, args.get("path") or ".", glob=args.get("glob"),
                           recursive=bool(args.get("recursive")))
        if name == "fs_read":
            return fs_read(root, args["path"], int(args.get("max_chars") or 4000))
        if name == "fs_write":
            return fs_write(root, args["path"], args.get("content") or "",
                            args.get("mode") or "create", permission=permission)
        if name == "fs_edit":
            return fs_edit(root, args["path"], args.get("old_text") or "",
                           args.get("new_text") or "", permission=permission)
        if name == "fs_mkdir":
            return fs_mkdir(root, args["path"], permission=permission)
        return {"error": "bad_tool", "tool": name}

    return executor


def scripted(*turns):
    """Each turn is either a list of (name, args) tool calls, or a final string."""
    state = {"i": 0}

    def complete_fn(**kwargs):
        turn = turns[min(state["i"], len(turns) - 1)]
        state["i"] += 1
        base = {"prompt_tokens": 10, "completion_tokens": 5, "elapsed_s": 0.01}
        if isinstance(turn, str):
            return {**base, "content": turn, "tool_calls": [], "finish_reason": "stop"}
        calls = [
            {"id": f"c{i}", "name": n, "arguments": json.dumps(a)}
            for i, (n, a) in enumerate(turn)
        ]
        return {**base, "content": "", "tool_calls": calls, "finish_reason": "tool_calls"}

    return complete_fn


def spec_for(root: Path, permission: str) -> AgentSpec:
    spec = AgentSpec(workspace_root=str(root))
    spec.model_pin.provider = "openrouter"
    spec.model_pin.name = "fake"
    spec.permissions.fs_write = permission  # type: ignore[assignment]
    spec.budgets.max_steps = 3
    return spec


# ------------------------------------------------- plan acceptance criterion 1


def test_create_summary_md_lands_on_disk(tmp_path: Path):
    (tmp_path / "a_Resume.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "b_Resume.pdf").write_bytes(b"%PDF-1.4")

    out = run_loop(
        spec_for(tmp_path, "allow"),
        "Create summary.md listing all resume PDFs",
        history=[],
        executor=make_executor(tmp_path, "allow"),
        complete_fn=scripted(
            [("fs_list", {"glob": "*.pdf"})],
            [("fs_write", {"path": "summary.md",
                           "content": "# Resumes\n\n- a_Resume.pdf\n- b_Resume.pdf\n"})],
            "Created summary.md with both resume PDFs.",
        ),
    )
    body = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "a_Resume.pdf" in body and "b_Resume.pdf" in body
    assert out["writes"] and out["writes"][0]["path"] == "summary.md"
    assert out["text"].strip()


# ------------------------------------------------- plan acceptance criterion 2


def test_edit_then_rerun_is_idempotent(tmp_path: Path):
    notes = tmp_path / "README_resume_usage_notes.md"
    notes.write_text("Status: X\n", encoding="utf-8")

    def go():
        return run_loop(
            spec_for(tmp_path, "allow"),
            "In README_resume_usage_notes.md replace X with Y",
            history=[],
            executor=make_executor(tmp_path, "allow"),
            complete_fn=scripted(
                [("fs_edit", {"path": "README_resume_usage_notes.md",
                              "old_text": "Status: X", "new_text": "Status: Y"})],
                "Replaced X with Y.",
            ),
        )

    go()
    assert notes.read_text(encoding="utf-8") == "Status: Y\n"
    second = go()
    assert notes.read_text(encoding="utf-8") == "Status: Y\n", "second run must not corrupt"
    assert second["text"].strip()


# ------------------------------------------------- plan acceptance criterion 3


def test_path_escape_through_the_loop_creates_nothing(tmp_path: Path):
    root = tmp_path / "ws"
    root.mkdir()
    out = run_loop(
        spec_for(root, "allow"),
        "Write ../outside.txt",
        history=[],
        executor=make_executor(root, "allow"),
        complete_fn=scripted(
            [("fs_write", {"path": "../outside.txt", "content": "escaped"})],
            "I could not write outside the workspace.",
        ),
    )
    assert not (tmp_path / "outside.txt").exists()
    assert list(root.iterdir()) == []
    assert any(t.get("error") == "path_escape" for t in out["traces"] if t.get("kind") == "tool")


# ------------------------------------------------------------------- gating


def test_deny_mode_never_reaches_the_executor(tmp_path: Path):
    touched = []

    def executor(name, args):
        touched.append(name)
        return {"ok": True}

    out = run_loop(
        spec_for(tmp_path, "deny"),
        "Create hack.md",
        history=[],
        executor=executor,
        complete_fn=scripted(
            [("fs_write", {"path": "hack.md", "content": "x"})],
            "File editing is off.",
        ),
    )
    assert touched == [], "a denied write must not reach the executor"
    assert not (tmp_path / "hack.md").exists()
    assert any(
        t.get("error") == "tool_not_enabled" for t in out["traces"] if t.get("kind") == "tool"
    )


def test_ask_mode_surfaces_needs_confirm(tmp_path: Path):
    out = run_loop(
        spec_for(tmp_path, "ask"),
        "Create plan.md",
        history=[],
        executor=make_executor(tmp_path, "ask"),
        complete_fn=scripted(
            [("fs_write", {"path": "plan.md", "content": "# Plan\n"})],
            "Waiting for your approval.",
        ),
    )
    assert not (tmp_path / "plan.md").exists()
    assert len(out["needs_confirm"]) == 1
    assert out["needs_confirm"][0]["token"]
    assert "+# Plan" in out["needs_confirm"][0]["diff"]


# ---------------------------------------------------------- never-silent path


def test_empty_model_reply_still_answers_the_user(tmp_path: Path):
    (tmp_path / "x.pdf").write_bytes(b"%PDF-1.4")
    out = run_loop(
        spec_for(tmp_path, "deny"),
        "How many PDFs?",
        history=[],
        executor=make_executor(tmp_path, "deny"),
        complete_fn=scripted([("fs_list", {"glob": "*.pdf"})], ""),
    )
    assert out["text"].strip(), "the UI must never receive an empty reply"
    assert "x.pdf" in out["text"]


def test_invented_tool_name_gets_valid_names_back(tmp_path: Path):
    out = run_loop(
        spec_for(tmp_path, "deny"),
        "Read the readme",
        history=[],
        executor=make_executor(tmp_path, "deny"),
        complete_fn=scripted([("read_the_readme", {})], "Sorry, I used the wrong tool."),
    )
    tool_traces = [t for t in out["traces"] if t.get("kind") == "tool"]
    assert tool_traces and tool_traces[0]["error"] == "bad_tool"
    tool_msg = [m for m in out["messages"] if m.get("role") == "tool"][0]
    payload = json.loads(tool_msg["content"])
    assert "fs_read" in payload["valid_tools"]
