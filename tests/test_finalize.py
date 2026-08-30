"""Phase 0: the chat must never be silent and must never emit raw JSON.

These cover the four Phase 0 findings that produced empty or unusable replies.
"""

from __future__ import annotations

from gateway.runtime.loop import (
    finalize,
    list_args_for,
    looks_like_json,
    summarize_results,
    wants_file_list,
)


def test_looks_like_json():
    assert looks_like_json('{"path": "a.md"}')
    assert looks_like_json('[{"a": 1}]')
    assert not looks_like_json("This folder has 3 PDFs.")
    assert not looks_like_json("")
    assert not looks_like_json("{not really json")


def test_empty_reply_is_replaced_by_rendered_tool_output():
    results = [
        {
            "tool": "fs_list",
            "result": {
                "count": 2,
                "glob": "*.pdf",
                "entries": [
                    {"path": "a.pdf", "type": "file"},
                    {"path": "b.pdf", "type": "file"},
                ],
            },
        }
    ]
    text = finalize("", results, None)
    assert text
    assert "a.pdf" in text and "b.pdf" in text
    assert not looks_like_json(text)


def test_raw_json_reply_is_replaced():
    """Finding #1: fs_list worked but the model echoed the JSON back."""
    raw = '{"path": ".", "entries": [{"name": "a.pdf"}]}'
    results = [
        {"tool": "fs_list", "result": {"count": 1, "entries": [{"path": "a.pdf", "type": "file"}]}}
    ]
    text = finalize(raw, results, None)
    assert text != raw
    assert "a.pdf" in text


def test_json_without_tools_is_never_shown():
    """1B models dump extension maps instead of calling fs_list."""
    raw = '{"\\.md": "Vinoth_N_GenAI_Engineer_ATS_Resu", ".txt": "Vinoth_N_Designed_Resume"}'.replace("\\", "")
    text = finalize(raw, [], None)
    assert text != raw
    assert not looks_like_json(text)
    assert "readable" in text.lower()


def test_wants_file_list():
    assert wants_file_list("What resume files are in this folder?")
    assert wants_file_list("List every PDF in this folder.")
    assert not wants_file_list("Summarize Vinoth's GenAI experience.")
    assert list_args_for("List every PDF") == {"path": ".", "glob": "*.pdf"}
    assert list_args_for("List every DOCX") == {"path": ".", "glob": "*.docx"}


def test_good_prose_reply_is_left_alone():
    text = finalize("There are 3 resume PDFs in this folder.", [], None)
    assert text == "There are 3 resume PDFs in this folder."


def test_timeout_gets_an_actionable_message():
    text = finalize("", [], "timeout")
    assert "ran out of time" in text.lower()
    assert text.strip()


def test_budget_stop_gets_a_message():
    assert "budget" in finalize("", [], "budget_stop").lower()


def test_total_failure_still_returns_something():
    text = finalize("", [], None)
    assert text.strip()
    assert "could not produce a readable answer" in text.lower()


def test_tool_error_is_explained_in_english():
    results = [{"tool": "fs_read", "result": {"error": "wrong_file", "detail": "file does not exist"}}]
    text = finalize("", results, None)
    assert "file does not exist" in text


def test_summarize_write_results():
    created = summarize_results(
        [{"tool": "fs_write", "result": {"ok": True, "created": True, "path": "summary.md", "bytes": 42}}]
    )
    assert "Created" in created and "summary.md" in created

    idempotent = summarize_results(
        [{"tool": "fs_edit", "result": {"ok": True, "already_applied": True, "path": "a.md"}}]
    )
    assert "already contained" in idempotent


def test_summarize_needs_confirm():
    text = summarize_results(
        [{"tool": "fs_write", "result": {"needs_confirm": True, "action": "write", "path": "s.md"}}]
    )
    assert "approval" in text.lower() and "s.md" in text


def test_summarize_rag_paths():
    text = summarize_results(
        [{"tool": "rag_search", "result": {"paths": ["r1.pdf", "r2.pdf"], "hits": []}}]
    )
    assert "r1.pdf" in text and "2 files" in text
