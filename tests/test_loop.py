from fastapi.testclient import TestClient

from gateway.app import make_app
from gateway.db import Store
from gateway.later import later_status
from gateway.runtime.loop import run_loop
from gateway.spec import AgentSpec


def test_later_not_in_v1_tools():
    s = later_status()
    assert s["v1_read_tools"] == ["rag_search", "fs_list", "fs_read"]
    assert s["v1_write_tools"] == ["fs_write", "fs_edit", "fs_mkdir"]
    assert s["v1_tools"] == s["v1_read_tools"] + s["v1_write_tools"]
    # Post-v1 surfaces must not have leaked into the tool roster.
    assert "excel_write" not in s["v1_tools"]
    assert "web_search" not in s["v1_tools"]
    assert "excel_write" in s["later"]
    assert s["later"]["mcp"]["v1"] == "denied"


def test_loop_with_fake(tmp_path):
    spec = AgentSpec(workspace_root=str(tmp_path))
    spec.model_pin.provider = "openrouter"
    spec.model_pin.name = "fake"

    def complete_fn(**kwargs):
        return {
            "content": "NEPTUNE account total is USD 15000 (cite DOC-001).",
            "tool_calls": [],
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "elapsed_s": 0.01,
            "finish_reason": "stop",
        }

    out = run_loop(
        spec,
        "What is the account total in file X?",
        history=[],
        executor=lambda n, a: {},
        complete_fn=complete_fn,
    )
    assert "15000" in out["text"]
    assert out["usage"]["turn_tokens"] > 0


def test_loop_executes_tool_calls(tmp_path):
    spec = AgentSpec(workspace_root=str(tmp_path))
    spec.budgets.max_steps = 3
    calls = {"n": 0}

    def complete_fn(*, messages, tools=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "rag_search", "arguments": '{"query":"demurrage"}'}
                ],
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "elapsed_s": 0.01,
                "finish_reason": "tool_calls",
            }
        return {
            "content": "Demurrage in DOC-001-NEPTUNE.txt",
            "tool_calls": [],
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "elapsed_s": 0.01,
            "finish_reason": "stop",
        }

    executed = []

    def executor(name, args):
        executed.append((name, args))
        return {"paths": ["DOC-001-NEPTUNE.txt"], "hits": []}

    out = run_loop(
        spec,
        "List files mentioning demurrage",
        history=[],
        executor=executor,
        complete_fn=complete_fn,
    )
    assert executed == [("rag_search", {"query": "demurrage"})]
    assert "DOC-001" in out["text"]
    assert any(t.get("kind") == "tool" for t in out["traces"])


def test_sanitize_drops_dangling_tool_calls():
    from gateway.runtime.loop import sanitize_messages

    msgs = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "rag_search"}}],
        },
        {"role": "user", "content": "q2"},
    ]
    clean = sanitize_messages(msgs)
    assert [m["role"] for m in clean] == ["user", "user"]


def test_later_pack_returns_501(tmp_path):
    app = make_app(tmp_path, Store(tmp_path / "app.sqlite"))
    client = TestClient(app)
    listed = client.get("/api/later")
    assert listed.status_code == 200
    assert listed.json()["later"]["excel_write"]["status"] == "stub"
    denied = client.get("/api/later/excel_write")
    assert denied.status_code == 501
    missing = client.get("/api/later/not_a_pack")
    assert missing.status_code == 404
