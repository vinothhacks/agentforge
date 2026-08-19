from gateway.runtime.tools import dispatch
from gateway.runtime.validate import SchemaReject, validate_args
from gateway.runtime.tools import TOOL_SCHEMAS


def test_schema_reject_before_execute():
    called = []

    def exec_(name, args):
        called.append(name)
        return {"ok": True}

    out = dispatch("rag_search", {}, exec_)
    assert out["error"] == "schema_reject"
    assert called == []

    out = dispatch("rag_search", {"query": "demurrage"}, exec_)
    assert out["ok"] is True
    assert called == ["rag_search"]


def test_unknown_tool():
    out = dispatch("excel_write", {"path": "x.xlsx"}, lambda n, a: {})
    assert out["error"] == "bad_tool"


def test_validate_args_direct():
    try:
        validate_args(TOOL_SCHEMAS["fs_read"], {}, "fs_read")
        assert False
    except SchemaReject as e:
        assert e.tool == "fs_read"
