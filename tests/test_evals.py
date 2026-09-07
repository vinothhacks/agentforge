from pathlib import Path

from gateway.compiler import compile_workspace
from gateway.evals import run_offline_golden, write_sample_docs
from gateway.runtime.tools import ALL_TOOLS, tools_for


def test_offline_golden(tmp_path: Path):
    write_sample_docs(tmp_path, n=12)
    scored = run_offline_golden(tmp_path)
    assert scored["enumeration"]["recall"] >= 0.90
    assert scored["enumeration"]["cause"] is None
    assert "incomplete_enumeration" in scored["taxonomy"]


def test_compiler_denies_long_term(tmp_path: Path):
    spec = compile_workspace(tmp_path)
    assert spec.memory.long_term is False
    # The compiler keeps the full roster; permissions.fs_write is what gates
    # writing. Hardcoding the read set here used to strip the write tools on
    # every CLI start, so the sidebar toggle did nothing.
    assert spec.tools == ALL_TOOLS
    assert spec.tools[:3] == ["rag_search", "fs_list", "fs_read"]
    assert spec.planner["enabled"] is False


def test_compiler_keeps_writes_gated_by_permission(tmp_path: Path):
    """Advertising the tools is not the same as allowing them to run."""
    spec = compile_workspace(tmp_path)
    spec.permissions.fs_write = "deny"
    assert tools_for(spec.tools, spec.permissions.fs_write) == ["rag_search", "fs_list", "fs_read"]
    assert tools_for(spec.tools, "allow") == ALL_TOOLS
    assert spec.permissions.shell == "deny"
    assert spec.permissions.send_email == "deny"
