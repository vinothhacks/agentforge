from pathlib import Path

from gateway.compiler import compile_pda_query
from gateway.evals import run_offline_golden, write_sample_pdas


def test_offline_golden(tmp_path: Path):
    write_sample_pdas(tmp_path, n=12)
    scored = run_offline_golden(tmp_path)
    assert scored["enumeration"]["recall"] >= 0.90
    assert scored["enumeration"]["cause"] is None
    assert "incomplete_enumeration" in scored["taxonomy"]


def test_compiler_denies_long_term(tmp_path: Path):
    spec = compile_pda_query(tmp_path)
    assert spec.memory.long_term is False
    assert spec.tools == ["rag_search", "fs_list", "fs_read"]
    assert spec.planner["enabled"] is False
