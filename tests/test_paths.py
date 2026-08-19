from pathlib import Path

import pytest

from gateway.paths import PathEscapeError, resolve_workspace, safe_join


def test_safe_join_blocks_dotdot(tmp_path: Path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "ok.txt").write_text("x")
    assert safe_join(root, "ok.txt").name == "ok.txt"
    with pytest.raises(PathEscapeError):
        safe_join(root, "../secret.txt")
    with pytest.raises(PathEscapeError):
        safe_join(root, "..\\secret.txt")


def test_resolve_workspace(tmp_path: Path):
    d = tmp_path / "pdas"
    d.mkdir()
    assert resolve_workspace(d) == d.resolve()
