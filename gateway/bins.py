"""Resolve uv-tool shims even when PATH is thin (Windows uvicorn)."""

from __future__ import annotations

import shutil
from pathlib import Path


def which_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    homes = [
        Path.home() / ".local" / "bin",
        Path.home() / "AppData" / "Roaming" / "Python" / "Scripts",
    ]
    for folder in homes:
        for cand in (folder / name, folder / f"{name}.exe"):
            if cand.is_file():
                return str(cand)
    return None
