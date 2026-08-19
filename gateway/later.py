"""Post-v1 packs: stubs only. v1 must not grow these surfaces.

Later: ingest UI, Excel-write + structured field extraction, web_search,
long-term memory, SMTP mail, allowlisted MCP.
"""

from __future__ import annotations

LATER_PACKS = {
    "ingest_ui": {
        "status": "stub",
        "v1": "CLI --ingest (decision b)",
        "note": "UI screen cut from v1 to absorb hybrid retrieval's +8h.",
    },
    "excel_write": {
        "status": "stub",
        "v1": "denied",
        "note": "Needs structured field extraction before a sheet export.",
    },
    "field_extraction": {
        "status": "stub",
        "v1": "denied",
        "note": "vessel/port/dates/line items/totals — rabbit hole; ships with excel_write.",
    },
    "web_search": {"status": "stub", "v1": "denied", "note": "Open-web injection surface."},
    "long_term_memory": {
        "status": "stub",
        "v1": "denied",
        "note": "Session durability is SQLite short_term only. Nothing written into standing instructions.",
    },
    "smtp_mail": {"status": "stub", "v1": "denied", "note": "SMTP app-password only if ever; no Gmail OAuth."},
    "mcp": {
        "status": "stub",
        "v1": "denied",
        "note": "Allowlisted hashes only. No pasted stdio. We classify destructive, not the server author.",
    },
}


def later_status() -> dict:
    return {"v1_tools": ["rag_search", "fs_list", "fs_read"], "later": LATER_PACKS}
