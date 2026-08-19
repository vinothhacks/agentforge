"""Later packs — not in v1.

ingest UI, Excel-write, structured field extraction, web_search,
long-term memory, SMTP mail, allowlisted MCP.

These are extension points. `GET /api/later` returns their stub status.
Do not implement them in v1. Shell, heartbeat, channels, skills marketplace
are rejected in the threat model, not deferred.
"""
