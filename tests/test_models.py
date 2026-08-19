from __future__ import annotations

import socket

from fastapi.testclient import TestClient

from gateway.app import make_app
from gateway.cli import first_free_port
from gateway.db import Store
from gateway.llmfit_bridge import fit_label, parse_json_blob, usable_recommend_row


def test_fit_label_against_free_ram() -> None:
    assert fit_label(2.5, 5.65) == "perfect"
    assert fit_label(4.0, 5.65) == "good"
    assert fit_label(1.0, 8.0) == "perfect"
    assert fit_label(12.0, 5.0) == "wont_fit"


def test_usable_recommend_skips_embeddings() -> None:
    assert not usable_recommend_row({"category": "Embedding", "params_b": 0.01, "name": "x-embed"})
    assert usable_recommend_row({"category": "Chat", "params_b": 4.0, "name": "Qwen3-4B", "runtime": "llama.cpp"})
    assert not usable_recommend_row({"category": "Chat", "params_b": 4.0, "name": "q", "runtime": "mlx"})


def test_parse_json_blob_skips_preamble() -> None:
    raw = "noise\n{ \"system\": { \"total_ram_gb\": 32 } }"
    assert parse_json_blob(raw)["system"]["total_ram_gb"] == 32


def test_first_free_port_skips_busy() -> None:
    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.bind(("127.0.0.1", 0))
    port = busy.getsockname()[1]
    try:
        got = first_free_port("127.0.0.1", port, span=8)
    finally:
        busy.close()
    assert got != port
    assert port < got <= port + 7


def test_pin_model_and_health(tmp_path) -> None:
    app = make_app(tmp_path, Store(tmp_path / "app.sqlite"))
    client = TestClient(app)
    r = client.post("/api/model", json={"provider": "ollama", "name": "qwen3:4b"})
    assert r.status_code == 200
    assert r.json()["model_pin"]["name"] == "qwen3:4b"
    h = client.get("/api/health").json()
    assert h["model"]["provider"] == "ollama"
    assert h["model"]["name"] == "qwen3:4b"
