"""Stub llmfit binary: emits recorded M0-shaped JSON. No network. No TUI."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SYSTEM = {
    "system": {
        "available_ram_gb": 16.0,
        "backend": "CPU",
        "cpu_cores": 8,
        "cpu_name": "stub-cpu",
        "gpu_available_gb": None,
        "gpu_count": 0,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "gpus": [],
        "has_gpu": False,
        "total_ram_gb": 32.0,
        "unified_memory": False,
    }
}

FIT_OK = {
    "models": [
        {
            "name": "Qwen/Qwen2.5-3B-Instruct",
            "ollama_name": "qwen2.5:3b",
            "fit_level": "Good",
            "fit_label": "good",
            "best_quant": "Q4_K_M",
            "estimated_tps": 42.0,
            "measured_tps": None,
            "parameter_count": "3B",
            "disk_size_gb": 2.0,
            "memory_required_gb": 3.1,
            "effective_context_length": 8192,
            "capability_ids": ["tool_use"],
            "capabilities": ["tools"],
            "gguf_sources": ["Qwen2.5-3B-Instruct-Q4_K_M.gguf"],
            "runtime": "ollama",
        },
        {
            "name": "hf/tiny-tools",
            "ollama_name": "tinyllama",
            "fit_level": "Perfect",
            "best_quant": "Q4_0",
            "estimated_tps": 80.0,
            "measured_tps": None,
            "parameter_count": "1B",
            "disk_size_gb": 0.6,
            "memory_required_gb": 0.8,
            "effective_context_length": 2048,
            "capability_ids": ["tool_use"],
            "gguf_sources": ["tinyllama.gguf"],
            "runtime": "ollama",
        },
        {
            "name": "hf/sharded-70b",
            "ollama_name": "giant:70b",
            "fit_level": "Tight",
            "best_quant": "Q4_K_M",
            "estimated_tps": 5.0,
            "measured_tps": None,
            "parameter_count": "70B",
            "disk_size_gb": 40.0,
            "memory_required_gb": 48.0,
            "effective_context_length": 4096,
            "capability_ids": [],
            "gguf_sources": ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf"],
            "runtime": "llama.cpp",
        },
        {
            "name": "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B",
            "ollama_name": None,
            "fit_level": "Good",
            "best_quant": "Q8_0",
            "estimated_tps": 184.8,
            "measured_tps": None,
            "parameter_count": "1.7B",
            "disk_size_gb": 1.8,
            "memory_required_gb": 2.2,
            "effective_context_length": 4096,
            "capability_ids": [],
            "gguf_sources": ["voice.gguf"],
            "runtime": "llama.cpp",
        },
        {
            "name": "org/cached-local",
            "ollama_name": "cached-local:latest",
            "fit_level": "Good",
            "best_quant": "Q4_0",
            "estimated_tps": 30.0,
            "measured_tps": None,
            "parameter_count": "1B",
            "disk_size_gb": 0.9,
            "memory_required_gb": 1.0,
            "effective_context_length": 2048,
            "capability_ids": [],
            "gguf_sources": ["cached-local.gguf"],
            "runtime": "ollama",
        },
        {
            "name": "org/chat-only",
            "ollama_name": "chatonly:1b",
            "fit_level": "Perfect",
            "best_quant": "Q4_0",
            "estimated_tps": 60.0,
            "measured_tps": None,
            "parameter_count": "1B",
            "disk_size_gb": 0.7,
            "memory_required_gb": 0.9,
            "effective_context_length": 2048,
            "capability_ids": [],
            "capabilities": ["chat"],
            "gguf_sources": ["chatonly.gguf"],
            "runtime": "ollama",
        },
    ]
}

FIT_DRIFT = {
    "models": [
        {
            "name": "drift/no-ollama-field",
            "fit_level": "Good",
            "best_quant": "Q4_0",
            "estimated_tps": 10.0,
            "parameter_count": "1B",
        }
    ]
}


def main(argv: list[str]) -> int:
    log = os.environ.get("FAKE_LLMFIT_ARGV")
    if log:
        Path(log).write_text(json.dumps(argv), encoding="utf-8")
    mode = os.environ.get("FAKE_LLMFIT_MODE", "ok")
    args = [a for a in argv if a not in {"--json", "--no-dashboard"}]
    if "system" in args:
        print(json.dumps(SYSTEM))
        return 0
    if "fit" in args:
        payload = FIT_DRIFT if mode == "schema-drift" else FIT_OK
        print(json.dumps(payload))
        return 0
    if "list" in args:
        print(json.dumps([
            {
                "name": "Qwen/Qwen2.5-3B-Instruct",
                "provider": "Qwen",
                "parameter_count": "3B",
                "quantization": "Q4_K_M",
                "min_ram_gb": 3.1,
                "use_case": "chat",
                "capabilities": ["tools"],
                "gguf_sources": ["Qwen2.5-3B-Instruct-Q4_K_M.gguf"],
                "context_length": 8192,
            },
            {
                "name": "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B",
                "provider": "marksverdhei",
                "parameter_count": "1.7B",
                "quantization": "Q8_0",
                "min_ram_gb": 2.2,
                "use_case": "embedding",
                "capabilities": [],
                "gguf_sources": ["voice.gguf"],
                "context_length": 4096,
            },
        ]))
        return 0
    if "bench" in args:
        print(json.dumps({"tokens_per_second": 11.5, "measured_tps": 11.5, "tps": 11.5}))
        return 0
    print(json.dumps({"error": "unknown", "argv": argv}), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
