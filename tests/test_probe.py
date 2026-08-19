"""Fake provider that honours the probe suite for some tags and not others."""

import time

import pytest

from gateway.probe.experiment import P1_TAGS, run_experiment
from gateway.probe.ladder import fail_forward
from gateway.probe.runner import run_probe
from gateway.providers import ProviderError, complete


class FakeLLM:
    def __init__(self, mode: str):
        self.mode = mode
        self.n = 0

    def __call__(self, *, messages, tools=None, **kwargs):
        self.n += 1
        user = ""
        for m in messages:
            if m.get("role") == "user":
                user = m.get("content") or ""
        if self.mode == "tools":
            if tools is None or "hello" in user.lower() or "pong" in user.lower() or "Reply with" in user:
                return {
                    "content": "pong" if "pong" in user.lower() or "Reply" in user else "hello",
                    "tool_calls": [],
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "elapsed_s": 0.01,
                    "finish_reason": "stop",
                }
            name = "lookup"
            args = '{"code":"CODE-1"}'
            roles = [m.get("role") for m in messages]
            if "tool" in roles or any("item-1" in str(m.get("content", "")) for m in messages if m.get("role") == "tool"):
                name = "read_item"
                args = '{"id":"item-1"}'
            return {
                "content": "",
                "tool_calls": [{"id": "c1", "name": name, "arguments": args}],
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "elapsed_s": 0.02,
                "finish_reason": "tool_calls",
            }
        return {
            "content": "I cannot use tools.",
            "tool_calls": [],
            "prompt_tokens": 8,
            "completion_tokens": 6,
            "elapsed_s": 0.01,
            "finish_reason": "stop",
        }


def test_probe_tools_pass():
    card = run_probe(
        provider="openrouter",
        name="fake-good",
        mode="full",
        n=3,
        complete_fn=FakeLLM("tools"),
    )
    assert card["verdict"] == "agent"
    assert card["measured"]["n"] == 3
    assert card["measured"]["per_step_success_lo95"] <= card["measured"]["per_step_success_mean"]
    assert card["measured"]["max_steps"] >= 1


def test_probe_chat_only():
    card = run_probe(
        provider="ollama",
        name="fake-bad",
        mode="fast",
        n=3,
        complete_fn=FakeLLM("chat"),
    )
    assert card["verdict"] == "chat_only"
    assert card["cause"] in {"no_tool_call", "bad_tool"}


def test_fail_forward_reaches_cloud():
    def probe(**kwargs):
        if kwargs["provider"] == "openrouter":
            return run_probe(complete_fn=FakeLLM("tools"), **{k: v for k, v in kwargs.items() if k != "complete_fn"})
        return run_probe(complete_fn=FakeLLM("chat"), **{k: v for k, v in kwargs.items() if k != "complete_fn"})

    out = fail_forward(
        provider="ollama",
        name="weak",
        alternate_tags=[("openrouter", "openai/gpt-4o-mini")],
        mode="fast",
        n=3,
        probe_fn=probe,
    )
    assert out["ok"] is True
    assert out["card"]["model_pin"]["provider"] == "openrouter"


def test_p1_tag_count():
    assert len(P1_TAGS) == 25


def test_p1_disagreement_number():
    catalog = {
        "models": [
            {"name": "qwen3:4b", "ollama_name": "qwen3:4b", "capability_ids": ["tool_use"]},
            {"name": "llama3.1:8b", "ollama_name": "llama3.1:8b", "capability_ids": ["tool_use"]},
            {"name": "tinyllama", "ollama_name": "tinyllama", "capability_ids": []},
        ]
    }

    def complete_fn(*, name, messages, tools=None, **kwargs):
        mode = "tools" if "qwen3" in name else "chat"
        return FakeLLM(mode)(messages=messages, tools=tools, name=name, **kwargs)

    result = run_experiment(
        n=5,
        mode="fast",
        complete_fn=complete_fn,
        llmfit_json=catalog,
        only_installed=False,
        api_key="test",
    )
    assert len(result["rows"]) == 25
    yes = [r for r in result["rows"] if r["catalog"] == "yes" and r["n"] == 5]
    assert len(yes) == 2
    fails = sum(1 for r in yes if r["verdict"] != "agent")
    assert result["D"] == fails / len(yes)
    assert result["D"] == 0.5
    assert result["gate"] == "product"


def test_complete_hard_timeout(monkeypatch):
    def hang(**kwargs):
        time.sleep(8)
        raise AssertionError("LiteLLM should not be waited on")

    monkeypatch.setattr("gateway.providers.litellm.completion", hang)
    t0 = time.perf_counter()
    with pytest.raises(ProviderError, match="timeout"):
        complete(
            provider="ollama",
            name="fake-hang",
            messages=[{"role": "user", "content": "hi"}],
            timeout=0.3,
        )
    assert time.perf_counter() - t0 < 2.0
