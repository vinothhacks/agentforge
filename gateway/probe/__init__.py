"""Probe suite: five steps that measure whether a model can *act* on this runtime+template.

1. Single tool call, valid args.
2. Omit required field → schema reject → does it recover?
3. Two-step: lookup then read.
4. Refuse-to-call: a question that needs no tool.
5. Timing: ttft / tok/s / working num_ctx.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from jsonschema import Draft202012Validator

from gateway.providers import ProviderError, complete


PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a shipping code. Use when the user asks for a code value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The code to look up, e.g. PDA-1"}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_item",
            "description": "Read an item by id returned from lookup.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
]


LOOKUP_SCHEMA = PROBE_TOOLS[0]["function"]["parameters"]
READ_SCHEMA = PROBE_TOOLS[1]["function"]["parameters"]


def _parse_args(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {}
    except json.JSONDecodeError:
        return {}


def _schema_ok(schema: dict[str, Any], args: dict[str, Any]) -> bool:
    v = Draft202012Validator(schema)
    return not any(v.iter_errors(args))


CompleteFn = Callable[..., dict[str, Any]]


def run_step(
    step: str,
    *,
    provider: str,
    name: str,
    num_ctx: int,
    api_key: str | None,
    complete_fn: CompleteFn = complete,
    timeout: float = 45.0,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    result: dict[str, Any] = {"step": step, "ok": False, "cause": None, "arg_valid": None}
    try:
        if step == "valid_call":
            resp = complete_fn(
                provider=provider,
                name=name,
                messages=[
                    {"role": "system", "content": "You are a tool-using assistant. Use lookup when asked for a code."},
                    {"role": "user", "content": "Look up code PDA-1 using the lookup tool."},
                ],
                tools=PROBE_TOOLS,
                num_ctx=num_ctx,
                api_key=api_key,
                timeout=timeout,
            )
            calls = resp["tool_calls"]
            if not calls:
                result["cause"] = "no_tool_call"
            else:
                args = _parse_args(calls[0]["arguments"])
                valid = calls[0]["name"] == "lookup" and _schema_ok(LOOKUP_SCHEMA, args)
                result["arg_valid"] = valid
                result["ok"] = valid
                if not valid:
                    result["cause"] = "schema_fail" if calls[0]["name"] == "lookup" else "bad_tool"
            result["ttft_s"] = resp["elapsed_s"]
            result["completion_tokens"] = resp["completion_tokens"]

        elif step == "recover_missing_arg":
            # Model is asked to call lookup but we will reject missing `code` if it does.
            resp = complete_fn(
                provider=provider,
                name=name,
                messages=[
                    {"role": "system", "content": "Use the lookup tool. Always include the code argument."},
                    {"role": "user", "content": "Look up the default PDA code."},
                ],
                tools=PROBE_TOOLS,
                num_ctx=num_ctx,
                api_key=api_key,
                timeout=timeout,
            )
            calls = resp["tool_calls"]
            if not calls:
                result["cause"] = "no_tool_call"
            else:
                args = _parse_args(calls[0]["arguments"])
                valid = _schema_ok(LOOKUP_SCHEMA, args)
                result["arg_valid"] = valid
                if valid:
                    result["ok"] = True
                else:
                    # Feed the schema error back; recovery = a second valid call.
                    retry = complete_fn(
                        provider=provider,
                        name=name,
                        messages=[
                            {"role": "system", "content": "Use the lookup tool."},
                            {"role": "user", "content": "Look up the default PDA code."},
                            {
                                "role": "assistant",
                                "content": resp["content"],
                                "tool_calls": [
                                    {
                                        "id": calls[0]["id"] or "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": calls[0]["name"],
                                            "arguments": calls[0]["arguments"],
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": calls[0]["id"] or "call_1",
                                "content": json.dumps(
                                    {"error": "schema_reject", "missing": "code", "hint": "call lookup with code=PDA-1"}
                                ),
                            },
                        ],
                        tools=PROBE_TOOLS,
                        num_ctx=num_ctx,
                        api_key=api_key,
                        timeout=timeout,
                    )
                    rcalls = retry["tool_calls"]
                    if rcalls and _schema_ok(LOOKUP_SCHEMA, _parse_args(rcalls[0]["arguments"])):
                        result["ok"] = True
                        result["arg_valid"] = True
                    else:
                        result["ok"] = False
                        result["cause"] = "schema_fail"

        elif step == "two_step":
            first = complete_fn(
                provider=provider,
                name=name,
                messages=[
                    {
                        "role": "system",
                        "content": "First call lookup with code PDA-1, then call read_item with id=item-1.",
                    },
                    {"role": "user", "content": "Find PDA-1 and then read the item."},
                ],
                tools=PROBE_TOOLS,
                num_ctx=num_ctx,
                api_key=api_key,
                timeout=timeout,
            )
            calls = first["tool_calls"]
            if not calls or calls[0]["name"] != "lookup":
                result["cause"] = "no_tool_call"
            else:
                args = _parse_args(calls[0]["arguments"])
                result["arg_valid"] = _schema_ok(LOOKUP_SCHEMA, args)
                second = complete_fn(
                    provider=provider,
                    name=name,
                    messages=[
                        {"role": "system", "content": "You already looked up PDA-1. Now call read_item with id item-1."},
                        {"role": "user", "content": "Now read the item."},
                        {
                            "role": "assistant",
                            "content": first["content"],
                            "tool_calls": [
                                {
                                    "id": calls[0]["id"] or "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": calls[0]["arguments"]},
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": calls[0]["id"] or "call_1",
                            "content": json.dumps({"id": "item-1", "ok": True}),
                        },
                    ],
                    tools=PROBE_TOOLS,
                    num_ctx=num_ctx,
                    api_key=api_key,
                    timeout=timeout,
                )
                scalls = second["tool_calls"]
                if scalls and scalls[0]["name"] == "read_item" and _schema_ok(READ_SCHEMA, _parse_args(scalls[0]["arguments"])):
                    result["ok"] = True
                    result["arg_valid"] = True
                else:
                    result["cause"] = "no_tool_call" if not scalls else "schema_fail"

        elif step == "no_tool":
            resp = complete_fn(
                provider=provider,
                name=name,
                messages=[
                    {"role": "system", "content": "Only use tools when needed. Answer greetings in plain text."},
                    {"role": "user", "content": "Say hello. Do not use any tool."},
                ],
                tools=PROBE_TOOLS,
                num_ctx=num_ctx,
                api_key=api_key,
                timeout=timeout,
            )
            if resp["tool_calls"]:
                result["cause"] = "bad_tool"
                result["ok"] = False
            elif (resp["content"] or "").strip():
                result["ok"] = True
            else:
                result["cause"] = "no_tool_call"

        elif step == "timing":
            resp = complete_fn(
                provider=provider,
                name=name,
                messages=[{"role": "user", "content": "Reply with the single word: pong"}],
                tools=None,
                num_ctx=num_ctx,
                api_key=api_key,
                timeout=timeout,
            )
            elapsed = resp["elapsed_s"] or 1e-6
            toks = max(1, resp["completion_tokens"] or 1)
            result["ok"] = bool((resp["content"] or "").strip())
            result["ttft_s"] = elapsed
            result["tps"] = toks / elapsed
            result["completion_tokens"] = resp["completion_tokens"]
            if not result["ok"]:
                result["cause"] = "timeout"
        else:
            result["cause"] = "bad_tool"
    except ProviderError as exc:
        msg = str(exc).lower()
        if "timeout" in msg:
            result["cause"] = "probe_timeout"
        elif "context" in msg or "num_ctx" in msg:
            result["cause"] = "ctx_overflow"
        elif "memory" in msg or "oom" in msg:
            result["cause"] = "oom"
        elif "template" in msg:
            result["cause"] = "template_mismatch"
        else:
            result["cause"] = "timeout"
        result["error"] = str(exc)
    result["elapsed_s"] = time.perf_counter() - t0
    return result


FULL_STEPS = ["valid_call", "recover_missing_arg", "two_step", "no_tool", "timing"]
FAST_STEPS = ["valid_call", "no_tool"]
