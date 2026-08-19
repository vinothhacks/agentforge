"""Probe runner: n repeats, time budget, Wilson lo95, card construction."""

from __future__ import annotations

import time
from typing import Any

from gateway.probe import FAST_STEPS, FULL_STEPS, run_step
from gateway.stats import max_steps_from_lo95, max_tools_from_lo95, wilson_interval

FAST_BUDGET_S = 90.0
FULL_BUDGET_S = 8 * 60.0
STEP_TIMEOUT_S = 90.0


def run_probe(
    *,
    provider: str,
    name: str,
    num_ctx: int = 8192,
    mode: str = "full",
    n: int = 3,
    api_key: str | None = None,
    complete_fn=None,
    quant: str | None = None,
    budget_s: float | None = None,
) -> dict[str, Any]:
    steps = FAST_STEPS if mode == "fast" else FULL_STEPS
    budget = budget_s if budget_s is not None else (FAST_BUDGET_S if mode == "fast" else FULL_BUDGET_S)
    deadline = time.perf_counter() + budget
    trials: list[dict[str, Any]] = []
    cause = None
    actual_mode = mode

    kwargs: dict[str, Any] = {
        "provider": provider,
        "name": name,
        "num_ctx": num_ctx,
        "api_key": api_key,
    }
    if complete_fn is not None:
        kwargs["complete_fn"] = complete_fn

    t_start = time.perf_counter()
    for i in range(n):
        if time.perf_counter() > deadline:
            cause = "probe_timeout"
            if actual_mode == "full":
                # Downgrade to fast subset for remaining budget.
                actual_mode = "fast"
                steps = FAST_STEPS
                deadline = time.perf_counter() + FAST_BUDGET_S
            else:
                break
        trial_steps = []
        for step in steps:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                cause = "probe_timeout"
                break
            trial_steps.append(run_step(step, timeout=min(STEP_TIMEOUT_S, remaining), **kwargs))
        trials.append({"repeat": i + 1, "steps": trial_steps})

    elapsed = time.perf_counter() - t_start
    # Score: a trial is a success if every scored step (except pure timing) is ok.
    scored = []
    arg_ok = 0
    arg_n = 0
    tps_samples: list[float] = []
    for trial in trials:
        step_ok = []
        for s in trial["steps"]:
            if s["step"] == "timing":
                if s.get("tps"):
                    tps_samples.append(float(s["tps"]))
                continue
            step_ok.append(bool(s.get("ok")))
            if s.get("arg_valid") is not None:
                arg_n += 1
                if s["arg_valid"]:
                    arg_ok += 1
        if step_ok:
            scored.append(all(step_ok))
        last_cause = next((s.get("cause") for s in reversed(trial["steps"]) if s.get("cause")), None)
        if last_cause and cause is None:
            cause = last_cause

    successes = sum(1 for x in scored if x)
    n_scored = len(scored)
    lo, hi = wilson_interval(successes, n_scored) if n_scored else (0.0, 1.0)
    mean = (successes / n_scored) if n_scored else 0.0
    arg_lo, arg_hi = wilson_interval(arg_ok, arg_n) if arg_n else (0.0, 1.0)
    tps = sum(tps_samples) / len(tps_samples) if tps_samples else None
    max_steps = max_steps_from_lo95(lo, measured_tps=tps)
    max_tools = max_tools_from_lo95(lo)
    verdict = "agent" if lo >= 0.40 and max_tools >= 3 else "chat_only"
    if n_scored == 0:
        verdict = "chat_only"
        cause = cause or "probe_timeout"

    digest = f"{provider}:{name}"
    card = {
        "model_pin": {
            "provider": provider,
            "name": name,
            "digest": digest,
            "quant": quant,
            "num_ctx": num_ctx,
        },
        "measured": {
            "n": n_scored,
            "successes": successes,
            "per_step_success_mean": round(mean, 4),
            "per_step_success_lo95": round(lo, 4),
            "per_step_success_hi95": round(hi, 4),
            "arg_valid_rate_lo95": round(arg_lo, 4),
            "arg_valid_rate_hi95": round(arg_hi, 4),
            "max_tools": max_tools,
            "max_steps": max_steps,
            "tps": tps,
            "probe_mode": actual_mode,
            "elapsed_s": round(elapsed, 2),
        },
        "verdict": verdict,
        "cause": None if verdict == "agent" else (cause or "no_tool_call"),
        "trials": trials,
    }
    return card
