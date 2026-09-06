"""Wilson score interval and lower-bound budget math.

max_steps / max_tools are derived from the *lower* 95% Wilson bound of
per_step_success — never from the mean.
"""

from __future__ import annotations

import math


Z95 = 1.96
MIN_SURVIVAL = 0.5

# Budget math to use when a pin was never probed. Deliberately below the 0.40
# `agent` bar so an unmeasured pin is never mistaken for a measured one: it
# still gets the 3 steps v1 needs, but nothing reports it as evidence.
UNMEASURED_LO95 = 0.40


def effective_lo95(value: object) -> float:
    """Coerce a possibly-missing per_step_success_lo95 to a usable float.

    `None` means "never probed" and maps to UNMEASURED_LO95 rather than 0.0,
    which would collapse max_steps to 1.
    """
    if value is None:
        return UNMEASURED_LO95
    try:
        return float(value)
    except (TypeError, ValueError):
        return UNMEASURED_LO95


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + z2 / (4 * n)) / n)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


def max_steps_from_lo95(lo95: float, measured_tps: float | None = None, wall_s: float = 90.0) -> int:
    """Derive max_steps from lower CI bound and optional latency cap.

    An `agent` pin (lo95 >= 0.40) always gets at least 3 steps: tool, cite, answer.
    n=3 all-pass Wilson lo95 is ~0.44; log(0.5)/log(0.44) would otherwise collapse to 1.
    """
    if lo95 <= 0.0:
        raw = 1
    elif lo95 >= 0.999:
        raw = 8
    else:
        raw = int(math.floor(math.log(MIN_SURVIVAL) / math.log(lo95)))
    raw = max(1, min(raw, 8))
    if lo95 >= 0.40:
        raw = max(raw, 3)
    if measured_tps is not None and measured_tps > 0:
        # Rough: 400 tokens/step * steps must fit in wall_s
        latency_cap = max(1, int(wall_s * measured_tps / 400.0))
        raw = min(raw, latency_cap)
        if lo95 >= 0.40:
            raw = max(raw, 3)
    return raw


def max_tools_from_lo95(lo95: float, v1_need: int = 3) -> int:
    """If the pin cannot reliably handle v1's 3 tools, report below need.

    n=3 all-pass Wilson lo95 is ~0.44, so the v1 bar is 0.40 not 0.70.
    """
    if lo95 >= 0.40:
        return max(v1_need, 3)
    if lo95 >= 0.25:
        return 2
    return 1
