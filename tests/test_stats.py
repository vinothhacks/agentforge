import math

from gateway.stats import max_steps_from_lo95, wilson_interval


def test_wilson_bounds():
    lo, hi = wilson_interval(5, 5)
    assert lo > 0.4
    assert hi == 1.0 or hi > 0.9
    lo0, hi0 = wilson_interval(0, 5)
    assert lo0 == 0.0
    assert hi0 < 0.6


def test_max_steps_uses_lower_bound_not_mean():
    # mean 0.93 with n=5 is not this function — we pass lo95=0.72
    steps = max_steps_from_lo95(0.72)
    assert 3 <= steps <= 8
    # 0.93^12 ≈ 0.42 would have been the old lie; lo95 path stays small
    assert max_steps_from_lo95(0.93) <= 10
    assert max_steps_from_lo95(0.0) == 1
    # n=3 all-pass lo95 ≈ 0.44 must still budget a tool + cite + answer turn
    assert max_steps_from_lo95(0.44) >= 3
