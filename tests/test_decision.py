"""Decision layer: alerts -> planned removals or failures -> costs, against hand-worked engines."""
import numpy as np
import pandas as pd
import pytest

from src.decision import (
    FAILURE,
    PLANNED,
    Costs,
    alert_age,
    alert_lower,
    alert_oracle,
    alert_point,
    break_even,
    cost_matched_tau,
    cost_summary,
    level_for_tau,
    simulate,
    sweep,
    true_rul,
    tune,
)


def engines(lifetimes: dict[int, int], bias: dict[int, float] | None = None) -> pd.DataFrame:
    """Complete engines with a prediction = true RUL + a constant bias per engine."""
    bias = bias or {}
    parts = []
    for unit, n in lifetimes.items():
        cycle = np.arange(1, n + 1)
        rul = n - cycle
        parts.append(pd.DataFrame({"unit": unit, "cycle": cycle, "pred": rul + bias.get(unit, 0.0)}))
    return pd.concat(parts, ignore_index=True)


def test_true_rul_counts_down_to_zero():
    f = engines({1: 5})
    assert true_rul(f).tolist() == [4, 3, 2, 1, 0]


def test_oracle_removes_every_engine_exactly_at_failure():
    f = engines({1: 100, 2: 150})
    out = simulate(f, alert_oracle(f, horizon=10), horizon=10)
    assert (out["outcome"] == PLANNED).all()
    assert out["wasted_cycles"].tolist() == [0.0, 0.0]
    assert out["cycles_flown"].tolist() == [100.0, 150.0]


def test_hand_worked_engine():
    # T = 100, horizon 10. Prediction 5 cycles too optimistic: pred = RUL + 5.
    # First alert when RUL + 5 <= 10  ->  RUL = 5  ->  cycle 95; removal booked at 105 > 100 -> FAILURE.
    f = engines({1: 100}, bias={1: 5.0})
    out = simulate(f, alert_point(f, horizon=10), horizon=10).iloc[0]
    assert out["alert_cycle"] == 95 and out["outcome"] == FAILURE and out["wasted_cycles"] == 0
    # A safety margin of 5 cycles fixes it: pred <= 15 -> RUL = 10 -> alert at 90, removal at 100 = just in time.
    out = simulate(f, alert_point(f, horizon=10, margin=5), horizon=10).iloc[0]
    assert out["alert_cycle"] == 90 and out["removal_cycle"] == 100 and out["outcome"] == PLANNED


def test_pessimistic_prediction_wastes_life():
    # pred = RUL - 20: alert when RUL - 20 <= 10 -> RUL = 30 -> cycle 70, removal 80 -> 20 cycles wasted.
    f = engines({1: 100}, bias={1: -20.0})
    out = simulate(f, alert_point(f, horizon=10), horizon=10).iloc[0]
    assert out["outcome"] == PLANNED and out["removal_cycle"] == 80 and out["wasted_cycles"] == 20


def test_no_alert_means_failure():
    f = engines({1: 50}, bias={1: 1000.0})
    out = simulate(f, alert_point(f, horizon=10), horizon=10).iloc[0]
    assert out["outcome"] == FAILURE and np.isnan(out["alert_cycle"])


def test_check_every_delays_the_alert():
    f = engines({1: 100})
    out = simulate(f, alert_oracle(f, horizon=10), horizon=10, check_every=7).iloc[0]
    # oracle wants cycle 90; first check cycle at or after 90 is 91 -> removal 101 > 100 -> failure
    assert out["alert_cycle"] == 91 and out["outcome"] == FAILURE


def test_age_rule_replaces_at_the_chosen_age():
    f = engines({1: 100, 2: 200})
    out = simulate(f, alert_age(f, horizon=10, replace_age=120), horizon=10)
    assert out.loc[1, "outcome"] == FAILURE                 # fails at 100 before the age-120 removal
    assert out.loc[2, "removal_cycle"] == 120 and out.loc[2, "wasted_cycles"] == 80


def test_cost_summary_cost_rate():
    f = engines({1: 100, 2: 100}, bias={1: 5.0, 2: -20.0})   # engine 1 fails, engine 2 removed at 80
    out = simulate(f, alert_point(f, horizon=10), horizon=10)
    s = cost_summary(out, Costs(planned=1, failure=10))
    assert (s["failures"], s["planned"]) == (1, 1)
    assert s["total_cost"] == 11
    assert s["cost_per_1000_cycles"] == pytest.approx(1000 * 11 / (100 + 80))


def test_lower_bound_rule_equals_point_plus_constant_margin():
    # A constant-width interval (split conformal) is just a point prediction with a safety margin.
    f = engines({1: 120, 2: 180}, bias={1: 4.0, 2: -3.0})
    f["lower"] = f["pred"] - 7.0
    a = simulate(f, alert_lower(f, horizon=10), horizon=10)
    b = simulate(f, alert_point(f, horizon=10, margin=7.0), horizon=10)
    pd.testing.assert_frame_equal(a, b)


def test_tune_finds_the_margin_that_avoids_failures():
    f = engines({1: 100, 2: 130}, bias={1: 6.0, 2: 6.0})     # every prediction 6 cycles too optimistic
    best, table = tune(f, lambda m: alert_point(f, 10, margin=m), range(0, 21), horizon=10, costs=Costs(1, 50))
    assert best == 6                                          # smallest margin with no failure, least waste
    assert table.loc[table["param"] == best, "failures"].item() == 0


def test_sweep_and_break_even():
    # Rule "risky" wastes no life but lets one engine of two fail; rule "safe" never fails but removes
    # the engines at cycles 75 and 70. Cheap failures favour "risky", expensive failures favour "safe".
    f = engines({1: 100, 2: 100}, bias={1: 5.0, 2: 0.0})
    risky = simulate(f, alert_point(f, horizon=10), horizon=10)
    safe = simulate(f, alert_point(f, horizon=10, margin=30), horizon=10)
    table = sweep({"risky": lambda c: risky, "safe": lambda c: safe}, ratios=[1, 2, 5, 10, 20])
    be = break_even(table, baseline="risky", challenger="safe")
    # risky: (r + 1) / 200 per cycle; safe: 2 / 145 per cycle -> equal at r = 400/145 - 1 = 1.76
    assert 1 < be < 2
    assert np.isnan(break_even(table, baseline="safe", challenger="safe"))


def test_cost_matched_quantile():
    assert cost_matched_tau(9) == pytest.approx(0.10)         # failure 9x planned -> act on the 10 % quantile
    assert cost_matched_tau(19) == pytest.approx(0.05)
    assert level_for_tau(0.05) == pytest.approx(0.90)          # the lower end of a central 90 % interval


def test_incomplete_engine_is_rejected():
    f = engines({1: 10}).iloc[2:]                              # cycles 3..10: not a complete engine
    with pytest.raises(AssertionError):
        simulate(f, np.zeros(len(f), bool), horizon=2)
