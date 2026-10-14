"""Summaries of the decision results: euros, break-even per seed, value captured, win rates."""
import numpy as np
import pandas as pd
import pytest

from src.summarize import add_value_captured, break_even_by_seed, break_even_summary, curve_costs, euros, win_rate


def fake_decisions():
    """Two seeds, one model. 'safe' wastes life but never fails; 'risky' fails one engine in ten."""
    rows = []
    for seed in (0, 1):
        for r in (1, 2, 5, 10, 20):
            def add(model, rule, failures, planned, flown):
                rows.append({"subset": "FD001", "seed": seed, "model": model, "horizon": 10, "ratio": float(r), "rule": rule,
                             "engines": 10, "failures": failures, "planned": planned, "cycles_flown": flown,
                             "cost_per_1000_cycles": 1000 * (r * failures + planned) / flown})
            add("lgbm", "risky", 1, 9, 2000)
            add("lgbm", "safe", 0, 10, 1800 + 50 * seed)
            add("none", "age (tuned)", 0, 10, 1500)
            add("none", "oracle", 0, 10, 2050)
    return pd.DataFrame(rows)


def test_euros():
    assert euros(2.5, planned_eur=1_000_000) == pytest.approx(2_500_000)


def test_break_even_per_seed_and_summary():
    dec = fake_decisions()
    be = break_even_by_seed(dec, "risky", "safe")
    assert len(be) == 2 and (be["break_even"] > 1).all() and (be["break_even"] < 5).all()
    # seed 1 flies the safe rule longer, so it breaks even earlier
    assert be.set_index("seed").loc[1, "break_even"] < be.set_index("seed").loc[0, "break_even"]
    s = break_even_summary(dec, "risky", "safe", planned_eur=1_000_000)
    assert s.loc[0, "seeds_breaking_even"] == 2
    assert s.loc[0, "failure_cost_eur_at_break_even"] == pytest.approx(1_000_000 * be["break_even"].median())


def test_model_free_rules_are_comparable_with_every_model():
    be = break_even_by_seed(fake_decisions(), "age (tuned)", "safe")
    assert len(be) == 2 and (be["model"] == "lgbm").all() and (be["break_even"] == 0).all()   # safe beats age everywhere


def test_value_captured_is_zero_for_age_and_one_for_oracle():
    d = add_value_captured(fake_decisions())
    assert np.allclose(d.loc[d["rule"] == "age (tuned)", "value_captured"], 0)
    assert np.allclose(d.loc[d["rule"] == "oracle", "value_captured"], 1)
    safe = d.loc[(d["rule"] == "safe") & (d["seed"] == 0), "value_captured"]
    assert ((safe > 0) & (safe < 1)).all()


def test_win_rate():
    w = win_rate(fake_decisions(), "risky", "safe")
    assert w.loc[("FD001", "lgbm", 10), 1.0] == 0 and w.loc[("FD001", "lgbm", 10), 20.0] == 1


def test_curve_costs_reprices_any_ratio():
    curves = pd.DataFrame({"failures": [2.0, 0.0], "planned": [8.0, 10.0], "cycles_flown": [2000.0, 1800.0]})
    c = curve_costs(curves, [1, 10])
    assert c.loc[(c["ratio"] == 10)].iloc[0]["cost_per_1000_cycles"] == pytest.approx(1000 * 28 / 2000)


def test_break_even_runs_counts_never_as_infinite():
    from src.summarize import break_even_runs
    dec = fake_decisions()
    t = break_even_runs(dec, "risky", ["safe"], planned_eur=1_000_000)
    assert t.loc[0, "runs"] == 2 and t.loc[0, "runs_breaking_even"] == 2
    assert 1 < t.loc[0, "median"] < 5 and t.loc[0, "median_failure_cost_eur"] == pytest.approx(1e6 * t.loc[0, "median"])
    never = break_even_runs(dec, "safe", ["risky"])                  # risky never beats safe at the top ratio
    assert never.loc[0, "runs_breaking_even"] == 0 and np.isinf(never.loc[0, "median"])
    assert isinstance(euros(pd.Series([1.0, 2.0]), 10), pd.Series)
