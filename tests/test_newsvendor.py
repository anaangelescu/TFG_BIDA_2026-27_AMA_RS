"""Newsvendor decision layer (retail generalisation) and the synthetic M5 stand-in."""
import numpy as np
import pandas as pd
import pytest

from src.m5 import add_features, feature_cols, split, synthetic_sales
from src.newsvendor import critical_ratio, decision_sweep, evaluate_stock, newsvendor_cost, pinball, upper_conformal_shift


def test_critical_ratio_mirrors_the_engine_threshold():
    from src.decision import cost_matched_tau
    for r in (1, 3, 19):
        assert critical_ratio(r) == pytest.approx(1 - cost_matched_tau(r))   # upper quantile vs lower quantile
    assert critical_ratio(19) == pytest.approx(0.95)


def test_newsvendor_cost_is_scaled_pinball_loss():
    rng = np.random.default_rng(0)
    d, q = rng.poisson(50, 200), rng.normal(50, 10, 200)
    cu, co = 4.0, 1.0
    tau = cu / (cu + co)
    np.testing.assert_allclose(newsvendor_cost(d, q, cu, co), (cu + co) * pinball(d, q, tau))


def test_stocking_the_critical_quantile_is_optimal_for_known_demand():
    rng = np.random.default_rng(1)
    d = rng.normal(100, 20, 200_000)
    r = 4.0
    costs = {q: newsvendor_cost(d, q, r, 1).mean() for q in np.arange(90, 141, 1.0)}
    best = min(costs, key=costs.get)
    assert abs(best - np.quantile(d, critical_ratio(r))) <= 1.5


def test_upper_conformal_shift_covers():
    rng = np.random.default_rng(2)
    res_cal, res_test = rng.normal(0, 5, 2000), rng.normal(0, 5, 20000)
    s = upper_conformal_shift(res_cal, 0.9)
    assert np.mean(res_test <= s) == pytest.approx(0.9, abs=0.02)
    assert upper_conformal_shift(res_cal[:5], 0.99) == np.inf


def test_decision_sweep_rules_and_costs():
    rng = np.random.default_rng(3)
    def frame(n):
        point = rng.uniform(80, 120, n)
        return pd.DataFrame({"point": point, "demand": point + rng.normal(5, 10, n)})   # forecast biased low by 5
    cal, test = frame(3000), frame(3000)
    out = decision_sweep(cal, test, ratios=[1, 10], buffers=np.arange(0, 40))
    piv = out.pivot(index="rule", columns="ratio", values="cost_per_period")
    assert piv.loc["point + tuned buffer", 10.0] < piv.loc["point", 10.0]          # a buffer pays when stockouts are dear
    assert out.loc[(out.rule == "conformal tau") & (out.ratio == 10), "stockout_share"].item() == pytest.approx(1 / 11, abs=0.03)
    assert evaluate_stock([10, 10], [12, 8], ratio=3)["cost_per_period"] == pytest.approx((2 * 1 + 2 * 3) / 2)


def test_synthetic_m5_features_never_look_less_than_28_days_back():
    long = synthetic_sales(n_series=2, n_days=500, seed=0)
    df = add_features(long)
    s0 = df[df["series"] == "S00"].set_index("day")
    assert s0.loc[100, "lag_28"] == s0.loc[72, "demand"]
    assert s0.loc[100, "roll_mean_7"] == pytest.approx(s0.loc[66:72, "demand"].mean())
    parts = split(df, {"train_end": 400, "cal_end": 450, "test_end": 500})
    assert parts["train"][feature_cols(df)].notna().all().all()
