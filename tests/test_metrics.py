"""NASA score, RMSE, coverage and width against hand-computed values."""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import coverage, coverage_by_stratum, mean_width, nasa_score, rmse


def test_nasa_score_is_zero_for_perfect_predictions():
    assert nasa_score([10, 50, 125], [10, 50, 125]) == 0.0


def test_nasa_score_hand_values():
    assert nasa_score([0], [10]) == pytest.approx(np.e - 1)      # late by 10  -> exp(10/10) - 1
    assert nasa_score([13], [0]) == pytest.approx(np.e - 1)      # early by 13 -> exp(13/13) - 1
    assert nasa_score([0], [20]) == pytest.approx(np.e**2 - 1)   # late by 20  -> exp(2) - 1
    assert nasa_score([0, 13], [10, 0]) == pytest.approx(2 * (np.e - 1))  # sums over samples


def test_nasa_score_penalises_late_more_than_early():
    assert nasa_score([50], [60]) > nasa_score([50], [40])       # same |d| = 10


def test_rmse():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_coverage_and_width():
    y = np.array([5.0, 10.0, 20.0, 30.0])
    lo = np.array([0.0, 11.0, 15.0, 30.0])
    hi = np.array([10.0, 12.0, 25.0, 31.0])
    assert coverage(y, lo, hi) == pytest.approx(0.75)   # 10 is outside [11, 12]; 30 is on the bound -> inside
    assert mean_width(lo, hi) == pytest.approx((10 + 1 + 10 + 1) / 4)


def test_coverage_by_stratum_counts_miss_direction():
    y = np.array([5.0, 30.0, 50.0, 90.0])
    lo = np.array([6.0, 20.0, 40.0, 60.0])     # first row: truth below the lower bound (late-alert side)
    hi = np.array([20.0, 40.0, 45.0, 80.0])    # third and fourth rows: truth above the upper bound
    t = coverage_by_stratum(y, lo, hi, ["near", "near", "mid", "mid"])
    assert t.loc["near", "coverage"] == pytest.approx(0.5) and t.loc["near", "below"] == pytest.approx(0.5)
    assert t.loc["mid", "coverage"] == 0 and t.loc["mid", "above"] == 1
    assert t.loc["mid", "n"] == 2 and t.loc["near", "mean_width"] == pytest.approx(17.0)


# --------------------------------------------------------------------------- conformal quantile / Mondrian
def test_conformal_quantile_matches_mapie_split_conformal():
    import warnings
    from mapie.regression import SplitConformalRegressor
    from sklearn.linear_model import LinearRegression
    from src.conformal import conformal_quantile
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2)); y = X @ [1.0, 2.0] + rng.normal(size=400)
    m = LinearRegression().fit(X[:250], y[:250])
    Xc, yc = X[250:351], y[250:351]
    for level in (0.5, 0.8, 0.9, 0.95):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, pis = SplitConformalRegressor(m, confidence_level=level, prefit=True).conformalize(Xc, yc).predict_interval(X[:1])
        assert (pis[0, 1, 0] - pis[0, 0, 0]) / 2 == pytest.approx(conformal_quantile(yc - m.predict(Xc), level))


def test_mondrian_calibrates_each_stratum_and_falls_back_when_small():
    from src.conformal import MondrianSCP
    pred = np.r_[np.full(100, 10.0), np.full(100, 80.0), np.full(10, 120.0)]
    resid = np.r_[np.arange(100) / 10.0, np.arange(100) * 1.0, np.arange(10) * 1.0]   # near failure: small errors
    m = MondrianSCP(edges=(50, 100), min_count=50, clip=None).fit(pred, pred + resid)
    hw = m.half_width(0.9)
    assert hw[0] == pytest.approx(9.0)          # ceil(101 * 0.9) = 91st of 0.0..9.9
    assert hw[1] == pytest.approx(90.0)         # 91st of 0..99
    assert hw[2] == pytest.approx(np.sort(np.abs(resid))[int(np.ceil(211 * 0.9)) - 1])   # 10 rows < 50 -> pooled
    lo, hi = m.predict_interval([10.0, 80.0], 0.9)
    assert hi[0] - lo[0] == pytest.approx(18.0) and hi[1] - lo[1] == pytest.approx(180.0)
    assert m.half_width(0.0).tolist() == [0.0, 0.0, 0.0]   # level 0 -> the point prediction itself


def test_single_stratum_mondrian_is_plain_split_conformal():
    from src.conformal import MondrianSCP, conformal_quantile
    rng = np.random.default_rng(1)
    pred = rng.uniform(0, 125, 300); y = pred + rng.normal(0, 10, 300)
    m = MondrianSCP(edges=(), clip=None).fit(pred, y)
    assert m.half_width(0.9)[0] == pytest.approx(conformal_quantile(y - pred, 0.9))


def test_cqr_lower_bound_matches_mapie_at_the_trained_level():
    from src.conformal import CQRLowerBound, fit_cqr, predict_interval
    from src.models import fit_quantile_trio
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.uniform(0, 10, 1600), "b": rng.normal(size=1600)})
    y = 3 * X["a"] + rng.normal(0, 1 + X["a"] / 3)
    Xtr, ytr, Xca, yca, Xte = X[:1000], y[:1000], X[1000:1400], y[1000:1400], X[1400:]
    trio = fit_quantile_trio(Xtr, ytr, 0.9, seed=0)
    _, lo_mapie, _ = predict_interval(fit_cqr(Xca, yca, 0.9, models=trio), Xte, clip=None)
    ours = CQRLowerBound(clip=None).fit(trio[0].predict(Xca), yca).lower(trio[0].predict(Xte), 0.9)
    np.testing.assert_allclose(ours, lo_mapie)
    # a higher level gives a lower (more cautious) bound
    assert (CQRLowerBound(clip=None).fit(trio[0].predict(Xca), yca).lower(trio[0].predict(Xte), 0.98) <= ours).all()
