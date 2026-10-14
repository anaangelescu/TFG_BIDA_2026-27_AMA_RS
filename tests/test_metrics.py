"""NASA score, RMSE, coverage and width against hand-computed values."""
import numpy as np
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
