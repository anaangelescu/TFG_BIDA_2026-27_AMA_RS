"""Metrics. All functions take plain arrays / Series and return floats."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """PHM08 scoring function (Saxena et al. 2008), summed over samples.

    d = pred - true.  Early predictions (d < 0) cost exp(-d/13) - 1, late ones (d >= 0) cost
    exp(d/10) - 1, so predicting an engine will last longer than it does is penalised harder.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def coverage(y_true, lower, upper) -> float:
    """Fraction of true values inside [lower, upper] (closed interval)."""
    y = np.asarray(y_true, float)
    return float(np.mean((np.asarray(lower, float) <= y) & (y <= np.asarray(upper, float))))


def mean_width(lower, upper) -> float:
    return float(np.mean(np.asarray(upper, float) - np.asarray(lower, float)))


def point_metrics(y_true, y_pred) -> dict[str, float]:
    return {"rmse": rmse(y_true, y_pred), "nasa_score": nasa_score(y_true, y_pred)}


def interval_metrics(y_true, lower, upper) -> dict[str, float]:
    return {"coverage": coverage(y_true, lower, upper), "mean_width": mean_width(lower, upper)}


def coverage_by_stratum(y_true, lower, upper, strata) -> pd.DataFrame:
    """Coverage per stratum, with the direction of the misses.

    `below` = share of rows whose truth lies under the lower bound (for RUL: the engine has LESS life
    than the interval admits, i.e. a late alert); `above` = share above the upper bound. Strata for
    calibration must come from PREDICTED RUL; true-RUL strata are for diagnostic plots only.
    """
    y = np.asarray(y_true, float)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    df = pd.DataFrame({"stratum": np.asarray(strata), "covered": (lo <= y) & (y <= hi), "below": y < lo, "above": y > hi, "width": hi - lo})
    out = df.groupby("stratum", observed=True).agg(
        n=("covered", "size"), coverage=("covered", "mean"), below=("below", "mean"), above=("above", "mean"), mean_width=("width", "mean")
    )
    return out
