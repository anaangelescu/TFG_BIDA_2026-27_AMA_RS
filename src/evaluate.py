"""Metrics. All functions take plain arrays / Series and return floats."""
from __future__ import annotations

import numpy as np


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
