"""Thin wrappers around MAPIE v1 (SplitConformalRegressor, ConformalizedQuantileRegressor).

Conformalisation always uses the rows of the CALIBRATION ENGINES, never engines seen in training.
"""
from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from typing import Iterable

import numpy as np
from mapie.regression import ConformalizedQuantileRegressor, SplitConformalRegressor

from src.config import CONFIDENCE_LEVEL, RUL_CAP, SEED
from src.models import _check_not_degenerate, fit_quantile_trio, make_lgbm_quantile, quantile_alphas

# The truth always lies in [0, RUL_CAP] (the label is capped), so clipping the bounds to that
# range never lowers coverage - it only removes physically impossible width (e.g. RUL < 0).
RUL_RANGE = (0.0, float(RUL_CAP))


@contextmanager
def _quiet():
    """Silence two expected, harmless messages while MAPIE runs.

    MAPIE hands numpy arrays to the estimator (sklearn then warns that the feature names are
    missing) and reports quantile crossings of the CQR models at INFO level via logging.basicConfig.
    Crossings are quantified explicitly in notebook 03 instead of being logged row by row.
    """
    previous = logging.root.manager.disable
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        logging.disable(logging.INFO)
        try:
            yield
        finally:
            logging.disable(previous)


def fit_scp(model, X_cal, y_cal, confidence_level: float | Iterable[float] = CONFIDENCE_LEVEL) -> SplitConformalRegressor:
    """Split conformal prediction around an ALREADY FITTED point model, absolute-residual score.

    `confidence_level` may be a list: one conformalisation yields one interval per level.
    """
    scp = SplitConformalRegressor(
        estimator=model, confidence_level=confidence_level, conformity_score="absolute", prefit=True
    )
    with _quiet():
        return scp.conformalize(X_cal, y_cal)


def fit_cqr(
    X_cal,
    y_cal,
    confidence_level: float = CONFIDENCE_LEVEL,
    models: tuple | None = None,
    X_train=None,
    y_train=None,
    seed: int = SEED,
) -> ConformalizedQuantileRegressor:
    """Conformalized quantile regression around three LightGBM quantile models (lower, upper, median).

    Either pass `models=(lower, upper, median)` already fitted on the training engines, or pass
    X_train / y_train and the three models are fitted here.
    """
    if models is None:
        assert X_train is not None and y_train is not None, "give models=(lower, upper, median) or X_train, y_train"
        models = fit_quantile_trio(X_train, y_train, confidence_level, seed)
    lower, upper, median = models
    cqr = ConformalizedQuantileRegressor(estimator=[lower, upper, median], confidence_level=confidence_level, prefit=True)
    with _quiet():
        return cqr.conformalize(X_cal, y_cal)


def fit_cqr_per_level(X_train, y_train, X_cal, y_cal, levels: Iterable[float], seed: int = SEED) -> dict[float, ConformalizedQuantileRegressor]:
    """One CQR per confidence level (the median model is shared, the two tail models are refitted)."""
    median = _check_not_degenerate(make_lgbm_quantile(0.5, seed).fit(X_train, y_train), X_train, 0.5)
    out = {}
    for level in levels:
        lo_a, hi_a, _ = quantile_alphas(level)
        lower = _check_not_degenerate(make_lgbm_quantile(lo_a, seed).fit(X_train, y_train), X_train, lo_a)
        upper = _check_not_degenerate(make_lgbm_quantile(hi_a, seed).fit(X_train, y_train), X_train, hi_a)
        out[level] = fit_cqr(X_cal, y_cal, level, models=(lower, upper, median))
    return out


def predict_intervals(cp, X, clip: tuple[float, float] | None = RUL_RANGE):
    """(y_pred, lower, upper); lower/upper have shape (n, n_levels). Bounds clipped to `clip`."""
    with _quiet():
        y_pred, y_pis = cp.predict_interval(X)  # y_pis: (n, 2, n_levels)
    lower, upper = y_pis[:, 0, :], y_pis[:, 1, :]
    if clip is not None:
        lower, upper = np.clip(lower, *clip), np.clip(upper, *clip)
    return np.asarray(y_pred), lower, upper


def predict_interval(cp, X, clip: tuple[float, float] | None = RUL_RANGE):
    """Single-level convenience: (y_pred, lower, upper) as 1-D arrays."""
    y_pred, lower, upper = predict_intervals(cp, X, clip)
    assert lower.shape[1] == 1, "use predict_intervals() for several confidence levels"
    return y_pred, lower[:, 0], upper[:, 0]
