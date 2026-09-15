"""Thin wrappers around MAPIE v1 (SplitConformalRegressor, ConformalizedQuantileRegressor).

Conformalisation always uses the rows of the CALIBRATION ENGINES, never engines seen in training.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from mapie.regression import ConformalizedQuantileRegressor, SplitConformalRegressor

from src.config import CONFIDENCE_LEVEL, RUL_CAP, SEED
from src.models import fit_quantile_trio, make_lgbm_quantile, quantile_alphas

# The truth always lies in [0, RUL_CAP] (the label is capped), so clipping the bounds to that
# range never lowers coverage - it only removes physically impossible width (e.g. RUL < 0).
RUL_RANGE = (0.0, float(RUL_CAP))


def fit_scp(model, X_cal, y_cal, confidence_level: float | Iterable[float] = CONFIDENCE_LEVEL) -> SplitConformalRegressor:
    """Split conformal prediction around an ALREADY FITTED point model, absolute-residual score.

    `confidence_level` may be a list: one conformalisation yields one interval per level.
    """
    scp = SplitConformalRegressor(
        estimator=model, confidence_level=confidence_level, conformity_score="absolute", prefit=True
    )
    return scp.conformalize(X_cal, y_cal)


def fit_cqr(
    X_train,
    y_train,
    X_cal,
    y_cal,
    confidence_level: float = CONFIDENCE_LEVEL,
    seed: int = SEED,
    models: tuple | None = None,
) -> ConformalizedQuantileRegressor:
    """Conformalized quantile regression with three LightGBM quantile models (lower, upper, median).

    Pass `models=(lower, upper, median)` to reuse already fitted quantile models.
    """
    lower, upper, median = models if models is not None else fit_quantile_trio(X_train, y_train, confidence_level, seed)
    cqr = ConformalizedQuantileRegressor(estimator=[lower, upper, median], confidence_level=confidence_level, prefit=True)
    return cqr.conformalize(X_cal, y_cal)


def fit_cqr_per_level(X_train, y_train, X_cal, y_cal, levels: Iterable[float], seed: int = SEED) -> dict[float, ConformalizedQuantileRegressor]:
    """One CQR per confidence level (the median model is shared, the two tail models are refitted)."""
    median = make_lgbm_quantile(0.5, seed).fit(X_train, y_train)
    out = {}
    for level in levels:
        lo_a, hi_a, _ = quantile_alphas(level)
        lower = make_lgbm_quantile(lo_a, seed).fit(X_train, y_train)
        upper = make_lgbm_quantile(hi_a, seed).fit(X_train, y_train)
        out[level] = fit_cqr(X_train, y_train, X_cal, y_cal, level, seed, models=(lower, upper, median))
    return out


def predict_intervals(cp, X, clip: tuple[float, float] | None = RUL_RANGE):
    """(y_pred, lower, upper); lower/upper have shape (n, n_levels). Bounds clipped to `clip`."""
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
