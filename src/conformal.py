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


# --------------------------------------------------------------------------- Mondrian (stratified) split conformal
def conformal_quantile(residuals, level: float) -> float:
    """Finite-sample split-conformal quantile of absolute residuals: the ceil((n + 1) * level)-th smallest.

    Identical to MAPIE's SplitConformalRegressor (checked in tests). Returns inf when the calibration
    set is too small for the level, and 0 for level 0 (a zero-width interval = the point prediction).
    """
    r = np.sort(np.abs(np.asarray(residuals, float)))
    if level <= 0:
        return 0.0
    k = int(np.ceil((len(r) + 1) * level))
    return float(r[k - 1]) if k <= len(r) else float("inf")


class MondrianSCP:
    """Split conformal prediction calibrated separately inside strata of the PREDICTED value.

    Exact conditional coverage is impossible without assumptions (Barber et al. 2019), but coverage
    within chosen groups is achievable (Vovk 2012; Bostrom & Johansson 2020): each stratum gets the
    conformal quantile of its own calibration residuals. Strata are bins of the point prediction
    (never of the true RUL, which is unknown when the decision is made). With edges=() there is one
    stratum and the result equals plain split conformal.

    A stratum with fewer than `min_count` calibration rows falls back to the pooled quantile.
    Any confidence level can be asked for at prediction time (needed for the cost-matched rule).
    """

    def __init__(self, edges=(25, 50, 75, 100), min_count: int = 50, clip: tuple[float, float] | None = RUL_RANGE):
        self.edges = tuple(float(e) for e in edges)
        self.min_count = min_count
        self.clip = clip

    def stratum(self, pred) -> np.ndarray:
        return np.digitize(np.asarray(pred, float), self.edges, right=True)

    def fit(self, pred_cal, y_cal) -> "MondrianSCP":
        pred_cal, y_cal = np.asarray(pred_cal, float), np.asarray(y_cal, float)
        res = np.abs(y_cal - pred_cal)
        s = self.stratum(pred_cal)
        self.pooled_ = np.sort(res)
        self.residuals_ = {k: np.sort(res[s == k]) for k in range(len(self.edges) + 1)}
        self.counts_ = {k: len(v) for k, v in self.residuals_.items()}
        return self

    def half_width(self, level: float) -> np.ndarray:
        """Half-width per stratum at `level` (index = stratum)."""
        pooled = conformal_quantile(self.pooled_, level)
        return np.array([
            conformal_quantile(r, level) if len(r) >= self.min_count else pooled for r in self.residuals_.values()
        ])

    def predict_interval(self, pred, level: float):
        """(lower, upper) around `pred`; bounds clipped to the feasible RUL range."""
        pred = np.asarray(pred, float)
        hw = self.half_width(level)[self.stratum(pred)]
        lower, upper = pred - hw, pred + hw
        if self.clip is not None:
            lower, upper = np.clip(lower, *self.clip), np.clip(upper, *self.clip)
        return lower, upper


# --------------------------------------------------------------------------- CQR lower bound at any level
class CQRLowerBound:
    """Lower bound of conformalized quantile regression (asymmetric correction, as MAPIE) at ANY level.

    Only the lower bound drives a removal decision. Fitted once on the calibration rows from the
    predictions of the lower quantile model, it returns the bound for any two-sided level L:
        lower = q_lo(x) - Q,  Q = the (1 + L)/2 quantile of the scores q_lo(x_i) - y_i,
    with MAPIE's finite-sample rule (np.quantile, method="higher", at (1 + L)/2 * (1 + 1/n)), so at the
    level the quantile models were trained for it reproduces MAPIE exactly (checked in tests). At other
    levels the lower quantile model is mis-specified but the correction still calibrates the lower tail.
    """

    def __init__(self, clip: tuple[float, float] | None = RUL_RANGE):
        self.clip = clip

    def fit(self, q_lo_cal, y_cal) -> "CQRLowerBound":
        self.scores_ = np.asarray(q_lo_cal, float) - np.asarray(y_cal, float)
        return self

    def correction(self, level: float) -> float:
        n = len(self.scores_)
        q = (1 + level) / 2 * (1 + 1 / n)
        if q > 1:
            return float("inf")
        return float(np.quantile(self.scores_, q, method="higher"))

    def lower(self, q_lo, level: float) -> np.ndarray:
        lo = np.asarray(q_lo, float) - self.correction(level)
        return np.clip(lo, *self.clip) if self.clip is not None else lo
