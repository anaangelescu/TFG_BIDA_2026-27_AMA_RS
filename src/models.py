"""Untuned baseline models with a fixed seed. No hyper-parameter search in week one."""
from __future__ import annotations

from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from src.config import SEED

# "Default-ish" LightGBM: slightly more, smaller steps than the library default (100 trees at 0.1)
# and row/column subsampling so that the seed actually matters. Fixed for every experiment.
LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
)


def make_linear() -> LinearRegression:
    """Ordinary least squares - the floor every other model must beat."""
    return LinearRegression()


def make_lgbm(seed: int = SEED, **overrides) -> LGBMRegressor:
    return LGBMRegressor(random_state=seed, **{**LGBM_PARAMS, **overrides})


def make_lgbm_quantile(alpha: float, seed: int = SEED) -> LGBMRegressor:
    """LightGBM trained with the pinball loss for quantile `alpha` (used by CQR)."""
    return make_lgbm(seed, objective="quantile", alpha=alpha)


def quantile_alphas(confidence_level: float) -> tuple[float, float, float]:
    """(lower, upper, median) quantile levels for a central interval at `confidence_level`."""
    return (1 - confidence_level) / 2, (1 + confidence_level) / 2, 0.5


def fit_quantile_trio(X, y, confidence_level: float, seed: int = SEED) -> tuple[LGBMRegressor, LGBMRegressor, LGBMRegressor]:
    """Fit the three quantile models CQR needs, in MAPIE's expected order: lower, upper, median."""
    return tuple(make_lgbm_quantile(a, seed).fit(X, y) for a in quantile_alphas(confidence_level))
