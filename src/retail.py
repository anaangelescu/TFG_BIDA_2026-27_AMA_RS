"""Retail experiment: the engine decision analysis repeated on demand data (M5 or synthetic).

    python -m src.run retail --data synthetic --seeds 0-4      # runs anywhere, for the pipeline
    python -m src.run retail --data m5 --seeds 0-4             # needs the Kaggle files (src/m5.py)

One LightGBM point model and one LightGBM quantile model per critical ratio are trained on the training
days; conformal shifts and tuned buffers use the calibration days; costs are measured on the test days.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.config import COST_RATIOS
from src.m5 import add_features, feature_cols, load_store_dept, split, synthetic_sales
from src.models import make_lgbm
from src.newsvendor import critical_ratio, decision_sweep


def run_retail(data: str = "synthetic", seed: int = 0, ratios=COST_RATIOS) -> pd.DataFrame:
    long = synthetic_sales(seed=seed) if data == "synthetic" else load_store_dept()
    df = add_features(long)
    parts = split(df)
    feats = feature_cols(df)
    tr, ca, te = parts["train"], parts["cal"], parts["test"]
    point = make_lgbm(seed).fit(tr[feats], tr["demand"])
    ca, te = ca.assign(point=point.predict(ca[feats])), te.assign(point=point.predict(te[feats]))
    q_cols = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tau in sorted({round(critical_ratio(r), 3) for r in ratios}):
            m = make_lgbm(seed, objective="quantile", alpha=tau).fit(tr[feats], tr["demand"])
            col = f"q_{tau:.3f}"
            ca[col], te[col] = m.predict(ca[feats]), m.predict(te[feats])
            q_cols[tau] = col
    out = decision_sweep(ca, te, ratios, buffers=_buffer_grid(tr), q_cols=q_cols)
    rmse = float(np.sqrt(np.mean((te["demand"] - te["point"]) ** 2)))
    return out.assign(data=data, seed=seed, series=te["series"].nunique(), test_days=te["day"].nunique(), point_rmse=rmse)


def _buffer_grid(train: pd.DataFrame) -> np.ndarray:
    """Buffers from 0 to ~1.5 x the typical daily demand, 101 steps (scale-free across datasets)."""
    top = 1.5 * float(train["demand"].median())
    return np.linspace(0, top, 101)
