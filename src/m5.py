"""M5 (Walmart) demand data for the retail generalisation, plus a synthetic stand-in for tests.

The M5 files are NOT in the repository and cannot be downloaded from inside the analysis container.
Download them once from Kaggle ("M5 Forecasting - Uncertainty", Makridakis et al. 2022):
    sales_train_evaluation.csv   (30,490 item-store series, days d_1 .. d_1941)
    calendar.csv, sell_prices.csv
and point M5_DATA_DIR at the folder (or put them in data/m5/).

Reduced scope (docs/research_direction.md §7): the 70 store x department series (M5 aggregation level
9), daily, 28-day-ahead direct forecasts. Lags start at 28 so a forecast for day t never uses demand
after t - 28: the 28-day horizon needs no recursion and no leakage.

Split by time, never at random (rolling origin):
    train        days up to   d_1857
    calibration  d_1858 .. d_1885   (28 days: conformal shifts and tuned buffers)
    test         d_1886 .. d_1913   (28 days)
    (d_1914 .. d_1941 is the official evaluation window: kept for a final check only.)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT, SEED

LAGS = (28, 35, 42, 49, 56, 364)
ROLL = (7, 28)
SPLIT_DAYS = {"train_end": 1857, "cal_end": 1885, "test_end": 1913}


def m5_dir() -> Path:
    return Path(os.environ.get("M5_DATA_DIR", ROOT / "data" / "m5"))


def load_store_dept(data_dir: Path | None = None) -> pd.DataFrame:
    """Long frame: series (store_dept), day (int), date, demand - the 70 level-9 series of M5."""
    d = Path(data_dir) if data_dir else m5_dir()
    path = d / "sales_train_evaluation.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: download M5 from Kaggle and set M5_DATA_DIR (see src/m5.py)")
    sales = pd.read_csv(path)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    agg = sales.groupby(["store_id", "dept_id"])[day_cols].sum()
    long = agg.stack().rename("demand").reset_index().rename(columns={"level_2": "d"})
    long["series"] = long["store_id"] + "_" + long["dept_id"]
    long["day"] = long["d"].str[2:].astype(int)
    cal = pd.read_csv(d / "calendar.csv", usecols=["d", "date"])
    long = long.merge(cal, on="d", how="left")
    return long[["series", "day", "date", "demand"]].sort_values(["series", "day"]).reset_index(drop=True)


def synthetic_sales(n_series: int = 12, n_days: int = 1941, seed: int = SEED) -> pd.DataFrame:
    """Store-level-like demand: level x weekly seasonality x slow trend, Poisson noise with heavier tails."""
    rng = np.random.default_rng(seed)
    parts = []
    t = np.arange(1, n_days + 1)
    for s in range(n_series):
        level = rng.uniform(200, 2000)
        weekly = 1 + rng.uniform(0.1, 0.4) * np.sin(2 * np.pi * (t + rng.integers(7)) / 7)
        trend = 1 + rng.uniform(-0.2, 0.4) * t / n_days
        mu = level * weekly * trend * rng.gamma(20, 1 / 20, n_days)          # day-to-day multiplicative shocks
        parts.append(pd.DataFrame({"series": f"S{s:02d}", "day": t, "demand": rng.poisson(mu).astype(float)}))
    return pd.concat(parts, ignore_index=True)


def add_features(long: pd.DataFrame) -> pd.DataFrame:
    """Lag and rolling features computed per series, all shifted by at least 28 days."""
    df = long.sort_values(["series", "day"]).copy()
    g = df.groupby("series")["demand"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    for w in ROLL:
        df[f"roll_mean_{w}"] = g.transform(lambda x, w=w: x.shift(28).rolling(w).mean())
    df["dow"] = df["day"] % 7
    df["series_code"] = df["series"].astype("category").cat.codes
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("lag_", "roll_mean_")) or c in ("dow", "series_code")]


def split(df: pd.DataFrame, days: dict = SPLIT_DAYS) -> dict[str, pd.DataFrame]:
    """Time split; rows with missing lags (the first year) are dropped from training."""
    feats = feature_cols(df)
    train = df[df["day"] <= days["train_end"]].dropna(subset=feats)
    cal = df[(df["day"] > days["train_end"]) & (df["day"] <= days["cal_end"])]
    test = df[(df["day"] > days["cal_end"]) & (df["day"] <= days["test_end"])]
    assert train["day"].max() < cal["day"].min() and cal["day"].max() < test["day"].min()
    return {"train": train, "cal": cal, "test": test}
