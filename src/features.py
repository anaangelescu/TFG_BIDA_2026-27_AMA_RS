"""Step 3 - preprocessing.

Pipeline (fitted on TRAINING ENGINES ONLY, then applied to any frame):
  1. operating regime  : FD002/FD004 -> KMeans(6) on the three settings; FD001/FD003 -> one regime
  2. z-score           : each sensor is standardised with the mean/std of ITS regime
  3. constant sensors  : dropped when their std is ~0 in every regime (learned, not hard-coded)
  4. rolling features  : per-unit rolling mean and std over ROLL_WINDOW cycles
The RUL label is min(max_cycle_of_unit - cycle, RUL_CAP) and needs no fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import (
    MULTI_REGIME_SUBSETS,
    N_REGIMES,
    ROLL_WINDOW,
    RUL_CAP,
    SEED,
    SENSOR_COLS,
    SETTING_COLS,
)

# A sensor whose standard deviation (raw units) is below this in every regime carries no signal.
# 0.01 separates the quantisation noise of s_6 (std ~0.0015) from the smallest real sensor
# variation (s_15 / BPR, std ~0.04 in FD001) by more than an order of magnitude each way.
CONSTANT_STD_TOL = 0.01


# --------------------------------------------------------------------------- labels
def add_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Append the piece-wise linear RUL label to a run-to-failure frame.

    RUL = min(max_cycle_of_unit - cycle, cap). Must be called on the FULL training frame
    (the last recorded cycle of each engine is its failure cycle).
    """
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = np.minimum(max_cycle - out["cycle"], cap).astype(int)
    return out


def official_rul(rul_df: pd.DataFrame, cap: int | None = RUL_CAP) -> pd.Series:
    """True RUL of each official test engine (indexed by unit), capped like the training label.

    Capping the test target keeps the definition of "RUL" identical on both sides; set cap=None
    to score against the raw NASA values instead.
    """
    y = rul_df.set_index("unit")["rul"].astype(float)
    return y.clip(upper=cap) if cap is not None else y


# --------------------------------------------------------------------------- variance table
def sensor_variance_table(df: pd.DataFrame, tol: float = CONSTANT_STD_TOL) -> pd.DataFrame:
    """Per-sensor mean/std/variance/#unique with a `constant` flag (std < tol). Used in the EDA."""
    s = df[SENSOR_COLS]
    table = pd.DataFrame({"mean": s.mean(), "std": s.std(), "var": s.var(), "n_unique": s.nunique()})
    table["constant"] = table["std"] < tol
    return table


# --------------------------------------------------------------------------- pipeline
@dataclass
class FeaturePipeline:
    """Regime normalisation + constant-sensor removal + rolling statistics. fit() on train engines only."""

    subset: str
    window: int = ROLL_WINDOW
    n_regimes: int = N_REGIMES
    seed: int = SEED
    # learned attributes (trailing underscore = set by fit)
    setting_scaler_: StandardScaler | None = None
    kmeans_: KMeans | None = None
    regime_mean_: pd.DataFrame | None = None
    regime_std_: pd.DataFrame | None = None
    dropped_sensors_: list[str] = field(default_factory=list)
    kept_sensors_: list[str] = field(default_factory=list)

    @property
    def multi_regime(self) -> bool:
        return self.subset in MULTI_REGIME_SUBSETS

    @property
    def feature_names(self) -> list[str]:
        k = self.kept_sensors_
        return k + [f"{s}_rm{self.window}" for s in k] + [f"{s}_rs{self.window}" for s in k]

    # ---- fit
    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        if self.multi_regime:
            # The six operating conditions form six tight clusters in setting space; the settings
            # are standardised first so that setting_1 (range 0-42) does not dominate the distance.
            self.setting_scaler_ = StandardScaler().fit(train_df[SETTING_COLS])
            self.kmeans_ = KMeans(n_clusters=self.n_regimes, n_init=10, random_state=self.seed).fit(
                self.setting_scaler_.transform(train_df[SETTING_COLS])
            )
        regime = self.assign_regime(train_df)
        grouped = train_df[SENSOR_COLS].groupby(regime.to_numpy())
        self.regime_mean_ = grouped.mean()
        self.regime_std_ = grouped.std()
        # A sensor is constant when it does not move in ANY regime (max over regimes of its std).
        self.dropped_sensors_ = [s for s in SENSOR_COLS if self.regime_std_[s].max() < CONSTANT_STD_TOL]
        self.kept_sensors_ = [s for s in SENSOR_COLS if s not in self.dropped_sensors_]
        return self

    # ---- helpers
    def assign_regime(self, df: pd.DataFrame) -> pd.Series:
        """Regime id per row (all zeros for single-condition subsets)."""
        if not self.multi_regime:
            return pd.Series(0, index=df.index, name="regime")
        labels = self.kmeans_.predict(self.setting_scaler_.transform(df[SETTING_COLS]))
        return pd.Series(labels, index=df.index, name="regime")

    def _zscore(self, df: pd.DataFrame, regime: pd.Series) -> np.ndarray:
        kept = self.kept_sensors_
        mean = self.regime_mean_.loc[regime.to_numpy(), kept].to_numpy()
        std = self.regime_std_.loc[regime.to_numpy(), kept].to_numpy()
        std = np.where(std < CONSTANT_STD_TOL, 1.0, std)  # a kept sensor may still be flat in ONE regime
        return (df[kept].to_numpy(dtype=float) - mean) / std

    # ---- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return unit, cycle, regime, z-scored sensors and per-unit rolling mean/std of them."""
        if self.regime_mean_ is None:
            raise RuntimeError("call fit() on the training engines first")
        regime = self.assign_regime(df)
        out = pd.DataFrame(self._zscore(df, regime), columns=self.kept_sensors_, index=df.index)
        out.insert(0, "regime", regime.to_numpy())
        out.insert(0, "cycle", df["cycle"].to_numpy())
        out.insert(0, "unit", df["unit"].to_numpy())
        out = out.sort_values(["unit", "cycle"])
        # min_periods=1 keeps the first cycles of every engine (rolling std of one value -> 0).
        roll = out.groupby("unit")[self.kept_sensors_].rolling(self.window, min_periods=1)
        r_mean = roll.mean().reset_index(level=0, drop=True).add_suffix(f"_rm{self.window}")
        r_std = roll.std().reset_index(level=0, drop=True).fillna(0.0).add_suffix(f"_rs{self.window}")
        return pd.concat([out, r_mean, r_std], axis=1)

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)


# --------------------------------------------------------------------------- X, y, groups
def to_xy(feat_df: pd.DataFrame, pipeline: FeaturePipeline, label: str | None = "rul"):
    """Split a transformed frame into X (features), y (label or None) and groups (unit)."""
    X = feat_df[pipeline.feature_names]
    y = feat_df[label] if label is not None and label in feat_df else None
    groups = feat_df["unit"]
    return X, y, groups


def build_features(train_df: pd.DataFrame, pipeline: FeaturePipeline, cap: int = RUL_CAP):
    """Label + transform a run-to-failure frame with an already fitted pipeline -> X, y, groups."""
    labelled = add_rul(train_df, cap)
    feat = pipeline.transform(labelled)
    feat["rul"] = labelled.loc[feat.index, "rul"].to_numpy()
    return to_xy(feat, pipeline, "rul")


def last_cycle(feat_df: pd.DataFrame) -> pd.DataFrame:
    """One row per engine: its last recorded cycle (the official test protocol)."""
    idx = feat_df.groupby("unit")["cycle"].idxmax()
    return feat_df.loc[idx].set_index("unit").sort_index()
