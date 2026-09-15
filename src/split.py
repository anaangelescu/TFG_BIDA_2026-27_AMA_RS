"""Engine-level splits. Rows of one engine are a correlated time series, so every split is by unit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import FRAC_CAL, FRAC_TEST, FRAC_TRAIN, SEED


def split_engines(
    df: pd.DataFrame,
    seed: int = SEED,
    frac_train: float = FRAC_TRAIN,
    frac_cal: float = FRAC_CAL,
    frac_test: float = FRAC_TEST,
) -> dict[str, np.ndarray]:
    """Return three disjoint sets of unit ids: {"train", "cal", "test"} (70 / 15 / 15 by default).

    Two GroupShuffleSplit calls with groups=unit: first train vs rest, then cal vs test.
    test_size is a fraction of GROUPS (engines), not of rows.
    """
    assert abs(frac_train + frac_cal + frac_test - 1.0) < 1e-9, "fractions must sum to 1"
    # Integer engine counts (fractions of groups suffer from float rounding: ceil(0.3 * 100) = 31).
    n_units = df["unit"].nunique()
    n_train, n_test = int(round(frac_train * n_units)), int(round(frac_test * n_units))
    n_cal = n_units - n_train - n_test
    first = GroupShuffleSplit(n_splits=1, train_size=n_train, test_size=n_cal + n_test, random_state=seed)
    train_idx, rest_idx = next(first.split(df, groups=df["unit"]))
    rest = df.iloc[rest_idx]
    second = GroupShuffleSplit(n_splits=1, train_size=n_cal, test_size=n_test, random_state=seed)
    cal_idx, test_idx = next(second.split(rest, groups=rest["unit"]))
    units = {
        "train": np.sort(df.iloc[train_idx]["unit"].unique()),
        "cal": np.sort(rest.iloc[cal_idx]["unit"].unique()),
        "test": np.sort(rest.iloc[test_idx]["unit"].unique()),
    }
    _assert_disjoint(units, df["unit"].unique())
    return units


def _assert_disjoint(units: dict[str, np.ndarray], all_units: np.ndarray) -> None:
    sets = {k: set(v) for k, v in units.items()}
    assert not (sets["train"] & sets["cal"]) and not (sets["train"] & sets["test"]) and not (sets["cal"] & sets["test"])
    assert sets["train"] | sets["cal"] | sets["test"] == set(all_units), "every engine must land in exactly one set"


def rows_of(df: pd.DataFrame, units: np.ndarray) -> pd.DataFrame:
    """All rows belonging to the given engines."""
    return df[df["unit"].isin(units)]
