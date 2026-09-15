"""Step 1 - load the raw C-MAPSS files and check their integrity.

Usage:  python -m src.load --check      (prints a compact summary per subset)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import COLUMNS, SUBSETS, find_data_dir

# Expected (train_rows, train_units, test_rows, test_units). Taken from the NASA readme and the
# thesis README; asserted on every load so a corrupted download is caught immediately.
EXPECTED_COUNTS: dict[str, tuple[int, int, int, int]] = {
    "FD001": (20631, 100, 13096, 100),
    "FD002": (53759, 260, 33991, 259),
    "FD003": (24720, 100, 16596, 100),
    "FD004": (61249, 249, 41214, 248),
}


def _read_cmapss(path: Path) -> pd.DataFrame:
    """Read one space-separated file (26 columns, no header, trailing blanks) into a typed frame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")
    assert df.shape[1] == len(COLUMNS), f"{path.name}: expected 26 columns, got {df.shape[1]}"
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df


def _check_contiguous_cycles(df: pd.DataFrame, name: str) -> None:
    """Every engine must run cycles 1..max with no gaps; models rely on this ordering."""
    per_unit = df.groupby("unit")["cycle"].agg(["min", "max", "count"])
    assert (per_unit["min"] == 1).all(), f"{name}: some engine does not start at cycle 1"
    assert (per_unit["max"] == per_unit["count"]).all(), f"{name}: gaps in cycle numbering"


def load_subset(name: str, data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df, rul_df) for one subset, e.g. "FD001".

    train_df / test_df have columns unit, cycle, setting_1..3, s_1..s_21.
    rul_df has columns unit, rul (true RUL after the last recorded cycle of each test engine).
    Row and unit counts are asserted against EXPECTED_COUNTS.
    """
    if name not in EXPECTED_COUNTS:
        raise ValueError(f"unknown subset {name!r}; choose one of {SUBSETS}")
    data_dir = Path(data_dir) if data_dir is not None else find_data_dir()

    train_df = _read_cmapss(data_dir / f"train_{name}.txt")
    test_df = _read_cmapss(data_dir / f"test_{name}.txt")
    rul = pd.read_csv(data_dir / f"RUL_{name}.txt", sep=r"\s+", header=None, names=["rul"], engine="python")
    # RUL_FD00X.txt lists one value per test engine in engine order (1..n).
    rul_df = pd.DataFrame({"unit": range(1, len(rul) + 1), "rul": rul["rul"].astype(int).to_numpy()})

    n_train_rows, n_train_units, n_test_rows, n_test_units = EXPECTED_COUNTS[name]
    assert len(train_df) == n_train_rows, f"{name}: train rows {len(train_df)} != {n_train_rows}"
    assert train_df["unit"].nunique() == n_train_units, f"{name}: train units != {n_train_units}"
    assert len(test_df) == n_test_rows, f"{name}: test rows {len(test_df)} != {n_test_rows}"
    assert test_df["unit"].nunique() == n_test_units, f"{name}: test units != {n_test_units}"
    assert len(rul_df) == n_test_units, f"{name}: RUL file has {len(rul_df)} rows, expected {n_test_units}"
    assert not train_df.isna().any().any() and not test_df.isna().any().any(), f"{name}: NaNs in data"
    _check_contiguous_cycles(train_df, f"train_{name}")
    _check_contiguous_cycles(test_df, f"test_{name}")
    return train_df, test_df, rul_df


def summarize(name: str, data_dir: Path | None = None) -> dict:
    """Compact per-subset summary (also used by the EDA notebook)."""
    train_df, test_df, rul_df = load_subset(name, data_dir)
    life = train_df.groupby("unit")["cycle"].max()          # run-to-failure lifetimes
    test_len = test_df.groupby("unit")["cycle"].max()       # truncated test histories
    return {
        "subset": name,
        "train_rows": len(train_df),
        "train_units": train_df["unit"].nunique(),
        "test_rows": len(test_df),
        "test_units": test_df["unit"].nunique(),
        "life_min": int(life.min()),
        "life_median": float(life.median()),
        "life_max": int(life.max()),
        "test_len_min": int(test_len.min()),
        "test_len_max": int(test_len.max()),
        "rul_min": int(rul_df["rul"].min()),
        "rul_median": float(rul_df["rul"].median()),
        "rul_max": int(rul_df["rul"].max()),
        # settings carry small noise; rounding reveals the number of distinct operating conditions
        "n_conditions": int(train_df[["setting_1", "setting_2", "setting_3"]].round({"setting_1": 0, "setting_2": 2, "setting_3": 0}).drop_duplicates().shape[0]),
    }


def summary_table(subsets: list[str] | None = None, data_dir: Path | None = None) -> pd.DataFrame:
    return pd.DataFrame([summarize(s, data_dir) for s in (subsets or SUBSETS)]).set_index("subset")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and check the C-MAPSS subsets.")
    parser.add_argument("--check", action="store_true", help="assert counts and print a summary for all subsets")
    parser.add_argument("--subset", default=None, help="only this subset, e.g. FD001")
    args = parser.parse_args()
    subsets = [args.subset] if args.subset else SUBSETS
    print(f"data dir: {find_data_dir()}")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary_table(subsets).T.astype(object))  # object dtype keeps the integers readable
    print("all integrity checks passed" if args.check else "")


if __name__ == "__main__":
    main()
