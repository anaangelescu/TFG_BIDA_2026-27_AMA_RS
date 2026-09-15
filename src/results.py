"""One row per experiment in results/runs.csv (fixed schema)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RUNS_CSV

RUN_COLUMNS = ["subset", "model", "seed", "rmse", "nasa_score", "coverage", "mean_width", "n_test"]
KEY = ["subset", "model", "seed"]


def append_run(
    subset: str,
    model: str,
    seed: int,
    rmse: float,
    nasa_score: float,
    n_test: int,
    coverage: float = np.nan,
    mean_width: float = np.nan,
    path: Path = RUNS_CSV,
) -> pd.DataFrame:
    """Append one row. A row with the same (subset, model, seed) is replaced so notebooks can be re-run
    without piling up duplicates. Point-only models leave coverage / mean_width as NaN."""
    row = pd.DataFrame(
        [[subset, model, int(seed), float(rmse), float(nasa_score), float(coverage), float(mean_width), int(n_test)]],
        columns=RUN_COLUMNS,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        runs = pd.read_csv(path)
        assert list(runs.columns) == RUN_COLUMNS, f"{path} has an unexpected schema"
        keep = ~(runs[KEY] == row.loc[0, KEY].to_numpy()).all(axis=1)
        runs = pd.concat([runs[keep], row], ignore_index=True)
    else:
        runs = row
    runs.to_csv(path, index=False, float_format="%.6g")
    return row


def load_runs(path: Path = RUNS_CSV) -> pd.DataFrame:
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame(columns=RUN_COLUMNS)
