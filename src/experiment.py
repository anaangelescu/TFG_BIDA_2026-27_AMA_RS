"""One (subset, seed) experiment end to end. Notebooks only call these small functions.

Data flow for one seed:
  load -> split ENGINES 70/15/15 -> fit FeaturePipeline on the 70% train engines only
       -> X, y, groups for train / cal / internal test (all rows of those engines)
       -> official test = last cycle of each test engine vs RUL_FD00X.txt (capped like the label)
       -> point models (linear, lgbm) -> SCP and CQR around LightGBM -> metrics -> runs.csv
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import CONFIDENCE_LEVEL, RUL_CAP, SEED
from src.conformal import RUL_RANGE, fit_cqr, fit_scp, predict_interval
from src.evaluate import interval_metrics, point_metrics
from src.features import FeaturePipeline, build_features, last_cycle, official_rul
from src.load import load_subset
from src.models import make_lgbm, make_linear
from src.results import append_run
from src.split import rows_of, split_engines

EVAL_SETS = ("test", "official")  # internal test engines (all rows) and the official file (last cycle)


@dataclass
class SplitData:
    """Everything one seed needs: fitted pipeline, engine sets and (X, y, groups) per set."""

    subset: str
    seed: int
    pipeline: FeaturePipeline
    units: dict[str, np.ndarray]
    sets: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]]

    def X(self, name: str) -> pd.DataFrame:
        return self.sets[name][0]

    def y(self, name: str) -> pd.Series:
        return self.sets[name][1]

    def groups(self, name: str) -> pd.Series:
        return self.sets[name][2]

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {n: {"engines": int(self.groups(n).nunique()), "rows": len(self.y(n)), "features": self.X(n).shape[1]} for n in self.sets}
        ).T


def prepare(subset: str = "FD001", seed: int = SEED, cap: int = RUL_CAP) -> SplitData:
    train_df, test_df, rul_df = load_subset(subset)
    units = split_engines(train_df, seed)
    # Scalers / regimes / constant-sensor detection see the 70% training engines only.
    pipeline = FeaturePipeline(subset, seed=seed).fit(rows_of(train_df, units["train"]))
    sets = {name: build_features(rows_of(train_df, u), pipeline, cap) for name, u in units.items()}
    # Official protocol: one prediction per test engine, at its last recorded cycle.
    feat_off = last_cycle(pipeline.transform(test_df))
    y_off = official_rul(rul_df, cap).loc[feat_off.index]
    sets["official"] = (feat_off[pipeline.feature_names], y_off, pd.Series(feat_off.index, index=feat_off.index, name="unit"))
    return SplitData(subset, seed, pipeline, units, sets)


def fit_point_models(data: SplitData, seed: int = SEED) -> dict[str, object]:
    X, y = data.X("train"), data.y("train")
    return {"linear": make_linear().fit(X, y), "lgbm": make_lgbm(seed).fit(X, y)}


def evaluate_point(models: dict, data: SplitData, eval_sets=EVAL_SETS) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        for s in eval_sets:
            y = data.y(s)
            rows.append({"model": name, "eval_set": s, **point_metrics(y, model.predict(data.X(s))), "n": len(y)})
    return pd.DataFrame(rows)


def fit_conformal(data: SplitData, lgbm, seed: int = SEED, confidence_level: float = CONFIDENCE_LEVEL) -> dict:
    """SCP wraps the fitted LightGBM; CQR trains its own three quantile LightGBMs on the train engines."""
    Xc, yc = data.X("cal"), data.y("cal")
    return {
        "lgbm_scp": fit_scp(lgbm, Xc, yc, confidence_level),
        "lgbm_cqr": fit_cqr(data.X("train"), data.y("train"), Xc, yc, confidence_level, seed),
    }


def evaluate_conformal(cps: dict, data: SplitData, eval_sets=EVAL_SETS, clip=RUL_RANGE):
    """Metrics table plus the raw (pred, lower, upper) per (model, eval_set) for plotting."""
    rows, preds = [], {}
    for name, cp in cps.items():
        for s in eval_sets:
            y = data.y(s)
            pred, lo, hi = predict_interval(cp, data.X(s), clip)
            preds[(name, s)] = (pred, lo, hi)
            rows.append({"model": name, "eval_set": s, **point_metrics(y, pred), **interval_metrics(y, lo, hi), "n": len(y)})
    return pd.DataFrame(rows), preds


def log_official(table: pd.DataFrame, subset: str, seed: int) -> None:
    """Append the official-test rows of a metrics table to results/runs.csv (one row per model)."""
    for _, r in table[table["eval_set"] == "official"].iterrows():
        append_run(
            subset, r["model"], seed, r["rmse"], r["nasa_score"], int(r["n"]),
            coverage=r.get("coverage", np.nan), mean_width=r.get("mean_width", np.nan),
        )


def run_seed(subset: str = "FD001", seed: int = SEED, confidence_level: float = CONFIDENCE_LEVEL, log: bool = True) -> dict:
    """Steps 4 + 5 for one seed. Returns everything the plotting notebooks need."""
    data = prepare(subset, seed)
    models = fit_point_models(data, seed)
    point_table = evaluate_point(models, data)
    cps = fit_conformal(data, models["lgbm"], seed, confidence_level)
    interval_table, preds = evaluate_conformal(cps, data)
    if log:
        log_official(point_table, subset, seed)
        log_official(interval_table, subset, seed)
    return {
        "data": data, "models": models, "conformal": cps,
        "point_table": point_table, "interval_table": interval_table, "predictions": preds,
    }
