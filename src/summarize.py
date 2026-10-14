"""Turn the cross-fitting CSVs into the thesis tables: costs in euros, break-even ratios, win rates.

All functions take the DataFrames written by `python -m src.run crossfit` (see src/run.py) and never
re-run a model. Seeds are summarised with the median and the share of seeds that agree, because ten
seeds are too few for a meaningful mean +/- standard error on a ratio.

Business reading of the numbers
-------------------------------
cost_per_1000_cycles   cost of maintenance events per 1,000 flight cycles, in units of ONE planned
                       removal. Multiply by PLANNED_REMOVAL_EUR for euros.
break-even ratio       the failure : planned cost ratio above which a challenger rule is cheaper than
                       the baseline. Multiply by PLANNED_REMOVAL_EUR for "an unplanned failure must cost
                       more than EUR X for the interval to pay off".
value captured         (cost of age-based maintenance - cost of the rule) / (age - oracle): the share of
                       the saving that perfect information would give which the rule actually captures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PLANNED_REMOVAL_EUR, RESULTS_DIR
from src.decision import Costs, break_even, cost_rate

KEYS = ["subset", "seed", "model", "horizon"]
MODEL_FREE = ("age (tuned)", "oracle")
REALISTIC_BAND = (1.4, 10.0)       # failure : planned ratio judged plausible for engine removals (docs/research_direction.md §5)


def load(out_dir: Path = RESULTS_DIR) -> dict[str, pd.DataFrame]:
    names = {"accuracy": "crossfit_accuracy.csv", "coverage": "crossfit_coverage.csv",
             "decisions": "crossfit_decisions.csv", "curves": "crossfit_curves.csv"}
    return {k: pd.read_csv(Path(out_dir) / v) for k, v in names.items() if (Path(out_dir) / v).exists()}


def euros(units_of_planned_removal, planned_eur: float = PLANNED_REMOVAL_EUR):
    """Convert a cost expressed in planned removals (cost rate, break-even ratio) into euros (keeps pandas types)."""
    if isinstance(units_of_planned_removal, (pd.Series, pd.DataFrame)):
        return units_of_planned_removal.astype(float) * planned_eur
    return np.asarray(units_of_planned_removal, float) * planned_eur


def with_model_free(dec: pd.DataFrame) -> pd.DataFrame:
    """Copy the model-free rules (age, oracle) into every model's block so they can be compared directly."""
    free = dec[dec["model"] == "none"]
    blocks = [dec[dec["model"] != "none"]]
    for m in dec.loc[dec["model"] != "none", "model"].unique():
        blocks.append(free.assign(model=m))
    return pd.concat(blocks, ignore_index=True)


def add_value_captured(dec: pd.DataFrame) -> pd.DataFrame:
    """Share of the age -> oracle saving captured by each rule (1 = as good as perfect information)."""
    d = with_model_free(dec)
    ref = d[d["rule"].isin(MODEL_FREE)].pivot_table(index=KEYS + ["ratio"], columns="rule", values="cost_per_1000_cycles")
    d = d.merge(ref.reset_index(), on=KEYS + ["ratio"], how="left")
    gap = d["age (tuned)"] - d["oracle"]
    d["value_captured"] = np.where(gap > 1e-12, (d["age (tuned)"] - d["cost_per_1000_cycles"]) / gap, np.nan)
    return d.drop(columns=list(MODEL_FREE))


def cost_table(dec: pd.DataFrame, horizon: int, model: str = "lgbm", subset: str | None = None,
               metric: str = "cost_per_1000_cycles", stat: str = "median") -> pd.DataFrame:
    """Rules x cost ratios, summarised over seeds (and over subsets unless one is given)."""
    d = with_model_free(dec)
    d = d[(d["horizon"] == horizon) & (d["model"] == model)]
    if subset is not None:
        d = d[d["subset"] == subset]
    return d.pivot_table(index="rule", columns="ratio", values=metric, aggfunc=stat)


def break_even_by_seed(dec: pd.DataFrame, baseline: str, challenger: str) -> pd.DataFrame:
    """Break-even failure : planned ratio per (subset, seed, model, horizon); 0 = cheaper everywhere, NaN = never."""
    d = with_model_free(dec)
    rows = []
    for key, g in d.groupby(KEYS):
        rules = set(g["rule"])
        if {baseline, challenger} <= rules:
            rows.append(dict(zip(KEYS, key), baseline=baseline, challenger=challenger,
                             break_even=break_even(g, baseline, challenger)))
    return pd.DataFrame(rows)


def break_even_summary(dec: pd.DataFrame, baseline: str, challenger: str, planned_eur: float = PLANNED_REMOVAL_EUR) -> pd.DataFrame:
    """Median break-even over seeds, how many seeds ever break even, and the failure cost in euros it implies."""
    be = break_even_by_seed(dec, baseline, challenger)
    if be.empty:
        return be

    def agg(s):
        finite = s.dropna()
        return pd.Series({"seeds": len(s), "seeds_breaking_even": len(finite),
                          "median_break_even": float(np.median(finite)) if len(finite) else np.nan,
                          "min": float(finite.min()) if len(finite) else np.nan,
                          "max": float(finite.max()) if len(finite) else np.nan})
    out = be.groupby(["subset", "model", "horizon"])["break_even"].apply(agg).unstack().reset_index()
    out["failure_cost_eur_at_break_even"] = euros(out["median_break_even"], planned_eur)
    out.insert(3, "challenger", challenger); out.insert(3, "baseline", baseline)
    return out


def break_even_runs(dec: pd.DataFrame, baseline: str, challengers, model: str = "lgbm", by=("horizon",),
                    planned_eur: float = PLANNED_REMOVAL_EUR) -> pd.DataFrame:
    """Headline break-even: the distribution over runs (subset x seed) of the run-level break-even ratio.

    A run that never breaks even counts as +inf, so the median is honest about runs where the
    challenger loses. Reported: median and quartiles (ratio and euros), and how many runs break even.
    Pooling costs across subsets before computing one break-even is avoided on purpose: subsets have
    different engine lifetimes, and the pooled number moves with the pooling choice.
    """
    rows = []
    for c in challengers:
        be = break_even_by_seed(dec, baseline, c)
        be = be[be["model"] == model]
        for key, g in be.groupby(list(by)):
            x = g["break_even"].fillna(np.inf).to_numpy()
            q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75], method="lower")
            key = key if isinstance(key, tuple) else (key,)
            hits = x[np.isfinite(x)]
            rows.append({**dict(zip(by, key)), "challenger": c, "runs": len(x), "runs_breaking_even": len(hits),
                         "median": q50, "q25": q25, "q75": q75, "median_failure_cost_eur": float(euros(q50, planned_eur)),
                         "median_if_breaking_even": float(np.median(hits)) if len(hits) else np.nan})
    return pd.DataFrame(rows)


def win_rate(dec: pd.DataFrame, baseline: str, challenger: str) -> pd.DataFrame:
    """Share of seeds in which the challenger is strictly cheaper, per (subset, model, horizon, ratio)."""
    d = with_model_free(dec)
    w = d[d["rule"].isin([baseline, challenger])].pivot_table(index=KEYS + ["ratio"], columns="rule", values="cost_per_1000_cycles")
    w = w.dropna()
    w["challenger_cheaper"] = (w[challenger] < w[baseline]).astype(float)
    return w.groupby(["subset", "model", "horizon", "ratio"])["challenger_cheaper"].mean().unstack("ratio")


def curve_costs(curves: pd.DataFrame, ratios) -> pd.DataFrame:
    """Cost rate of every dial setting at every ratio (long format), from the `curves` output."""
    parts = []
    for r in np.atleast_1d(ratios):
        parts.append(curves.assign(ratio=float(r), cost_per_1000_cycles=cost_rate(curves, Costs.from_ratio(r))))
    return pd.concat(parts, ignore_index=True)


def oracle_dial(curves: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """Best setting of each dial chosen WITH hindsight on the evaluation engines (an upper bound on tuning)."""
    c = curve_costs(curves, [ratio])
    idx = c.groupby(["subset", "seed", "model", "dial", "horizon"])["cost_per_1000_cycles"].idxmin()
    return c.loc[idx].reset_index(drop=True)


def coverage_table(cov: pd.DataFrame, by: str = "pred", model: str = "lgbm", subset: str | None = None,
                   value: str = "coverage") -> pd.DataFrame:
    """Method x stratum table of coverage (or below / above / mean_width), mean over seeds."""
    c = cov[(cov["by"] == by) & (cov["model"] == model)]
    if subset is not None:
        c = c[c["subset"] == subset]
    t = c.pivot_table(index="method", columns=["stratum", "label"], values=value, aggfunc="mean")
    t.columns = [lab for _, lab in t.columns]
    return t
