"""Engine-level cross-fitting for the decision experiment: every training engine is evaluated exactly once.

For fold k of N_FOLDS (engines shuffled with the seed):
    evaluation engines  = fold k
    calibration engines = fold k + 1 (mod K)  -> conformal calibration AND tuning of the decision rules
    training engines    = the other K - 2 folds -> feature pipeline, point models, quantile models
Nothing fitted ever sees the engines it is evaluated on. Pooling the evaluation folds gives decisions and
coverage over ALL run-to-failure engines of a subset (100-260) instead of the 15 internal-test engines.

Outputs per (subset, seed), all aggregated (no raw predictions are stored):
    accuracy  - RMSE per model on all evaluation cycles and on cycles with true RUL <= 50
    coverage  - coverage, miss direction and width per stratum (predicted and true RUL), SCP / Mondrian / CQR
    decisions - failures, planned removals, wasted and flown cycles per (model, rule, horizon, cost ratio)
    curves    - the same sums for every value of each rule's dial (safety margin, confidence level,
                replacement age) on the evaluation engines: the cost at ANY ratio, and the cost of a
                planner moving the dial by hand, follow from these without re-running anything
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from src.config import COST_RATIOS, HORIZONS, N_FOLDS, RUL_CAP, SEED, STRATA_EDGES
from src.conformal import CQRLowerBound, MondrianSCP, fit_cqr, predict_interval
from src.decision import Costs, best_threshold, cost_matched_tau, level_for_tau, threshold_outcomes, true_rul
from src.features import FeaturePipeline, build_features
from src.load import load_subset
from src.models import fit_quantile_trio, make_lgbm, make_linear
from src.split import rows_of

LEVEL = 0.90                                                    # fixed-level interval rules
TUNE_LEVELS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99)
MARGINS = np.arange(0, 101, 1.0)
TRUE_EDGES = (25, 50, 75, 100, RUL_CAP)                          # diagnostic strata on the TRUE RUL
CURVE_LEVELS = tuple(np.round(np.r_[np.arange(0, 0.95, 0.05), 0.95, 0.97, 0.98, 0.99], 2))
SUMS = ["engines", "failures", "planned", "wasted_cycles", "cycles_flown"]


def fold_roles(folds: list[np.ndarray], f: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(train, calibration, evaluation) engines for fold f: eval = f, cal = f + 1 (mod K), train = the rest."""
    k = len(folds)
    train = np.concatenate([folds[j] for j in range(k) if j not in (f, (f + 1) % k)])
    return train, folds[(f + 1) % k], folds[f]


def engine_folds(units, k: int = N_FOLDS, seed: int = SEED) -> list[np.ndarray]:
    """Shuffle the engine ids with `seed` and cut them into k nearly equal folds."""
    perm = np.random.default_rng(seed).permutation(np.sort(np.unique(units)))
    return [np.sort(f) for f in np.array_split(perm, k)]


# --------------------------------------------------------------------------- one fold
def _frame(train_df, units, pipeline):
    X, y, g = build_features(rows_of(train_df, units), pipeline)
    base = pd.DataFrame({"unit": g.to_numpy(), "cycle": train_df.loc[X.index, "cycle"].to_numpy(), "y": y.to_numpy()}, index=X.index)
    base["true_rul"] = true_rul(base).to_numpy()
    return X, base


def fold_predictions(train_df, subset, seed, units_train, units_cal, units_eval) -> dict:
    """Point predictions, CQR bounds and fitted (Mondrian) calibrators for one fold."""
    pipeline = FeaturePipeline(subset, seed=seed).fit(rows_of(train_df, units_train))
    X_tr, tr = _frame(train_df, units_train, pipeline)
    X_ca, ca = _frame(train_df, units_cal, pipeline)
    X_ev, ev = _frame(train_df, units_eval, pipeline)
    models = {"linear": make_linear().fit(X_tr, tr["y"]), "lgbm": make_lgbm(seed).fit(X_tr, tr["y"])}
    out = {"cal": {}, "eval": {}, "scp": {}, "mondrian": {}}
    for name, model in models.items():
        p_ca, p_ev = model.predict(X_ca), model.predict(X_ev)
        out["cal"][name] = ca.assign(pred=p_ca)
        out["eval"][name] = ev.assign(pred=p_ev)
        out["scp"][name] = MondrianSCP(edges=()).fit(p_ca, ca["y"])
        out["mondrian"][name] = MondrianSCP(edges=STRATA_EDGES).fit(p_ca, ca["y"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trio = fit_quantile_trio(X_tr, tr["y"], LEVEL, seed)
        cqr = fit_cqr(X_ca, ca["y"], LEVEL, models=trio)
        for role, X in (("cal", X_ca), ("eval", X_ev)):
            _, lo, hi = predict_interval(cqr, X)
            out[role]["lgbm"] = out[role]["lgbm"].assign(cqr_lower=lo, cqr_upper=hi, q_lo=trio[0].predict(X))
    out["cqr_lower"] = {"lgbm": CQRLowerBound().fit(out["cal"]["lgbm"]["q_lo"], ca["y"])}
    return out


# --------------------------------------------------------------------------- rules
def _rule_curves(fp: dict, model: str, horizon: int) -> dict:
    """For every rule: a function costs -> (sums over the EVALUATION engines, tuned parameter).

    Tuned rules pick their parameter on the CALIBRATION engines for the given costs, then apply it once.
    """
    cal, ev = fp["cal"][model], fp["eval"][model]
    scp, mon = fp["scp"][model], fp["mondrian"][model]
    H = horizon

    def fixed(score_ev):
        curve = threshold_outcomes(ev, score_ev, [H], H)
        return lambda costs: (curve.iloc[0], np.nan)

    def tuned_threshold(score_cal, score_ev, thresholds):
        cal_curve = threshold_outcomes(cal, score_cal, thresholds, H)
        ev_curve = threshold_outcomes(ev, score_ev, thresholds, H).set_index("threshold")
        def run(costs):
            t = best_threshold(cal_curve, costs)
            return ev_curve.loc[t].rename(None), t
        return run

    # lower(frame, level) -> lower bound of the interval at that level for the rows of `frame`
    def mondrian_lower(calib):
        return lambda frame, level: calib.predict_interval(frame["pred"], level)[0]

    def level_rule(lower, level_of_costs):
        cache = {}
        def run(costs):
            lev = level_of_costs(costs)
            if lev not in cache:
                cache[lev] = threshold_outcomes(ev, lower(ev, lev), [H], H).iloc[0]
            return cache[lev], lev
        return run

    def tuned_level(lower):
        cal_rows = pd.concat([threshold_outcomes(cal, lower(cal, l), [H], H).assign(threshold=l) for l in TUNE_LEVELS])
        ev_rows = {l: threshold_outcomes(ev, lower(ev, l), [H], H).iloc[0] for l in TUNE_LEVELS}
        def run(costs):
            l = best_threshold(cal_rows.reset_index(drop=True), costs)
            return ev_rows[l], l
        return run

    def matched(costs):
        return max(0.0, level_for_tau(cost_matched_tau(costs.ratio)))

    rules = {
        "point": fixed(ev["pred"]),
        "point + tuned margin": tuned_threshold(cal["pred"], ev["pred"], H + MARGINS),
        "SCP 90%": fixed(scp.predict_interval(ev["pred"], LEVEL)[0]),
        "SCP cost-matched": level_rule(mondrian_lower(scp), matched),
        "Mondrian 90%": fixed(mon.predict_interval(ev["pred"], LEVEL)[0]),
        "Mondrian cost-matched": level_rule(mondrian_lower(mon), matched),
        "Mondrian tuned level": tuned_level(mondrian_lower(mon)),
    }
    if model in fp.get("cqr_lower", {}):
        cq = fp["cqr_lower"][model]
        cqr_lower = lambda frame, level: cq.lower(frame["q_lo"], level)
        rules["CQR 90%"] = fixed(ev["cqr_lower"])
        rules["CQR cost-matched"] = level_rule(cqr_lower, matched)
        rules["CQR tuned level"] = tuned_level(cqr_lower)
    return rules


def _model_free_rules(fp: dict, horizon: int) -> dict:
    cal, ev = fp["cal"]["lgbm"], fp["eval"]["lgbm"]
    H = horizon
    ages = np.arange(40, int(max(cal["cycle"].max(), ev["cycle"].max())) + 2, 2)
    cal_curve = threshold_outcomes(cal, -cal["cycle"], H - ages, H)          # alert when cycle >= age - H
    ev_curve = threshold_outcomes(ev, -ev["cycle"], H - ages, H).set_index("threshold")
    oracle = threshold_outcomes(ev, ev["true_rul"], [H], H).iloc[0]

    def age(costs):
        t = best_threshold(cal_curve, costs)
        return ev_curve.loc[t].rename(None), H - t                              # report the replacement age
    return {"age (tuned)": age, "oracle": lambda costs: (oracle, np.nan)}


def _dial_curves(fp: dict, horizon: int) -> dict:
    """Evaluation-engine sums for every setting of each rule's dial (no tuning involved).

    point margin   alert when pred <= H + margin          value = margin (cycles)
    SCP / Mondrian / CQR level   alert when lower <= H     value = confidence level
    age            alert when cycle >= age - H             value = replacement age (cycles)
    """
    H, out = horizon, {}
    for model in ("linear", "lgbm"):
        ev = fp["eval"][model]
        out[(model, "point margin")] = threshold_outcomes(ev, ev["pred"], H + MARGINS, H).assign(threshold=MARGINS)
        fams = {"SCP level": lambda l, c=fp["scp"][model]: c.predict_interval(ev["pred"], l)[0],
                "Mondrian level": lambda l, c=fp["mondrian"][model]: c.predict_interval(ev["pred"], l)[0]}
        if model in fp.get("cqr_lower", {}):
            fams["CQR level"] = lambda l, c=fp["cqr_lower"][model]: c.lower(ev["q_lo"], l)
        for fam, lower in fams.items():
            out[(model, fam)] = pd.concat([threshold_outcomes(ev, lower(l), [H], H).assign(threshold=l) for l in CURVE_LEVELS], ignore_index=True)
    ev = fp["eval"]["lgbm"]
    ages = np.arange(40, 561, 2)                                             # FD004 engines live up to 543 cycles
    out[("none", "age")] = threshold_outcomes(ev, -ev["cycle"], H - ages, H).assign(threshold=ages)
    return out


# --------------------------------------------------------------------------- metrics
def _coverage_rows(ev: pd.DataFrame, lower, upper, method: str, model: str) -> list[dict]:
    rows = []
    y, lo, hi = ev["y"].to_numpy(), np.asarray(lower), np.asarray(upper)
    for by, values, edges in (("pred", ev["pred"].to_numpy(), STRATA_EDGES), ("true", ev["true_rul"].to_numpy(), TRUE_EDGES)):
        strata = np.digitize(values, edges, right=True)
        labels = [f"<= {edges[0]}"] + [f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:])] + [f"> {edges[-1]}"]
        for k in range(len(edges) + 1):
            m = strata == k
            if not m.any():
                continue
            rows.append({"model": model, "method": method, "level": LEVEL, "by": by, "stratum": k, "label": labels[k],
                         "n": int(m.sum()), "engines": int(ev.loc[m, "unit"].nunique()),
                         "coverage": float(np.mean((lo[m] <= y[m]) & (y[m] <= hi[m]))),
                         "below": float(np.mean(y[m] < lo[m])), "above": float(np.mean(y[m] > hi[m])),
                         "mean_width": float(np.mean(hi[m] - lo[m]))})
    return rows


def run_subset_seed(subset: str, seed: int, k: int = N_FOLDS, ratios=COST_RATIOS, horizons=HORIZONS, verbose: bool = True):
    """Cross-fit one (subset, seed); return (accuracy, coverage, decisions) DataFrames."""
    t0 = time.time()
    train_df, _, _ = load_subset(subset)
    folds = engine_folds(train_df["unit"], k, seed)
    sums: dict = {}
    curve_sums: dict = {}
    params: dict = {}
    eval_frames: dict = {m: [] for m in ("linear", "lgbm")}
    cov_rows: list = []
    for f in range(k):
        tr_u, ca_u, ev_u = fold_roles(folds, f)
        fp = fold_predictions(train_df, subset, seed, tr_u, ca_u, ev_u)
        for model in ("linear", "lgbm"):
            ev = fp["eval"][model]
            eval_frames[model].append(ev)
            for method, calib in (("SCP", fp["scp"][model]), ("Mondrian", fp["mondrian"][model])):
                lo, hi = calib.predict_interval(ev["pred"], LEVEL)
                cov_rows += [dict(r, fold=f) for r in _coverage_rows(ev, lo, hi, method, model)]
            if model == "lgbm":
                cov_rows += [dict(r, fold=f) for r in _coverage_rows(ev, ev["cqr_lower"], ev["cqr_upper"], "CQR", model)]
        for H in horizons:
            for (model, family), curve in _dial_curves(fp, H).items():
                for row in curve.itertuples(index=False):
                    acc = curve_sums.setdefault((model, family, H, float(row.threshold)), np.zeros(5))
                    acc += np.array([getattr(row, c) for c in SUMS], float)
            rule_sets = {m: _rule_curves(fp, m, H) for m in ("linear", "lgbm")}
            rule_sets["none"] = _model_free_rules(fp, H)
            for model, rules in rule_sets.items():
                for rule, run in rules.items():
                    for r in ratios:
                        res, p = run(Costs.from_ratio(r))
                        key = (model, rule, H, float(r))
                        acc = sums.setdefault(key, np.zeros(5))
                        acc += res[["engines", "failures", "planned", "wasted_cycles", "cycles_flown"]].to_numpy(float)
                        params.setdefault(key, []).append(p)
    # decisions: pooled over the K evaluation folds = every engine once
    life = train_df.groupby("unit")["cycle"].max()
    dec = pd.DataFrame([
        {"subset": subset, "seed": seed, "model": m, "rule": rule, "horizon": H, "ratio": r,
         "engines": int(v[0]), "failures": int(v[1]), "planned": int(v[2]),
         "mean_wasted_cycles": v[3] / v[0], "cycles_flown": v[4],
         "cost_per_1000_cycles": 1000.0 * (r * v[1] + v[2]) / v[4],
         "param_mean": float(np.nanmean(params[(m, rule, H, r)])) if np.isfinite(params[(m, rule, H, r)]).any() else np.nan,
         "mean_life": float(life.mean())}
        for (m, rule, H, r), v in sums.items()
    ])
    curves = pd.DataFrame([
        {"subset": subset, "seed": seed, "model": m, "dial": fam, "horizon": H, "value": v,
         **dict(zip(SUMS, map(float, s)))}
        for (m, fam, H, v), s in curve_sums.items()
    ])
    # coverage: pool the folds (weighted by rows)
    cov = pd.DataFrame(cov_rows)
    for c in ("coverage", "below", "above", "mean_width"):
        cov[c + "_w"] = cov[c] * cov["n"]
    g = cov.groupby(["model", "method", "level", "by", "stratum", "label"], as_index=False)[["n", "engines", "coverage_w", "below_w", "above_w", "mean_width_w"]].sum()
    for c in ("coverage", "below", "above", "mean_width"):
        g[c] = g[c + "_w"] / g["n"]
    cov = g.drop(columns=[c for c in g.columns if c.endswith("_w")]).assign(subset=subset, seed=seed)
    # accuracy
    acc_rows = []
    for model, frames in eval_frames.items():
        ev = pd.concat(frames)
        near = ev["true_rul"] <= 50
        acc_rows.append({"subset": subset, "seed": seed, "model": model, "rows": len(ev), "engines": ev["unit"].nunique(),
                         "rmse_all": float(np.sqrt(np.mean((ev["pred"] - ev["y"]) ** 2))),
                         "rmse_near_failure": float(np.sqrt(np.mean((ev.loc[near, "pred"] - ev.loc[near, "y"]) ** 2))),
                         "rmse_far": float(np.sqrt(np.mean((ev.loc[~near, "pred"] - ev.loc[~near, "y"]) ** 2)))})
    if verbose:
        print(f"{subset} seed {seed}: {len(dec)} decision rows, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(acc_rows), cov, dec, curves
