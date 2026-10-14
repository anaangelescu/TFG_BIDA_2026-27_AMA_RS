"""The same decision layer outside aviation: stocking a product under uncertain demand (the newsvendor).

Why this module exists (tutor's point: "the framework is the product"): the engine question and the
retail question have the same shape.

    engines                                  retail
    -------                                  ------
    unplanned failure (act too late)         stockout: lost margin (stocked too little)
    early removal (act too early)            overstock: holding / markdown (stocked too much)
    act when the LOWER bound <= horizon      stock up to an UPPER quantile of demand
    tau = 1 / (1 + ratio)                    tau = ratio / (1 + ratio),  ratio = c_stockout / c_overstock

The cost-optimal stock is the tau-quantile of demand (Arrow, Harris & Marschak 1951), and the expected
newsvendor cost is (c_under + c_over) x the pinball loss at tau (Gneiting 2011). So "which forecast is
best" depends on the cost ratio exactly as in the engine case, and the same sweep / break-even analysis
applies. Rules compared on any demand data (src/m5.py loads the M5 Walmart data; tests use synthetic):

    point              stock the point forecast
    point + buffer     stock point + b, b tuned on calibration days (the honest baseline, as on engines)
    conformal tau      stock point + the conformal tau-quantile of calibration residuals (one dial: tau)
    quantile tau       stock a conformalised quantile model at tau (adaptive width, the CQR analogue)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def critical_ratio(ratio: float) -> float:
    """Cost-optimal quantile level for stocking when a stockout costs `ratio` times an overstock."""
    return ratio / (1.0 + ratio)


def newsvendor_cost(demand, stock, c_under: float, c_over: float) -> np.ndarray:
    """Per-period cost: c_under per unit short, c_over per unit left over."""
    d, q = np.asarray(demand, float), np.asarray(stock, float)
    return c_under * np.maximum(d - q, 0.0) + c_over * np.maximum(q - d, 0.0)


def pinball(demand, q, tau: float) -> np.ndarray:
    d, q = np.asarray(demand, float), np.asarray(q, float)
    return np.maximum(tau * (d - q), (tau - 1) * (d - q))


def upper_conformal_shift(residuals, tau: float) -> float:
    """One-sided split-conformal shift: the ceil((n + 1) tau)-th smallest residual (demand - forecast).

    Stocking forecast + shift covers demand with probability >= tau under exchangeability.
    tau = 0 gives -inf (stock nothing beyond the clip); a too-small calibration set gives +inf.
    """
    r = np.sort(np.asarray(residuals, float))
    if tau <= 0:
        return -np.inf
    k = int(np.ceil((len(r) + 1) * tau))
    return float(r[k - 1]) if k <= len(r) else np.inf


def evaluate_stock(demand, stock, ratio: float, c_over: float = 1.0) -> dict[str, float]:
    """Fleet-style summary: cost per period (units of one overstocked item), fill rate, stockout share."""
    d, q = np.asarray(demand, float), np.maximum(np.asarray(stock, float), 0.0)
    cost = newsvendor_cost(d, q, ratio * c_over, c_over)
    return {"cost_per_period": float(cost.mean()), "stockout_share": float(np.mean(d > q)),
            "fill_rate": float(np.minimum(d, q).sum() / max(d.sum(), 1e-12)),
            "mean_overstock": float(np.maximum(q - d, 0).mean())}


def decision_sweep(cal: pd.DataFrame, test: pd.DataFrame, ratios, buffers=None, q_cols: dict | None = None) -> pd.DataFrame:
    """Compare the stocking rules on `test` periods, every dial tuned or calibrated on `cal` periods.

    Both frames need columns `demand` and `point`; `q_cols` maps a quantile level tau to a column with
    a quantile forecast at that level (optional, for the 'quantile tau' rule; the nearest level is used).
    """
    buffers = np.arange(0, 51) if buffers is None else np.asarray(buffers, float)
    res_cal = (cal["demand"] - cal["point"]).to_numpy()
    rows = []
    for r in ratios:
        tau = critical_ratio(r)

        def add(rule, stock_test, dial=np.nan):
            rows.append({"ratio": float(r), "tau": tau, "rule": rule, "dial": dial, **evaluate_stock(test["demand"], stock_test, r)})

        add("point", test["point"])
        cal_costs = [newsvendor_cost(cal["demand"], np.maximum(cal["point"] + b, 0), r, 1.0).mean() for b in buffers]
        b = float(buffers[int(np.argmin(cal_costs))])
        add("point + tuned buffer", test["point"] + b, b)
        add("conformal tau", test["point"] + upper_conformal_shift(res_cal, tau), tau)
        if q_cols:
            lvl = min(q_cols, key=lambda t: abs(t - tau))
            # conformalise the nearest quantile model: shift so that P(demand <= q + s) >= tau on cal
            s = upper_conformal_shift((cal["demand"] - cal[q_cols[lvl]]).to_numpy(), tau)
            add("quantile tau", test[q_cols[lvl]] + s, lvl)
    return pd.DataFrame(rows)
