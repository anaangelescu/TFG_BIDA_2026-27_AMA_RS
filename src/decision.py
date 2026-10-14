"""Decision layer - turn RUL predictions into maintenance decisions and costs.

This is the part of the thesis that answers the business question: not "how far off is the
prediction?" but "what does acting on it cost?". Everything here is model-agnostic: it takes a
per-cycle frame of complete run-to-failure engines with the model outputs as columns.

Maintenance model (stated in the thesis, every parameter explicit)
------------------------------------------------------------------
* An engine fails right after its last recorded cycle T (true RUL at cycle t = T - t).
* At every check cycle t a rule may raise an ALERT. Planning a removal needs `horizon` cycles of
  notice (parts, slot, spare engine), so the first alert at cycle t_a books a removal at
  t_r = t_a + horizon.
* If t_r <= T the removal is PLANNED and the engine's remaining life T - t_r is wasted.
  Otherwise (no alert, or an alert with less than `horizon` cycles left) the engine FAILS at T.
* Cost: a planned removal costs `Costs.planned`, a failure costs `Costs.failure`. Wasted life is
  priced through the cost RATE (renewal-reward criterion, Barlow & Proschan): total cost divided
  by total cycles flown, so removing every engine early cannot look cheap.
* The cost ratio failure / planned is unknown for a synthetic dataset; it is swept, and the result
  is the break-even ratio at which one rule becomes cheaper than another.

Rules (each returns one boolean alert per row)
----------------------------------------------
point      pred <= horizon + margin       (margin = safety buffer, tuned per cost ratio)
lower      lower <= horizon               (lower bound of a prediction interval)
age        cycle >= replace_age - horizon (time-based maintenance, the business baseline)
oracle     true RUL <= horizon            (perfect information, the lower bound on cost)

With split conformal prediction the interval has the same half-width q for every engine, so the
`lower` rule is exactly the `point` rule with margin q. An interval can only beat a point rule with
a TUNED margin if its width adapts to the engine (CQR). That comparison is hypothesis 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

PLANNED, FAILURE = "planned", "failure"


# --------------------------------------------------------------------------- costs
@dataclass(frozen=True)
class Costs:
    """Cost of a planned removal and of an unplanned failure (any currency; ratio is what matters)."""

    planned: float = 1.0
    failure: float = 10.0

    @property
    def ratio(self) -> float:
        return self.failure / self.planned

    @classmethod
    def from_ratio(cls, ratio: float, planned: float = 1.0) -> "Costs":
        return cls(planned=planned, failure=ratio * planned)


def cost_matched_tau(ratio: float) -> float:
    """Quantile level a decision-maker should act on for a failure : planned cost ratio.

    Act when P(RUL <= horizon) >= tau with tau = c_planned / (c_planned + c_failure) (Elkan 2001;
    the newsvendor critical ratio). A guide for choosing the interval level, not an exact optimum
    under the cost-rate criterion above.
    """
    return 1.0 / (1.0 + ratio)


def level_for_tau(tau: float) -> float:
    """Confidence level of a central two-sided interval whose LOWER bound is the tau-quantile."""
    return 1.0 - 2.0 * tau


# --------------------------------------------------------------------------- frame helpers
def true_rul(frame: pd.DataFrame) -> pd.Series:
    """Uncapped true RUL per row: last recorded cycle of the engine minus the current cycle.

    `frame` must hold COMPLETE run-to-failure engines (rows from the training files), never the
    truncated official test file.
    """
    return frame.groupby("unit")["cycle"].transform("max") - frame["cycle"]


def _check_frame(frame: pd.DataFrame) -> None:
    missing = {"unit", "cycle"} - set(frame.columns)
    assert not missing, f"decision frame needs columns {missing}"
    per_unit = frame.groupby("unit")["cycle"].agg(["min", "max", "count"])
    assert (per_unit["min"] == 1).all() and (per_unit["max"] == per_unit["count"]).all(), (
        "decision frame must hold complete engines (cycles 1..T, no gaps)"
    )


# --------------------------------------------------------------------------- alert rules
def alert_point(frame: pd.DataFrame, horizon: int, margin: float = 0.0, col: str = "pred") -> np.ndarray:
    return (frame[col].to_numpy(float) <= horizon + margin)


def alert_lower(frame: pd.DataFrame, horizon: int, col: str = "lower") -> np.ndarray:
    return (frame[col].to_numpy(float) <= horizon)


def alert_age(frame: pd.DataFrame, horizon: int, replace_age: int) -> np.ndarray:
    return (frame["cycle"].to_numpy() >= replace_age - horizon)


def alert_oracle(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    return (true_rul(frame).to_numpy() <= horizon)


# --------------------------------------------------------------------------- simulation
def simulate(frame: pd.DataFrame, alert, horizon: int, check_every: int = 1) -> pd.DataFrame:
    """One row per engine: failure cycle, first alert, removal cycle, outcome, wasted and flown cycles.

    Alerts are only looked at on check cycles (cycle % check_every == 0).
    """
    _check_frame(frame)
    alert = np.asarray(alert, bool)
    assert alert.shape == (len(frame),), "one alert flag per row"
    df = pd.DataFrame({"unit": frame["unit"].to_numpy(), "cycle": frame["cycle"].to_numpy(), "alert": alert})
    df = df[df["cycle"] % check_every == 0] if check_every > 1 else df
    failure_cycle = frame.groupby("unit")["cycle"].max()
    alert_cycle = df[df["alert"]].groupby("unit")["cycle"].min().reindex(failure_cycle.index)

    removal = alert_cycle + horizon
    planned = removal.notna() & (removal <= failure_cycle)
    out = pd.DataFrame(
        {
            "failure_cycle": failure_cycle,
            "alert_cycle": alert_cycle,
            "removal_cycle": removal.where(planned),
            "outcome": np.where(planned, PLANNED, FAILURE),
        }
    )
    out["wasted_cycles"] = (failure_cycle - out["removal_cycle"]).where(planned, 0).astype(float)
    out["cycles_flown"] = out["removal_cycle"].where(planned, failure_cycle).astype(float)
    return out


def cost_summary(outcomes: pd.DataFrame, costs: Costs) -> dict[str, float]:
    """Fleet-level counts and costs. `cost_per_1000_cycles` is the headline number."""
    is_fail = outcomes["outcome"] == FAILURE
    n = len(outcomes)
    total = costs.failure * is_fail.sum() + costs.planned * (~is_fail).sum()
    flown = outcomes["cycles_flown"].sum()
    return {
        "engines": n,
        "failures": int(is_fail.sum()),
        "planned": int((~is_fail).sum()),
        "failure_rate": float(is_fail.mean()),
        "mean_wasted_cycles": float(outcomes["wasted_cycles"].mean()),
        "total_cost": float(total),
        "cycles_flown": float(flown),
        "cost_per_1000_cycles": float(1000.0 * total / flown),
    }


# --------------------------------------------------------------------------- tuning and sweeps
def tune(
    frame: pd.DataFrame,
    make_alert: Callable[[float], np.ndarray],
    params: Iterable[float],
    horizon: int,
    costs: Costs,
    check_every: int = 1,
) -> tuple[float, pd.DataFrame]:
    """Pick the parameter (margin, replacement age, ...) with the lowest cost rate on `frame`.

    Tune on the CALIBRATION engines and evaluate on the internal test engines, so the tuned point
    rule is a fair, well-chosen baseline rather than a straw man.
    """
    rows = []
    for p in params:
        summary = cost_summary(simulate(frame, make_alert(p), horizon, check_every), costs)
        rows.append({"param": p, **summary})
    table = pd.DataFrame(rows)
    best = table.loc[table["cost_per_1000_cycles"].idxmin(), "param"]
    return float(best), table


def sweep(
    rules: Mapping[str, Callable[[Costs], pd.DataFrame]],
    ratios: Iterable[float],
    planned: float = 1.0,
) -> pd.DataFrame:
    """Cost of every rule at every failure : planned cost ratio.

    Each rule is a callable costs -> per-engine outcomes (from `simulate`). Fixed rules ignore the
    costs; tuned rules (point with margin, age) re-tune for each ratio.
    """
    rows = []
    for ratio in ratios:
        costs = Costs.from_ratio(ratio, planned)
        for name, rule in rules.items():
            rows.append({"rule": name, "ratio": float(ratio), **cost_summary(rule(costs), costs)})
    return pd.DataFrame(rows)


def break_even(table: pd.DataFrame, baseline: str, challenger: str, metric: str = "cost_per_1000_cycles") -> float:
    """Smallest cost ratio at which `challenger` becomes cheaper than `baseline`.

    Linear interpolation on log(ratio) between the two sweep points that bracket the sign change.
    Returns 0.0 if the challenger is cheaper everywhere and NaN if it never is.
    """
    wide = table.pivot(index="ratio", columns="rule", values=metric).sort_index()
    gap = (wide[baseline] - wide[challenger]).to_numpy()  # > 0 means the challenger is cheaper
    ratios = wide.index.to_numpy(float)
    if gap[0] > 0:
        return 0.0
    cheaper = np.flatnonzero(gap > 0)
    if cheaper.size == 0:
        return float("nan")
    i = int(cheaper[0])
    x0, x1, g0, g1 = np.log(ratios[i - 1]), np.log(ratios[i]), gap[i - 1], gap[i]
    return float(np.exp(x0 + (x1 - x0) * (-g0) / (g1 - g0)))
