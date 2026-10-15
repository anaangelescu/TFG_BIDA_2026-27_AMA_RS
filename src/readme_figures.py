"""Draw the result figures shown in the README (business view) from results/*.csv.

    python -m src.readme_figures

Writes figures/readme_break_even_eur.png, readme_rule_outcomes.png and readme_retail.png.
(figures/readme_planner.png is a screenshot of app/planner.html.)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from src import summarize as S
from src.config import RESULTS_DIR
from src.plots import plot_break_even_eur, plot_rule_outcomes, plot_retail_relative, save

NAMES = {"age (tuned)": "Replace at a fixed age", "point": "Point prediction, no margin", "point + tuned margin": "Point + tuned safety margin",
         "SCP 90%": "Split conformal 90 %", "Mondrian 90%": "Mondrian 90 %", "CQR 90%": "CQR 90 %",
         "Mondrian tuned level": "Mondrian, tuned level", "CQR tuned level": "CQR, tuned level"}


def main() -> None:
    dec = S.load()["decisions"]
    order = ["SCP 90%", "Mondrian 90%", "Mondrian tuned level", "CQR tuned level", "CQR 90%"]
    be = S.break_even_runs(dec, "point + tuned margin", order).assign(challenger=lambda t: t["challenger"].map(NAMES))
    save(plot_break_even_eur(be, S.REALISTIC_BAND, order=[NAMES[o] for o in order]), "readme_break_even_eur.png")
    d = S.with_model_free(dec)
    d = d[(d.model == "lgbm") & (d.horizon == 10) & (d.ratio == 5.0)].assign(fail100=lambda t: 100 * t.failures / t.engines)
    rows = (d.groupby("rule")[["fail100", "mean_wasted_cycles"]].median().reindex(list(NAMES)).reset_index()
            .rename(columns={"mean_wasted_cycles": "waste"}).assign(rule=lambda t: t["rule"].map(NAMES)))
    save(plot_rule_outcomes(rows, "What each rule does to 100 engines (10 cycles of notice, failure = 5 × a planned removal)"), "readme_rule_outcomes.png")
    ret = pd.read_csv(RESULTS_DIR / "retail_synthetic_decisions.csv")
    t = ret.pivot_table(index="rule", columns="ratio", values="cost_per_period", aggfunc="median").reindex(["point + tuned buffer", "conformal tau", "quantile tau", "point"])
    t.index = ["tuned buffer", "constant conformal shift", "conformalised quantile model", "point forecast"]
    save(plot_retail_relative(t, "tuned buffer", "Retail (synthetic demand): cost of each stocking rule vs a tuned buffer"), "readme_retail.png")
    print("figures/readme_*.png written")


if __name__ == "__main__":
    main()
