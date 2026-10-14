"""Build app/planner.html: the removal planner, with the cross-fitting results embedded as JSON.

    python app/build_planner.py            # after `python -m src.run crossfit ...`

The page is self-contained (no server, no network): open app/planner.html in a browser.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import COST_RATIOS, HORIZONS, PLANNED_REMOVAL_EUR, RESULTS_DIR  # noqa: E402
from src.summarize import REALISTIC_BAND, break_even_runs, with_model_free  # noqa: E402

DIALS = ["point margin", "SCP level", "Mondrian level", "CQR level", "age"]
TABLE_RULES = ["age (tuned)", "point", "point + tuned margin", "SCP 90%", "Mondrian 90%", "CQR 90%",
               "Mondrian tuned level", "CQR tuned level", "oracle"]


def _curves(cur: pd.DataFrame) -> dict:
    out = {}
    cur = cur[cur["model"].isin(["lgbm", "none"])]
    subsets = sorted(cur["subset"].unique())
    for subset in subsets + ["ALL"]:
        c_s = cur if subset == "ALL" else cur[cur["subset"] == subset]
        for h in sorted(c_s["horizon"].unique()):
            block = {}
            for dial in DIALS:
                c = c_s[(c_s["horizon"] == h) & (c_s["dial"] == dial)]
                g = c.groupby("value")[["engines", "failures", "planned", "wasted_cycles", "cycles_flown"]].sum().reset_index()
                if dial == "age":                                   # drop ages beyond every engine's life
                    g = g[g["failures"] < g["engines"]]
                block[dial] = {"v": g["value"].round(4).tolist(), "n": g["engines"].astype(int).tolist(),
                               "f": g["failures"].astype(int).tolist(), "p": g["planned"].astype(int).tolist(),
                               "w": g["wasted_cycles"].astype(int).tolist(), "c": g["cycles_flown"].astype(int).tolist()}
            out[f"{subset}|{int(h)}"] = block
    return out


def _decisions(dec: pd.DataFrame) -> dict:
    """Median over runs of per-run numbers (cost index vs perfect information, saving vs age-based)."""
    d = with_model_free(dec)
    d = d[d["model"] == "lgbm"]
    key = ["subset", "seed", "horizon", "ratio"]
    ref = d[d["rule"].isin(["oracle", "age (tuned)"])].pivot_table(index=key, columns="rule", values="cost_per_1000_cycles").reset_index()
    d = d.merge(ref, on=key)
    d["index"] = 100 * d["cost_per_1000_cycles"] / d["oracle"]
    d["save"] = 1 - d["cost_per_1000_cycles"] / d["age (tuned)"]
    d["fail100"] = 100 * d["failures"] / d["engines"]
    out = {}
    for subset in sorted(d["subset"].unique()) + ["ALL"]:
        d_s = d if subset == "ALL" else d[d["subset"] == subset]
        for h in sorted(d_s["horizon"].unique()):
            med = d_s[d_s["horizon"] == h].groupby(["rule", "ratio"])[["index", "save", "fail100", "mean_wasted_cycles"]].median().reset_index()
            out[f"{subset}|{int(h)}"] = {
                r: {"ratio": g["ratio"].tolist(), "index": g["index"].round(2).tolist(), "save": g["save"].round(4).tolist(),
                    "fail100": g["fail100"].round(2).tolist(), "waste": g["mean_wasted_cycles"].round(2).tolist()}
                for r, g in med[med["rule"].isin(TABLE_RULES)].groupby("rule")}
    return out


def _break_even(dec: pd.DataFrame) -> dict:
    """Run-level break-even of each interval rule against the tuned margin, per fleet and notice period."""
    ch = ["SCP 90%", "Mondrian 90%", "CQR 90%", "Mondrian tuned level", "CQR tuned level"]
    out = {}
    per_subset = break_even_runs(dec, "point + tuned margin", ch, by=("subset", "horizon"))
    pooled = break_even_runs(dec, "point + tuned margin", ch, by=("horizon",)).assign(subset="ALL")
    for _, r in pd.concat([per_subset, pooled]).iterrows():
        fin = lambda v: None if not np.isfinite(v) else round(float(v), 3)
        out.setdefault(f"{r['subset']}|{int(r['horizon'])}", []).append(
            {"rule": r["challenger"], "runs": int(r["runs"]), "hits": int(r["runs_breaking_even"]),
             "median": fin(r["median"]), "q25": fin(r["q25"]), "q75": fin(r["q75"]), "median_hit": fin(r["median_if_breaking_even"])})
    return out


def main() -> None:
    dec = pd.read_csv(RESULTS_DIR / "crossfit_decisions.csv")
    cur = pd.read_csv(RESULTS_DIR / "crossfit_curves.csv")
    data = {
        "meta": {"planned_eur": PLANNED_REMOVAL_EUR, "band": list(REALISTIC_BAND), "ratios": list(map(float, COST_RATIOS)),
                 "horizons": list(HORIZONS), "seeds": int(dec["seed"].nunique()),
                 "subsets": sorted(dec["subset"].unique()),
                 "engines": {s: int(g.loc[g["seed"] == g["seed"].min(), "engines"].max()) for s, g in dec.groupby("subset")}},
        "curves": _curves(cur),
        "decisions": _decisions(dec),
        "break_even": _break_even(dec),
    }
    template = (ROOT / "app" / "planner_template.html").read_text()
    html = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (ROOT / "app" / "planner.html").write_text(html)
    print(f"app/planner.html written ({len(html) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
