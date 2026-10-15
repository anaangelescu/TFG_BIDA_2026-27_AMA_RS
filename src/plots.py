"""Figures for the EDA (step 2) and the interval diagnostics (step 6). Each function returns a Figure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, ROLL_WINDOW, RUL_CAP, SENSOR_COLS


def save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=120, bbox_inches="tight")


# --------------------------------------------------------------------------- EDA
def plot_lifetime_hist(train_df: pd.DataFrame, subset: str, bins: int = 25) -> plt.Figure:
    life = train_df.groupby("unit")["cycle"].max()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(life, bins=bins, color="steelblue", edgecolor="white")
    ax.axvline(life.median(), color="k", ls="--", label=f"median = {life.median():.0f}")
    ax.set(xlabel="engine lifetime (cycles)", ylabel="engines", title=f"{subset}: run-to-failure lifetimes (n={len(life)})")
    ax.legend()
    return fig


def plot_sensor_traces(train_df: pd.DataFrame, units, subset: str, sensors=SENSOR_COLS, ncols: int = 3) -> plt.Figure:
    nrows = int(np.ceil(len(sensors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.3 * nrows), sharex=True)
    for ax, s in zip(axes.ravel(), sensors):
        for u in units:
            e = train_df[train_df["unit"] == u]
            ax.plot(e["cycle"], e[s], lw=0.8, label=f"unit {u}")
        ax.set_title(s, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(sensors):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=6)
    fig.suptitle(f"{subset}: raw sensor readings of {len(units)} random engines over cycles", y=1.0)
    fig.tight_layout()
    return fig


def plot_rul_correlation(corr: pd.DataFrame, subset: str) -> plt.Figure:
    """`corr` : DataFrame indexed by sensor with one column per RUL definition (e.g. capped / uncapped)."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    corr.plot.bar(ax=ax, width=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(ylabel="Pearson r with RUL", title=f"{subset}: correlation of each sensor with RUL", ylim=(-1, 1))
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    return fig


def plot_settings_scatter(train_df: pd.DataFrame, subset: str, regime: pd.Series | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    c = regime if regime is not None else train_df["setting_3"]
    label = "KMeans regime" if regime is not None else "setting_3"
    sc = axes[0].scatter(train_df["setting_1"], train_df["setting_2"], c=c, s=4, cmap="tab10", alpha=0.6)
    axes[0].set(xlabel="setting_1 (altitude)", ylabel="setting_2 (Mach)")
    axes[1].scatter(train_df["setting_1"], train_df["setting_3"], c=c, s=4, cmap="tab10", alpha=0.6)
    axes[1].set(xlabel="setting_1 (altitude)", ylabel="setting_3 (TRA)")
    fig.colorbar(sc, ax=axes, label=label, fraction=0.03)
    fig.suptitle(f"{subset}: the three operating settings")
    return fig


def plot_before_after_regime(raw_df: pd.DataFrame, feat_df: pd.DataFrame, unit: int, sensor: str, subset: str, window: int = ROLL_WINDOW) -> plt.Figure:
    """Raw vs within-regime z-scored trace of one sensor for one engine (FD002/FD004)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    r = raw_df[raw_df["unit"] == unit].sort_values("cycle")
    f = feat_df[feat_df["unit"] == unit].sort_values("cycle")
    axes[0].scatter(r["cycle"], r[sensor], s=6, c=f["regime"], cmap="tab10")
    axes[0].set(title=f"{sensor} raw (colour = regime)", xlabel="cycle")
    axes[1].plot(f["cycle"], f[sensor], lw=0.8, label="z-scored within regime")
    axes[1].plot(f["cycle"], f[f"{sensor}_rm{window}"], lw=1.5, label=f"rolling mean ({window} cycles)")
    axes[1].set(title=f"{sensor} after regime normalisation", xlabel="cycle")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{subset}: unit {unit}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- step 6
def plot_pred_vs_true(y_true, y_pred, lower, upper, title: str) -> plt.Figure:
    """Engines sorted by true RUL; point prediction with its interval as a vertical bar."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    order = np.argsort(y_true)
    x = np.arange(len(order))
    inside = (lower <= y_true) & (y_true <= upper)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.vlines(x, lower[order], upper[order], color="lightsteelblue", lw=2, label="interval")
    ax.scatter(x, y_pred[order], s=12, color="tab:blue", label="prediction", zorder=3)
    ax.scatter(x, y_true[order], s=14, marker="x", c=np.where(inside[order], "k", "tab:red"), label="true (red = missed)", zorder=4)
    ax.set(xlabel="test engines sorted by true RUL", ylabel="RUL (cycles)", title=title, ylim=(-3, RUL_CAP + 8))
    ax.legend(loc="upper left", fontsize=8)
    return fig


def plot_calibration(levels, panels: dict[str, dict[str, list[float]]], title: str) -> plt.Figure:
    """Empirical coverage against nominal confidence level.

    `panels` maps a panel title (e.g. an evaluation set) to {method: coverages per level}.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5), squeeze=False)
    for ax, (panel, coverages) in zip(axes[0], panels.items()):
        ax.plot([0.4, 1], [0.4, 1], "k--", lw=1, label="perfect calibration")
        for name, cov in coverages.items():
            ax.plot(levels, cov, "o-", label=name)
        ax.set(xlabel="nominal confidence level", ylabel="empirical coverage", title=panel, xlim=(0.45, 1), ylim=(0.4, 1.02))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_width_vs_rul(y_true, widths: dict[str, np.ndarray], title: str, n_bins: int = 10) -> plt.Figure:
    """Interval width against true RUL: scatter plus a binned mean per method."""
    y_true = np.asarray(y_true, float)
    bins = np.linspace(0, RUL_CAP, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, w in widths.items():
        w = np.asarray(w, float)
        ax.scatter(y_true, w, s=8, alpha=0.35)
        which = np.clip(np.digitize(y_true, bins) - 1, 0, n_bins - 1)
        means = [w[which == b].mean() if np.any(which == b) else np.nan for b in range(n_bins)]
        ax.plot(centers, means, "o-", label=f"{name} (binned mean)")
    ax.set(xlabel="true RUL (cycles)", ylabel="interval width (cycles)", title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig


# --------------------------------------------------------------------------- conditional coverage and decisions (notebooks 06-07)
METHOD_COLORS = {"SCP": "#7f7f7f", "Mondrian": "#1f77b4", "CQR": "#d62728"}


def plot_coverage_by_stratum(tables: dict[str, pd.DataFrame], nominal: float, title: str) -> plt.Figure:
    """Grouped bars: coverage per stratum (columns) for each method (rows), one panel per table.

    `tables` maps a panel title (e.g. "by predicted RUL") to a method x stratum table from
    src.summarize.coverage_table. The dashed line is the nominal level.
    """
    fig, axes = plt.subplots(1, len(tables), figsize=(6.5 * len(tables), 4), squeeze=False)
    for ax, (panel, t) in zip(axes[0], tables.items()):
        x = np.arange(t.shape[1])
        width = 0.8 / len(t.index)
        for i, method in enumerate(t.index):
            ax.bar(x + (i - (len(t.index) - 1) / 2) * width, t.loc[method].to_numpy(), width,
                   label=method, color=METHOD_COLORS.get(method))
        ax.axhline(nominal, color="k", ls="--", lw=1, label=f"nominal {nominal:.0%}")
        ax.set_xticks(x, t.columns, rotation=0)
        ax.set(ylim=(0.4, 1.02), ylabel="coverage", xlabel="RUL stratum (cycles)", title=panel)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_cost_curves(table: pd.DataFrame, title: str, band: tuple[float, float] | None = None,
                     rules: list[str] | None = None, ylabel: str = "cost per 1,000 cycles (planned removals)") -> plt.Figure:
    """Cost rate against the failure : planned cost ratio (log axis), one line per rule.

    `table` is rules x ratios (src.summarize.cost_table). `band` shades the plausible ratio range.
    """
    rules = rules or list(table.index)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if band:
        ax.axvspan(*band, color="gold", alpha=0.15, label="plausible ratio band")
    for rule in rules:
        if rule in table.index:
            style = "k--" if rule == "oracle" else ("-" if "tuned" in rule or rule.startswith("age") else "o-")
            ax.plot(table.columns.astype(float), table.loc[rule].to_numpy(), style, label=rule, lw=1.6, ms=4)
    ax.set_xscale("log")
    ax.set(xlabel="failure : planned removal cost ratio", ylabel=ylabel, title=title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_dial_tradeoff(points: dict[str, pd.DataFrame], title: str) -> plt.Figure:
    """Failures per 100 engines against mean wasted cycles per engine, one curve per dial.

    Each frame has columns failures, wasted_cycles, engines (summed over folds), one row per dial
    setting. The curve closer to the origin offers better trade-offs at every cost ratio.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, c in points.items():
        ax.plot(c["wasted_cycles"] / c["engines"], 100 * c["failures"] / c["engines"], "o-", ms=3, label=name)
    ax.set(xlabel="mean wasted cycles per engine (life thrown away)", ylabel="failures per 100 engines", title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- README result figures (business view)
# Reference palette (light): surface, ink, recessive grid, categorical slots 1-4 in fixed order.
_SURFACE, _INK, _INK2, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
_BAND = "#f6e7c4"


def _style(ax):
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=9)
    ax.xaxis.label.set_color(_INK2); ax.yaxis.label.set_color(_INK2)
    ax.title.set_color(_INK)


def plot_break_even_eur(be: pd.DataFrame, band: tuple[float, float], planned_eur_m: float = 1.0,
                        order: list[str] | None = None) -> plt.Figure:
    """Failure cost (EUR M) above which each interval rule beats the tuned safety margin, one panel per notice period.

    `be` is src.summarize.break_even_runs(...): one row per (horizon, challenger) with median, q25, q75 (ratios,
    inf = never) and runs / runs_breaking_even. Dot = median run, line = middle half of the runs.
    """
    order = order or list(dict.fromkeys(be["challenger"]))
    horizons = sorted(be["horizon"].unique())
    lo, hi = 1.0, 100.0
    fig, axes = plt.subplots(1, len(horizons), figsize=(6.2 * len(horizons), 0.62 * len(order) + 1.6), sharey=True, squeeze=False)
    fig.patch.set_facecolor(_SURFACE)
    for ax, h in zip(axes[0], horizons):
        _style(ax)
        ax.axvspan(band[0] * planned_eur_m, band[1] * planned_eur_m, color=_BAND, zorder=0)
        ax.text(np.sqrt(band[0] * band[1]) * planned_eur_m, len(order) - 0.35, "plausible range", ha="center", va="bottom",
                fontsize=8, color=_INK2)
        for i, rule in enumerate(order):
            r = be[(be["horizon"] == h) & (be["challenger"] == rule)].iloc[0]
            y = len(order) - 1 - i
            med, q25, q75 = (float(r[c]) for c in ("median", "q25", "q75"))
            label_runs = f"{int(r['runs_breaking_even'])}/{int(r['runs'])} runs"
            if np.isfinite(med):
                x0, x1 = max(q25, lo / planned_eur_m), min(q75, hi / planned_eur_m)
                ax.plot([x0 * planned_eur_m, x1 * planned_eur_m], [y, y], color=_SERIES[0], lw=2, solid_capstyle="round", zorder=2)
                if not np.isfinite(q75):
                    ax.annotate("", xy=(hi, y), xytext=(x1 * planned_eur_m * 0.8, y),
                                arrowprops=dict(arrowstyle="->", color=_SERIES[0], lw=1.5))
                xm = max(med, lo / planned_eur_m) * planned_eur_m
                ax.plot(xm, y, "o", ms=8, color=_SERIES[0], mec=_SURFACE, mew=2, zorder=3)
                txt = f"€{med * planned_eur_m:.1f} M" if med > 0 else "always"
                ax.text(xm, y + 0.22, f"{txt}  ·  {label_runs}", fontsize=8.5, color=_INK, ha="center", va="bottom")
            else:
                ax.text(lo * 1.08, y, f"does not pay off in most runs  ·  {label_runs}", fontsize=8.5, color=_INK2, va="center")
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100], ["€1 M", "€2 M", "€5 M", "€10 M", "€20 M", "€50 M", "€100 M"])
        ax.set_ylim(-0.6, len(order) - 0.1)
        ax.set_yticks(range(len(order)), list(reversed(order)), color=_INK)
        ax.grid(axis="x", color=_GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("cost of one unplanned failure (planned removal = €1 M)")
        ax.set_title(f"{h} cycles of notice", loc="left", fontsize=11, fontweight="semibold")
    fig.suptitle("From which failure cost does acting on the interval beat a tuned safety margin?", x=0.01, ha="left",
                 fontsize=12.5, fontweight="semibold", color=_INK)
    fig.text(0.01, -0.03, "Dot: median of 40 runs (4 subsets × 10 seeds). Line: middle half of the runs; arrow = some runs never break even.",
             fontsize=8.5, color=_INK2)
    fig.tight_layout()
    return fig


def plot_rule_outcomes(rows: pd.DataFrame, title: str) -> plt.Figure:
    """Two panels side by side: failures per 100 engines and wasted cycles per engine, one bar per rule.

    `rows` has columns rule, fail100, waste (one row per rule, in display order).
    """
    n = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.48 * n + 1.5), sharey=True)
    fig.patch.set_facecolor(_SURFACE)
    y = np.arange(n)[::-1]
    for ax, col, label, color in ((axes[0], "fail100", "failures per 100 engines", _SERIES[1]),
                                  (axes[1], "waste", "cycles of life wasted per engine", _SERIES[0])):
        _style(ax)
        vals = rows[col].to_numpy(float)
        ax.barh(y, vals, height=0.62, color=color, edgecolor=_SURFACE, linewidth=2)
        top = max(vals.max(), 1.0)
        for yi, v in zip(y, vals):
            ax.text(v + top * 0.015, yi, f"{v:.1f}", va="center", fontsize=8.5, color=_INK)
        ax.set_xlim(0, top * 1.15)
        ax.set_title(label, loc="left", fontsize=10.5, color=_INK)
        ax.grid(axis="x", color=_GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0].set_yticks(y, rows["rule"], color=_INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, fontweight="semibold", color=_INK)
    fig.tight_layout()
    return fig


def plot_retail_relative(table: pd.DataFrame, baseline: str, title: str, cap: float = 2.0) -> plt.Figure:
    """Cost of each stocking rule relative to `baseline` (= 1) across stockout : overstock ratios, log x.

    `table` is rules x ratios (median cost per day). Lines above `cap` are clipped and labelled.
    """
    rel = table.div(table.loc[baseline], axis=1)
    x = rel.columns.astype(float)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    fig.patch.set_facecolor(_SURFACE); _style(ax)
    ends = {}
    for color, rule in zip(_SERIES, rel.index):
        v = rel.loc[rule].to_numpy(float)
        vc = np.minimum(v, cap)
        ax.plot(x, vc, "-o", color=color, lw=2, ms=4.5, mec=_SURFACE, mew=1)
        ends[rule] = (vc[-1], rule if v[-1] <= cap else f"{rule} (off scale: {v[-1]:.1f}× at {x[-1]:.0f} : 1)")
    # direct labels at the line ends, nudged apart so they never overlap
    placed, gap = [], 0.06
    for rule, (y, lab) in sorted(ends.items(), key=lambda kv: kv[1][0]):
        y_lab = max(y, placed[-1] + gap) if placed else y
        placed.append(y_lab)
        ax.annotate(lab, (x[-1], y), xytext=(x[-1] * 1.12, y_lab), textcoords="data", va="center", fontsize=8.5, color=_INK)
    ax.axhline(1.0, color=_INK2, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(list(x), [f"{v:g}" for v in x])
    ax.set_ylim(0.7, cap * 1.03)
    ax.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], ["−20 %", "same", "+20 %", "+40 %", "+60 %", "+80 %", "+100 %"])
    ax.set_xlabel("stockout : overstock cost ratio")
    ax.set_ylabel(f"cost per day vs {baseline}")
    ax.grid(color=_GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold")
    ax.set_xlim(x[0] * 0.9, x[-1] * 1.1)
    ax.set_ylim(min(0.7, min(p for p in placed) - 0.08), cap * 1.03)
    fig.tight_layout()
    return fig
