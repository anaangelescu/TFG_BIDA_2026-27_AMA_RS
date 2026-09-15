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
