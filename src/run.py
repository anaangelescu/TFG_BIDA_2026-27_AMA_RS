"""Command-line entry point for the long experiments.

    python -m src.run crossfit --subsets FD001 FD002 FD003 FD004 --seeds 0-9 --folds 5

Writes (appending one block per subset and seed, so an interrupted run keeps what it finished):
    results/crossfit_accuracy.csv   RMSE per model, all cycles and near failure
    results/crossfit_coverage.csv   coverage / miss direction / width per stratum and method
    results/crossfit_decisions.csv  failures, removals, wasted life, cost rate per rule and cost ratio
    results/crossfit_curves.csv     the same sums for every setting of each rule's dial (margin, level, age)
A (subset, seed) already present in crossfit_decisions.csv is skipped unless --force is given.

    python -m src.run retail --data synthetic --seeds 0-4      (or --data m5 once the Kaggle files are in place)

Writes results/retail_<data>_decisions.csv: cost per period, stockout share and fill rate per stocking
rule and stockout : overstock cost ratio (src/newsvendor.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import N_FOLDS, RESULTS_DIR

OUTPUTS = {"accuracy": "crossfit_accuracy.csv", "coverage": "crossfit_coverage.csv",
           "decisions": "crossfit_decisions.csv", "curves": "crossfit_curves.csv"}


def parse_seeds(text: str) -> list[int]:
    """'0-9' -> [0..9]; '0,3,5' -> [0, 3, 5]; '4' -> [4]."""
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def _done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    d = pd.read_csv(path, usecols=["subset", "seed"]).drop_duplicates()
    return set(zip(d["subset"], d["seed"].astype(int)))


def _append(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def crossfit(subsets, seeds, folds: int = N_FOLDS, out_dir: Path = RESULTS_DIR, force: bool = False) -> None:
    from src.crossfit import run_subset_seed

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / v for k, v in OUTPUTS.items()}
    done = set() if force else _done(paths["decisions"])
    for subset in subsets:
        for seed in seeds:
            if (subset, seed) in done:
                print(f"{subset} seed {seed}: already in {paths['decisions'].name}, skipped", flush=True)
                continue
            parts = dict(zip(("accuracy", "coverage", "decisions", "curves"), run_subset_seed(subset, seed, k=folds)))
            for name in ("accuracy", "coverage", "curves", "decisions"):        # decisions last = the "done" marker
                _append(parts[name], paths[name])


def retail(data: str, seeds, out_dir: Path = RESULTS_DIR) -> None:
    from src.retail import run_retail

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"retail_{data}_decisions.csv"
    table = pd.concat([run_retail(data, s) for s in seeds], ignore_index=True)
    table.to_csv(path, index=False)
    print(f"{len(table)} rows -> {path}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cf = sub.add_parser("crossfit", help="engine-level cross-fitting: accuracy, conditional coverage, decisions")
    cf.add_argument("--subsets", nargs="+", default=["FD001"])
    cf.add_argument("--seeds", default="0-9")
    cf.add_argument("--folds", type=int, default=N_FOLDS)
    cf.add_argument("--out", type=Path, default=RESULTS_DIR)
    cf.add_argument("--force", action="store_true", help="recompute (subset, seed) pairs already on disk")
    rt = sub.add_parser("retail", help="the same decision analysis on demand data (newsvendor): M5 or synthetic")
    rt.add_argument("--data", choices=["synthetic", "m5"], default="synthetic")
    rt.add_argument("--seeds", default="0-4")
    rt.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)
    if args.cmd == "crossfit":
        crossfit(args.subsets, parse_seeds(args.seeds), args.folds, args.out, args.force)
    elif args.cmd == "retail":
        retail(args.data, parse_seeds(args.seeds), args.out)


if __name__ == "__main__":
    main(sys.argv[1:])
