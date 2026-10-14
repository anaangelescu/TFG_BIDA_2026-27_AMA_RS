"""Cross-fitting bookkeeping: every engine is evaluated exactly once and never trains its own model."""
import numpy as np
import pandas as pd

from src.crossfit import engine_folds, fold_roles
from src.decision import Costs, best_threshold, threshold_outcomes
from src.run import parse_seeds


def test_folds_partition_the_engines():
    units = np.repeat(np.arange(1, 101), 3)                 # 100 engines, several rows each
    folds = engine_folds(units, k=5, seed=7)
    allu = np.concatenate(folds)
    assert sorted(allu) == list(range(1, 101)) and len(allu) == 100
    assert [len(f) for f in folds] == [20] * 5
    assert not np.array_equal(np.concatenate(engine_folds(units, 5, 8)), allu)   # seed changes the folds


def test_each_fold_keeps_roles_disjoint_and_evaluates_every_engine_once():
    folds = engine_folds(np.arange(1, 53), k=5, seed=0)
    evaluated = []
    for f in range(5):
        tr, ca, ev = fold_roles(folds, f)
        assert not (set(tr) & set(ca)) and not (set(tr) & set(ev)) and not (set(ca) & set(ev))
        assert set(tr) | set(ca) | set(ev) == set(range(1, 53))
        evaluated += list(ev)
    assert sorted(evaluated) == list(range(1, 53))


def test_parse_seeds():
    assert parse_seeds("0-3") == [0, 1, 2, 3]
    assert parse_seeds("1,4,7-8") == [1, 4, 7, 8]


def test_curve_sums_add_up_across_folds():
    # Summing threshold curves over disjoint engine groups equals the curve on all engines at once.
    rng = np.random.default_rng(0)
    parts = []
    for unit, T in enumerate([90, 120, 150, 200, 170, 130], start=1):
        c = np.arange(1, T + 1)
        parts.append(pd.DataFrame({"unit": unit, "cycle": c, "pred": (T - c) + rng.normal(0, 6, T)}))
    f = pd.concat(parts, ignore_index=True)
    thr = np.arange(10, 40)
    whole = threshold_outcomes(f, f["pred"], thr, 10)
    a, b = f[f["unit"] <= 3], f[f["unit"] > 3]
    parts_sum = threshold_outcomes(a, a["pred"], thr, 10).iloc[:, 1:] + threshold_outcomes(b, b["pred"], thr, 10).iloc[:, 1:]
    pd.testing.assert_frame_equal(whole.iloc[:, 1:], parts_sum)
    assert best_threshold(whole, Costs(1, 100)) >= best_threshold(whole, Costs(1, 2))   # dearer failures -> earlier alerts
