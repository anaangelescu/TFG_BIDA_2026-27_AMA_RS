"""RUL labelling, engine-level splits and the feature pipeline's fit-on-train-only behaviour."""
import numpy as np
import pandas as pd
import pytest

from src.config import KNOWN_CONSTANT_SENSORS, RUL_CAP, SENSOR_COLS, SETTING_COLS, find_data_dir
from src.features import FeaturePipeline, add_rul, sensor_variance_table
from src.split import split_engines


def toy_frame(lifetimes: dict[int, int], seed: int = 0) -> pd.DataFrame:
    """Run-to-failure frame with the raw C-MAPSS columns and noisy sensors."""
    rng = np.random.default_rng(seed)
    parts = []
    for unit, n in lifetimes.items():
        df = pd.DataFrame({"unit": unit, "cycle": np.arange(1, n + 1)})
        for c in SETTING_COLS:
            df[c] = 0.0
        for s in SENSOR_COLS:
            df[s] = rng.normal(size=n)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- RUL label
def test_rul_counts_down_to_zero_when_uncapped():
    df = add_rul(toy_frame({1: 5}), cap=10_000)
    assert df["rul"].tolist() == [4, 3, 2, 1, 0]


def test_rul_is_capped_and_ends_at_zero():
    df = add_rul(toy_frame({1: 200}), cap=125)
    assert df["rul"].max() == 125
    assert (df["rul"].iloc[:75] == 125).all()          # 200 - 125 = 75 early cycles saturate
    assert df["rul"].tolist()[-3:] == [2, 1, 0]
    assert (np.diff(df["rul"]) <= 0).all()             # never increases within an engine


def test_default_cap_is_the_named_constant():
    assert RUL_CAP == 125
    assert add_rul(toy_frame({1: 300}))["rul"].max() == RUL_CAP


def test_rul_is_computed_per_unit():
    df = add_rul(toy_frame({1: 3, 2: 4}), cap=10_000)
    assert df.loc[df["unit"] == 1, "rul"].tolist() == [2, 1, 0]
    assert df.loc[df["unit"] == 2, "rul"].tolist() == [3, 2, 1, 0]


# --------------------------------------------------------------------------- split by engine
def test_split_is_disjoint_complete_and_seeded():
    df = toy_frame({u: 10 + u % 7 for u in range(1, 101)})
    units = split_engines(df, seed=0)
    assert (len(units["train"]), len(units["cal"]), len(units["test"])) == (70, 15, 15)
    assert not set(units["train"]) & set(units["cal"]) and not set(units["train"]) & set(units["test"])
    assert set(units["train"]) | set(units["cal"]) | set(units["test"]) == set(range(1, 101))
    assert np.array_equal(split_engines(df, seed=0)["train"], units["train"])   # reproducible
    assert not np.array_equal(split_engines(df, seed=1)["train"], units["train"])  # seed matters


# --------------------------------------------------------------------------- feature pipeline
def test_constant_sensors_are_detected_and_dropped():
    df = toy_frame({1: 50, 2: 60})
    df["s_1"] = 518.67                                   # exactly constant
    df["s_6"] = np.where(df.index % 2 == 0, 21.60, 21.61)  # quantisation noise only
    pipe = FeaturePipeline("FD001").fit(df)
    assert pipe.dropped_sensors_ == ["s_1", "s_6"]
    assert not {"s_1", "s_6"} & set(pipe.feature_names)
    assert len(pipe.feature_names) == 19 * 3            # sensor + rolling mean + rolling std
    table = sensor_variance_table(df)
    assert table.loc[["s_1", "s_6"], "constant"].all() and not table.loc["s_2", "constant"]


def test_scaler_uses_training_statistics_only():
    train = toy_frame({1: 100, 2: 100}, seed=1)
    other = toy_frame({3: 100}, seed=2)
    other[SENSOR_COLS] += 5.0                            # shifted distribution
    pipe = FeaturePipeline("FD001").fit(train)
    z_train = pipe.transform(train)[pipe.kept_sensors_].mean().abs().max()
    z_other = pipe.transform(other)[pipe.kept_sensors_].mean().min()
    assert z_train < 0.3                                 # training data is centred by its own stats
    assert z_other > 3.0                                 # the shift is preserved: stats came from train


def test_rolling_features_are_computed_per_unit():
    df = toy_frame({1: 30, 2: 30})
    pipe = FeaturePipeline("FD001").fit(df)
    feat = pipe.transform(df)
    first = feat.groupby("unit").head(1)
    assert np.allclose(first["s_2_rs10"], 0.0)                     # one sample -> std 0, not NaN
    assert np.allclose(first["s_2_rm10"], first["s_2"])            # one sample -> mean = value
    assert not feat.isna().any().any()


# --------------------------------------------------------------------------- real data (skipped if absent)
def _data_available() -> bool:
    try:
        find_data_dir()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _data_available(), reason="C-MAPSS files not found")
def test_fd001_constant_sensors_match_the_literature():
    from src.load import load_subset

    train_df, _, _ = load_subset("FD001")
    pipe = FeaturePipeline("FD001").fit(train_df)
    assert pipe.dropped_sensors_ == KNOWN_CONSTANT_SENSORS
