"""Project-wide constants. Every other module imports from here; nothing is hard-coded twice."""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]  # the v1/ folder (repository root)
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RUNS_CSV = RESULTS_DIR / "runs.csv"

# --------------------------------------------------------------------------- protocol constants
# Piece-wise linear RUL target: RUL = min(max_cycle_of_unit - cycle, RUL_CAP).
# The cap is stated once here because this choice alone moves RMSE by several cycles.
RUL_CAP = 125

# Default seed for every random operation (splits, KMeans, LightGBM subsampling, engine sampling).
SEED = 0

# Rolling-statistics window (in cycles) used by features.py.
ROLL_WINDOW = 10

# Number of operating-condition regimes for FD002 / FD004 (six known conditions).
N_REGIMES = 6

# Nominal confidence level of the conformal intervals.
CONFIDENCE_LEVEL = 0.90

# Engine-level split fractions (train / calibration / internal test).
FRAC_TRAIN, FRAC_CAL, FRAC_TEST = 0.70, 0.15, 0.15

# --------------------------------------------------------------------------- decision experiment (src/crossfit.py)
# Engine-level cross-fitting: fold k is evaluated, fold k+1 calibrates (conformal + rule tuning), the rest train.
N_FOLDS = 5
# Predicted-RUL strata for Mondrian calibration and for coverage reporting (cycles).
STRATA_EDGES = (25, 50, 75, 100)
# Failure : planned-removal cost ratios swept in the decision analysis (synthetic data -> assumptions).
COST_RATIOS = (1, 1.5, 2, 3, 5, 10, 20, 50, 100)
# Cycles of notice needed to plan a removal (parts, slot, spare engine).
HORIZONS = (10, 30)
# Assumed cost of one planned removal (shop visit), in euros, used ONLY to express results in euros.
# CFM56-7B planned restoration US$1.1-1.3 million (Aircraft Commerce 2008) ~ EUR 1.0-1.15 million.
PLANNED_REMOVAL_EUR = 1_000_000

# --------------------------------------------------------------------------- data layout
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
MULTI_REGIME_SUBSETS = {"FD002", "FD004"}  # six operating conditions -> regime normalisation

ID_COLS = ["unit", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ID_COLS + SETTING_COLS + SENSOR_COLS  # the 26 raw columns, in file order

# Sensors that are constant in the single-condition subsets (FD001 / FD003). features.py
# DETECTS constant sensors from the training engines; this list only documents the expectation
# and is checked by the tests.
KNOWN_CONSTANT_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]

# Name of the NASA folder as it comes out of the official zip.
NASA_FOLDER = "6. Turbofan Engine Degradation Simulation Data Set-CMAPSSData"


def find_data_dir() -> Path:
    """Locate the folder holding the 12 raw txt files.

    Search order (first hit wins):
      1. $CMAPSS_DATA_DIR
      2. <repo>/data/raw                  (layout described in README §6)
      3. <repo>/../data/raw
      4. <repo>/../data/<NASA_FOLDER>     (the zip unpacked next to the repo, as on this machine)
      5. any sub-folder of <repo>/../data that contains train_FD001.txt
    """
    candidates: list[Path] = []
    if os.environ.get("CMAPSS_DATA_DIR"):
        candidates.append(Path(os.environ["CMAPSS_DATA_DIR"]))
    candidates += [
        ROOT / "data" / "raw",
        ROOT.parent / "data" / "raw",
        ROOT.parent / "data" / NASA_FOLDER,
    ]
    sibling_data = ROOT.parent / "data"
    if sibling_data.is_dir():
        candidates += sorted(p for p in sibling_data.iterdir() if p.is_dir())
    for cand in candidates:
        if (cand / "train_FD001.txt").is_file():
            return cand
    raise FileNotFoundError(
        "Could not find the C-MAPSS txt files. Set CMAPSS_DATA_DIR or unzip the NASA archive "
        f"into {ROOT / 'data' / 'raw'}."
    )
