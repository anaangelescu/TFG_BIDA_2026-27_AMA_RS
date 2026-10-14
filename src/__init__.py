"""Uncertainty-aware RUL prediction on NASA C-MAPSS.

Modules
-------
config      constants (RUL cap, seed, window, paths) shared by everything else
load        read the raw txt files and assert their integrity
features    RUL labelling, regime normalisation, rolling features -> X, y, groups
split       train / calibration / test splits BY ENGINE
models      untuned Linear and LightGBM factories
conformal   thin wrappers around MAPIE v1 (SplitConformalRegressor, CQR)
evaluate    RMSE, NASA score, coverage, mean width, coverage per stratum (with miss direction)
decision    maintenance rules, removal/failure simulation, cost rate, cost-ratio sweep, break-even
results     one row per run in results/runs.csv
experiment  glue: one (subset, seed) run end to end; decision_frame for the decision layer
plots       EDA and interval figures
"""
