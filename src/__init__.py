"""When does uncertainty pay? Prediction intervals and maintenance decisions on NASA C-MAPSS.

Modules
-------
config      constants (RUL cap, seed, window, folds, strata, cost ratios, euros) shared by everything else
load        read the raw txt files and assert their integrity
features    RUL labelling, regime normalisation, rolling features -> X, y, groups
split       train / calibration / test splits BY ENGINE
models      untuned Linear and LightGBM factories, LightGBM quantile trio
conformal   MAPIE v1 wrappers (split conformal, CQR), Mondrian split conformal, CQR lower bound at any level
evaluate    RMSE, NASA score, coverage, mean width, coverage per stratum (with miss direction)
decision    maintenance rules, removal/failure simulation, cost rate, threshold curves, sweep, break-even
crossfit    engine-level cross-fitting: every engine evaluated once -> accuracy, coverage, decisions, curves
summarize   thesis tables from the cross-fitting CSVs: euros, break-even per seed, win rates, value captured
newsvendor  the same decision layer for stocking under uncertain demand (retail generalisation)
m5          M5 Walmart loader (store x department) and a synthetic stand-in; retail runs the experiment
run         command line: python -m src.run crossfit ... | python -m src.run retail ...
results     one row per official-test run in results/runs.csv (week-1 notebooks)
experiment  glue for the week-1 notebooks: one (subset, seed) run end to end; decision_frame
plots       EDA, interval, coverage-by-stratum, cost-curve and trade-off figures
"""
