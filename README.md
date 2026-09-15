# tfg-rul-uncertainty

**How much can we trust a prediction of engine life?**
Calibrated uncertainty for turbofan Remaining Useful Life (RUL) prediction on the NASA C-MAPSS dataset.

Bachelor's thesis (TFG) — Ana Maria Angelescu & Rune Snauwaert, Business Intelligence and Data Analysis, La Salle Barcelona 2026–2027.

---

## 1. What this repository is

Most published work on C-MAPSS predicts that an engine has "34 cycles left" and stops there. This project adds a confidence interval to that prediction using **conformal prediction**, and measures whether the interval can be trusted.

**Research question.** Which model produces the most trustworthy confidence intervals for remaining useful life — and is it the same model that produces the most accurate point predictions?

**Hypothesis.** Accuracy and reliability do not coincide. Conformal prediction will give all models valid coverage (~90% of true RUL values inside the 90% interval), but interval widths will differ substantially, and the model with the lowest RMSE will not necessarily be the one with the narrowest reliable intervals.

**Sub-questions.**
1. Does coverage hold in the region that matters — engines close to failure?
2. Do intervals narrow as an engine approaches failure?
3. If maintenance is triggered by the interval's lower bound instead of the point prediction, how do false alarms and missed failures change?

**Models.** Linear regression (floor) · LightGBM · LSTM — all under one identical protocol.
**Uncertainty methods.** Split conformal prediction (SCP) · Conformalized quantile regression (CQR), via [MAPIE](https://mapie.readthedocs.io) v1.

---

## 2. Positioning in the literature

Conformal prediction for RUL is not new; this project extends it.

| Prior work | Did | Left open (this project) |
|---|---|---|
| Javanmardi & Hüllermeier (2023), *Conformal Prediction Intervals for RUL Estimation*, Int. J. PHM (arXiv 2212.14612) | First CP for RUL; conformalized a deep CNN and gradient boosting on C-MAPSS; SCP, CQR, non-exchangeable variants | No cross-family accuracy-vs-reliability comparison; no near-failure conditional coverage; no decision simulation |
| Sensors 2026 (doi 10.3390/s26072249) | LSTM quantile regression + CQR on C-MAPSS, multi-seed | Single model family |
| RESS 2026 (S0951832026005740) | Isotonic calibration + SCP on C-MAPSS and batteries | Single proposed architecture |

**Contribution.** A controlled cross-family comparison under one protocol; conditional coverage near failure; interval behaviour as failure approaches; an alert-policy simulation converting intervals into decisions. Javanmardi's gradient-boosting result is reproduced first as a sanity check.

---

## 3. Data

### 3.1 Source
NASA Prognostics Center of Excellence, *Turbofan Engine Degradation Simulation Data Set* (item 6 of the Prognostics Data Repository).

- Download: `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip` (~12 MB)
- Landing page: NASA PCoE Data Set Repository (mirrored at `data.phmsociety.org/nasa/`)
- Citation: A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
- Paper: A. Saxena, K. Goebel, D. Simon, N. Eklund (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08.

Unzip into `data/raw/`. The raw files are never modified.

### 3.2 Integrity check
After download, verify:

| File | Rows | Units | File | Rows | Units |
|---|---|---|---|---|---|
| train_FD001.txt | 20,631 | 100 | test_FD001.txt | 13,096 | 100 |
| train_FD002.txt | 53,759 | 260 | test_FD002.txt | 33,991 | 259 |
| train_FD003.txt | 24,720 | 100 | test_FD003.txt | 16,596 | 100 |
| train_FD004.txt | 61,249 | 249 | test_FD004.txt | 41,214 | 248 |

`RUL_FD00X.txt` contains one true RUL per test engine. `src/load.py` asserts these counts.

### 3.3 Subsets
| Subset | Operating conditions | Fault modes | Difficulty |
|---|---|---|---|
| FD001 | 1 | 1 (HPC) | baseline |
| FD002 | 6 | 1 (HPC) | conditions vary |
| FD003 | 1 | 2 (HPC, fan) | fault modes vary |
| FD004 | 6 | 2 (HPC, fan) | hardest |

### 3.4 File format
Space-separated text, 26 columns, no header:

| Col | Name | Meaning |
|---|---|---|
| 1 | `unit` | Engine ID |
| 2 | `cycle` | Flight cycle index |
| 3–5 | `setting_1..3` | Altitude, Mach number, throttle resolver angle |
| 6–26 | `s_1..s_21` | Sensor readings |

| Sensor | Symbol | Meaning | Note |
|---|---|---|---|
| s_1 | T2 | Fan inlet temperature | constant (FD001/3) |
| s_2 | T24 | LPC outlet temperature | |
| s_3 | T30 | HPC outlet temperature | |
| s_4 | T50 | LPT outlet temperature | |
| s_5 | P2 | Fan inlet pressure | constant (FD001/3) |
| s_6 | P15 | Bypass-duct pressure | near-constant |
| s_7 | P30 | HPC outlet pressure | |
| s_8 | Nf | Physical fan speed | |
| s_9 | Nc | Physical core speed | |
| s_10 | epr | Engine pressure ratio | constant (FD001/3) |
| s_11 | Ps30 | HPC outlet static pressure | |
| s_12 | phi | Fuel flow / Ps30 | |
| s_13 | NRf | Corrected fan speed | |
| s_14 | NRc | Corrected core speed | |
| s_15 | BPR | Bypass ratio | |
| s_16 | farB | Burner fuel–air ratio | constant (FD001/3) |
| s_17 | htBleed | Bleed enthalpy | |
| s_18 | Nf_dmd | Demanded fan speed | constant |
| s_19 | PCNfR_dmd | Demanded corrected fan speed | constant |
| s_20 | W31 | HPT coolant bleed | |
| s_21 | W32 | LPT coolant bleed | |

**Not used:** N-CMAPSS (repository item 17, ~10 GB) — out of scope. PHM08 Challenge (item 7) — optional extra run-to-failure trajectories if access is granted; never required.

---

## 4. Protocol — rules that are never broken

These are fixed for the entire project and stated in the thesis. Changing any of them invalidates comparability between runs.

1. **Split by engine, never by row.** All train / conformalization / test splits use `groups = unit` (`GroupShuffleSplit`, `GroupKFold`). Rows within an engine are correlated; row-level splits produce leaked, inflated scores.
2. **Three disjoint engine groups** from each training file: 70% train, 15% conformalization (calibration), 15% internal test. The official `test_FD00X.txt` + `RUL_FD00X.txt` is a fourth, final held-out set.
3. **RUL label** = `min(max_cycle(unit) − cycle, 125)`. The cap of 125 is a named constant (`RUL_CAP`). Stated explicitly because this choice alone moves RMSE by several cycles across the literature.
4. **Constant sensors dropped** for FD001/FD003: s_1, s_5, s_6, s_10, s_16, s_18, s_19 (14 sensors remain).
5. **Operating-condition normalization** for FD002/FD004: the three settings are clustered into 6 regimes (KMeans, fixed seed); each sensor is z-scored within its regime.
6. **Scalers fitted on training engines only.**
7. **LSTM windows** of length 30, stride 1, target = RUL at the last cycle of the window.
8. **Official-test predictions** use the last cycle (or last window) of each test engine.
9. **Seeds.** Every random operation takes an explicit seed. Every experiment runs for seeds 0–9.
10. **Every run is logged** as one row in `results/runs.csv`: `subset, model, uq_method, seed, rmse, nasa_score, coverage, mean_width, coverage_near_failure, n_test, timestamp, git_hash`.
    *Week-1 implementation (`src/results.py`):* `subset, model, seed, rmse, nasa_score, coverage, mean_width, n_test` — the UQ method is folded into `model` (`lgbm_scp`, `lgbm_cqr`); the remaining columns are added when the full grid starts. Only official-test results are logged; internal-test numbers live in the notebooks.
11. **Conformalization on separate engines.** Rows within an engine are a time series and not exchangeable; engines are approximately exchangeable. Calibration therefore uses whole engines, evaluation is per engine, and this assumption is stated in the thesis.

---

## 5. Metrics

| Metric | Definition | Direction |
|---|---|---|
| RMSE | Root mean squared error of point RUL, in cycles | lower |
| NASA score | `Σ (exp(−d/13) − 1)` for `d < 0`, `Σ (exp(d/10) − 1)` for `d ≥ 0`, with `d = pred − true`; penalizes late predictions more | lower |
| Coverage | Fraction of test engines whose true RUL lies inside the interval | ≈ nominal (0.90) |
| Mean width | Mean interval width in cycles | lower, *only if coverage holds* |
| Coverage near failure | Coverage restricted to engines with true RUL < 50 | ≈ nominal |
| Width-vs-RUL curve | Interval width as a function of true RUL | should decrease as RUL → 0 |
| Training time | Wall-clock seconds per fit | reported, not optimized |

Reference points on FD001 official test: RMSE ≈ 12–14 for strong deep models, ≈ 16–19 for classical ML.

**Statistical testing.** 10 seeds per model per subset. Pairwise: Wilcoxon signed-rank test (paired). Multi-model across subsets: Friedman test + critical-difference diagram (Demšar 2006). Report p-values and effect sizes, not just means.

---

## 6. Repository layout

```
v1/                               # this repository (the raw data lives OUTSIDE it, see §7)
├── src/                          # installable package: `pip install -e .` then `import src.<module>`
│   ├── config.py                 # RUL_CAP=125, SEED=0, ROLL_WINDOW=10, N_REGIMES=6, paths, data-dir lookup
│   ├── load.py                   # load_subset(name) -> train_df, test_df, rul_df; asserts counts; `python -m src.load --check`
│   ├── features.py               # add_rul, FeaturePipeline (regimes, z-score, constant sensors, rolling stats), build_features -> X, y, groups
│   ├── split.py                  # split_engines: 70/15/15 BY ENGINE with GroupShuffleSplit
│   ├── models.py                 # LinearRegression, untuned LGBMRegressor, LightGBM quantile models
│   ├── conformal.py              # fit_scp / fit_cqr around MAPIE v1, interval prediction clipped to [0, RUL_CAP]
│   ├── evaluate.py               # rmse, nasa_score, coverage, mean_width
│   ├── results.py                # append_run -> results/runs.csv (one row per subset, model, seed)
│   ├── experiment.py             # prepare / fit_point_models / fit_conformal / run_seed: one (subset, seed) end to end
│   └── plots.py                  # EDA figures and the interval diagnostics
├── notebooks/
│   ├── 01_eda.ipynb              # step 2: FD001 then FD002
│   ├── 02_baselines.ipynb        # steps 3-4: pipeline check, Linear vs LightGBM, official test
│   ├── 03_conformal.ipynb        # step 5: SCP vs CQR, what happens inside each
│   ├── 04_interval_plots.ipynb   # step 6: pred-vs-true with intervals, calibration curve, width vs RUL
│   ├── 05_seed_variability.ipynb # step 7: seeds 0-4, mean ± std
│   ├── 06_lstm.ipynb             # (planned)
│   └── 07_analysis.ipynb         # (planned) statistical tests and thesis figures
├── tests/
│   ├── test_labels.py            # RUL cap, engine-level split, pipeline fitted on train only, constant sensors
│   └── test_metrics.py           # NASA score, RMSE, coverage, width against hand-computed values
├── results/runs.csv              # log of every run (re-running a notebook replaces its rows, key = subset, model, seed)
├── figures/                      # every figure produced by the notebooks
├── pyproject.toml                # package metadata, pytest config
├── requirements.txt              # pinned versions used for every logged number
├── .gitignore                    # .venv, data/, caches
└── README.md
```

Planned for later phases: `src/alert_sim.py` (alert-policy simulation) and `src/run.py` (CLI over the full grid).

---

## 7. Setup

```bash
git clone <this repo>
cd v1
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                      # makes `import src.…` work from notebooks and tests

# data - the 12 txt files are looked up in this order (src/config.py: find_data_dir):
#   $CMAPSS_DATA_DIR  ->  v1/data/raw  ->  ../data/raw  ->  ../data/<NASA folder as unzipped>
# so either unzip the archive into data/raw/ or keep it next to the repo (as on the development machine)
python -m src.load --check        # asserts row and unit counts for all subsets

# tests and notebooks
pytest
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb   # or open them in VS Code with the .venv kernel
```

`requirements.txt` pins the exact versions used for every number in `results/` and `figures/`
(numpy, pandas, scikit-learn, lightgbm, matplotlib, mapie, pytest, ipykernel, nbconvert).
`torch` (LSTM phase), `scipy`/`seaborn` (analysis phase) and `mlflow` (optional) are added when those phases start.

**MAPIE note.** This project uses the v1 API (`SplitConformalRegressor`, `ConformalizedQuantileRegressor`, `.conformalize()`, `.predict_interval()`, `confidence_level=`). Tutorials using `MapieRegressor` or `alpha=` are pre-v1 and will not run.

---

## 8. Running experiments

```bash
# one run
python -m src.run --subset FD001 --model lgbm --uq scp --seed 0

# full grid (4 subsets × 3 models × 2 UQ methods × 10 seeds = 240 runs)
python -m src.run --all
```

Each run appends one row to `results/runs.csv`. `notebooks/05_analysis.ipynb` reads that file and produces every table and figure in the thesis; no result is ever typed by hand.

---

## 9. Roadmap

| Phase | Months | Deliverable | Status |
|---|---|---|---|
| Literature review, data pipeline | 1–2 | `src/load.py`, `src/features.py`, `01_eda.ipynb`, integrity checks passing | ☐ |
| Linear and LightGBM baselines | 3 | `02_baselines.ipynb`, FD001 RMSE in plausible range, Javanmardi GB numbers reproduced | ☐ |
| LSTM | 4 | `04_lstm.ipynb`, windows and pinball-loss quantile variant | ☐ |
| Conformal layer, full grid, statistics | 5 | `03_conformal.ipynb`, 240 runs logged, `05_analysis.ipynb`, alert simulation | ☐ |
| Writing and defence | 6 | thesis, figures, code frozen at a tagged release | ☐ |

### Week-1 status (2026-09-15)

Steps 1-7 of the first-week plan are implemented and executed on FD001 (all notebooks run end to end,
`pytest` passes, `results/runs.csv` holds 5 seeds × {linear, lgbm, lgbm_scp, lgbm_cqr}).
Decisions taken in code that the thesis text must state:

- **Official-test truth is capped at `RUL_CAP`** like the training label (`features.official_rul`); scoring against the raw NASA values is one argument away.
- **Interval bounds are clipped to [0, 125]** (`conformal.RUL_RANGE`). The truth always lies there, so clipping never lowers coverage; it only removes impossible width.
- **Conformalisation uses all rows of the calibration engines**, evaluation on the official file uses one row per engine (its last cycle). The two row populations differ (notebook 02), which is part of why single-seed coverage can miss 0.90.
- **LightGBM quantile models use `boost_from_average=False`** (`models.make_lgbm_quantile`). With the default start (the α-quantile of the label = the cap for α ≥ 0.6) the pinball gradient is identical for every row and the α = 0.75 model collapsed to a constant 125.
- **Five-seed FD001 result (official test, 90 % intervals):** LightGBM RMSE 19.5 ± 0.8; SCP coverage 0.89 ± 0.04 with mean width 51 ± 6; CQR coverage 0.86 ± 0.05 with mean width 49 ± 4 (CQR's median model RMSE 19.9 ± 0.7). Untuned, seeds 0-4 only.

**First-month checklist**
- [x] Data downloaded from the official link; counts verified; `data/raw` in `.gitignore`.
- [ ] Saxena 2008, Angelopoulos & Bates 2021, Javanmardi & Hüllermeier 2023 read with notes.
- [x] `requirements.txt` pinned; `runs.csv` logging in place.
- [x] EDA done for FD001 and FD002 with figures saved.
- [x] LightGBM on FD001 producing a plausible RMSE on the official test set.
- [x] SCP and CQR working on FD001 across 5 seeds; coverage ≈ 0.90.
- [ ] One page of notes on the exchangeability assumption (becomes a methods section).

---

## 10. Reading list

1. Saxena, Goebel, Simon, Eklund (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* PHM08. — the dataset paper.
2. Angelopoulos & Bates (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv 2107.07511.
3. Romano, Patterson, Candès (2019). *Conformalized Quantile Regression.* NeurIPS.
4. Javanmardi & Hüllermeier (2023). *Conformal Prediction Intervals for Remaining Useful Lifetime Estimation.* Int. J. PHM; arXiv 2212.14612. — anchor paper.
5. Vollert & Theissler (2021). *Challenges of Machine Learning-based RUL Prognosis: A Review on NASA's C-MAPSS Data Set.* IEEE ETFA. — justifies the protocol.
6. Heimes (2008). *Recurrent Neural Networks for RUL Estimation.* PHM08. / Zheng et al. (2017). *LSTM Network for RUL Estimation.* ICPHM.
7. Barber, Candès, Ramdas, Tibshirani (2022). *Conformal Prediction Beyond Exchangeability.* — for the exchangeability discussion.
8. Demšar (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR. — for the statistical tests.
9. Sensors 2026 (doi 10.3390/s26072249) and RESS 2026 (S0951832026005740) — recent conformal RUL work to compare numbers against.
10. EASA (2024). *Artificial Intelligence Concept Paper — Guidance for Level 1 & 2 ML Applications.* — why uncertainty matters for certification (motivation chapter).

Tutorials: MAPIE documentation (*Use a pre-trained model*, CQR example); `kpeters/exploring-nasas-turbofan-dataset` (preprocessing conventions).

---

## 11. Acknowledgements and license

Data: NASA Ames Prognostics Center of Excellence, Prognostics Data Repository. Code: [MIT / choose a license]. Thesis text: © [Your name], [University], [year].
