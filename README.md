# tfg-rul-uncertainty

**When does uncertainty pay? Conditional coverage and the decision value of prediction intervals for turbofan maintenance**
*(working title, proposed after tutor feedback on 14 Oct 2026 and still to be agreed)*

Bachelor's thesis (TFG). Ana-Maria Angelescu & Rune Snauwaert, Business Intelligence and Data Analytics, La Salle Barcelona, 2026–2027. Tutor: Jonas Viklund.

---

## 0. Status at a glance

| | |
|---|---|
| Pipeline (FD001) | Done: load, features, engine-level splits, Linear + LightGBM, split conformal (SCP) and conformalised quantile regression (CQR), 5 seeds, 20 runs logged |
| Preliminary finding | 90% intervals cover ≈ 0.85–0.89 *on average*, but not where it matters. Seed 0, by predicted RUL: split conformal covers 1.00 near failure and 0.49 in mid-life; CQR narrows near failure and covers only ≈ 0.75 there (§8) |
| Decision layer | v0.1 added: `src/decision.py` (maintenance rules, removal/failure simulation, cost rate, cost-ratio sweep, break-even) with tests |
| Direction | Proposed: measure where coverage fails → calibrate where decisions are made → price it in euros. To agree with the tutor before rewriting chapters |
| Next | Notebook 06 (coverage by stratum, 10 seeds, bootstrap); Mondrian calibration; decision sweep with more engines (cross-fitting) |

---

## 1. How the research question evolved

The question changed four times. Each step happened because the previous version turned out to be already published, too big, or not a business question. The thesis keeps the useful part of each.

| # | Question | Why we moved on | What we kept |
|---|---|---|---|
| 1 | *The Living Engine*: an on-board, real-time engine health system | Far too broad for one TFG (tutor) | The motivation chapter: unplanned engine events are expensive |
| 2 | Does deep learning beat simple models on **accuracy** (RMSE) under a fair, leak-free protocol? | Accuracy comparisons on C-MAPSS are everywhere | The protocol: engine-level splits, RUL cap stated, seeds, statistical tests |
| 3 | Which model gives the most **trustworthy intervals**, and is it the most accurate one? | Conformal intervals for engine life on C-MAPSS are published (Javanmardi & Hüllermeier 2023; Robinson 2026) | The conformal pipeline (SCP, CQR) |
| 4a | *(Claude Code chat)* Marginal coverage hides **where** intervals fail. Split conformal over-covers near failure and under-covers mid-life; CQR does the opposite near failure | Not a dead end. This is a measured, partly open gap (§3) | **Kept as the technical core** |
| 4b | *(tutor, 14 Oct)* A business thesis starts from the **decision** the interval supports: price false alarms vs missed failures, sweep cost ratios, report break-even in euros, test generalisation | Not a dead end | **Kept as the main thread** |

So the current proposal is 4a + 4b. **Accuracy (RMSE) is no longer the object of study.** It stays as a reported metric and as the thing we show is *not* a sufficient guide (H4).

---

## 2. Research question and hypotheses (proposed)

**Research question.** Where do conformal prediction intervals for engine life fail to cover, what does that cost a maintenance planner, and when does acting on a calibrated interval beat acting on a well-tuned point prediction?

| | Hypothesis | Evidence needed |
|---|---|---|
| H1 | A 90% interval that is valid *on average* misses badly in specific remaining-life ranges, and the range depends on the method: constant width (SCP) fails in mid-life, adaptive width (CQR) fails near failure | Coverage by predicted-RUL stratum, along full engine trajectories, engine-level bootstrap, 10 seeds, 4 subsets |
| H2 | Calibrating separately within predicted-RUL strata (Mondrian conformal) brings each stratum near 90% at a measurable width cost. Exact conditional coverage is impossible (Barber et al., 2019), so the strata are chosen by what the maintenance decision needs | Per-stratum coverage and width, SCP vs CQR vs Mondrian |
| H3 | A constant-width interval adds nothing beyond a tuned safety margin (it *is* that rule). An adaptive or stratified interval beats the tuned point rule only above a break-even ratio of failure cost to planned-removal cost | Decision simulation (§7), cost-ratio sweep 1:1 to 100:1, break-even with bootstrap band |
| H4 | Ranking models by RMSE is not the same as ranking them by decision cost | Linear vs LightGBM vs LSTM on both metrics |
| H5 *(optional)* | The same framework transfers to retail demand (M5 Uncertainty: stockout vs overstock) | Reduced M5 test, same break-even plot |

---

## 3. What is new, and what we must not claim

Detailed review with sources: [`docs/research_direction.md`](docs/research_direction.md).

**Already published, so not our contribution:** conformal intervals for engine life on C-MAPSS (Javanmardi & Hüllermeier 2023; Robinson 2026; Diao et al. 2026); comparing several model families' intervals (Xu & Zio 2023; Moccardi et al. 2025; BenAbdennour 2026); asymmetric intervals and coverage by remaining-life bin at the *final* test window (Robinson 2026); "urgent-zone" coverage (BenAbdennour 2026); cost-based maintenance from probabilistic RUL (Mitici et al. 2023; Lee & Mitici 2023; Kamariotis et al. 2024).

**Still open (no peer-reviewed study found):**
1. Coverage measured **along whole test trajectories**, stratified by *predicted* RUL, with engine-level uncertainty, instead of ~20 final-window engines per bin.
2. **Correcting** it with Mondrian calibration over predicted-RUL strata and reporting the width it costs.
3. Pricing the result: interval-triggered vs **tuned** point-triggered maintenance across cost ratios, with the **break-even in euros**. Robinson (2026) lists "maintenance optimization models that explicitly leverage the calibrated prediction intervals" as future work.

**Claims to avoid:**
- "first conformal RUL"
- "adaptive intervals were never validated" (Robinson 2026 and BenAbdennour 2026 did partial checks)
- "statistically guaranteed coverage" (see §5, rule 12)
- "conditional coverage achieved" (impossible in general; we target chosen strata)

**Risk to check first:** Zhu et al. (2026, *RESS* 267) claim, in the abstract, that an interval-based maintenance strategy beats point-based ones. Read the full text before claiming gap 3.

---

## 4. Data

NASA Prognostics Center of Excellence, *Turbofan Engine Degradation Simulation Data Set* (C-MAPSS, item 6 of the [Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)). Download: `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip` (~12 MB). Citation: Saxena, A., & Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set*. NASA Ames Research Center.

| Subset | Train engines | Test engines | Operating conditions | Fault modes |
|---|---|---|---|---|
| FD001 | 100 | 100 | 1 | 1 (HPC) |
| FD002 | 260 | 259 | 6 | 1 (HPC) |
| FD003 | 100 | 100 | 1 | 2 (HPC, fan) |
| FD004 | 249 | 248 | 6 | 2 (HPC, fan) |

`src/load.py` asserts the row and engine counts of all 12 files on every load.

<details>
<summary>File format and the 21 sensors</summary>

Space-separated text, 26 columns, no header: `unit`, `cycle`, `setting_1..3` (altitude, Mach, throttle resolver angle), `s_1..s_21`.

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

</details>

Not used: N-CMAPSS (~10 GB, out of scope); PHM08 challenge data (optional extra run-to-failure engines if NASA grants access). Optional generalisation test: M5 Uncertainty (Walmart sales, Kaggle), reduced to the 154 aggregated series.

---

## 5. Protocol: rules that are never broken

1. **Split by engine, never by row** (`GroupShuffleSplit`, `groups = unit`). Rows of one engine are a correlated time series.
2. **Three disjoint engine groups** per training file: 70% train, 15% calibration, 15% internal test. Exact integer engine counts (`src/split.py`). The official `test_FD00X.txt` is a fourth, final set.
3. **RUL label** = `min(max_cycle − cycle, 125)` (`RUL_CAP` in `src/config.py`). Official-test truth is capped the same way (`features.official_rul`).
4. **Constant sensors are detected** on the training engines (std < 0.01 in every regime), not hard-coded.
5. **FD002/FD004:** settings clustered into 6 regimes (KMeans, fixed seed); each sensor z-scored within its regime.
6. **Everything that is fitted sees training engines only:** scalers, regimes, models.
7. **Features:** z-scored sensors + 10-cycle rolling mean and std per engine (42 features on FD001).
8. **Official test** = last cycle of each test engine.
9. **Seeds:** every random operation takes an explicit seed; reported results use seeds 0–9 (week 1: 0–4).
10. **Every official-test run is logged** to `results/runs.csv` (`subset, model, seed, rmse, nasa_score, coverage, mean_width, n_test`). No number in the thesis is typed by hand.
11. **Conformalise on whole calibration engines.** Engines are roughly exchangeable; rows within an engine are not.
12. **Coverage wording.** "Intervals calibrated to a nominal 90% marginal coverage, verified empirically." The guarantee is marginal and assumes exchangeable calibration and test cases. Because rows within an engine are dependent and C-MAPSS test engines are truncated, it holds only approximately, at engine level. We report the coverage actually achieved, overall and per stratum.
13. **Strata use predicted RUL, never true RUL.** True RUL is unknown when the decision is made. True-RUL bins are only for diagnostic plots.
14. **Decisions are simulated on complete engines** (calibration and internal-test engines from the training files), never on the truncated official file.

---

## 6. Metrics

| Layer | Metric | Notes |
|---|---|---|
| Accuracy | RMSE (cycles); NASA score (late errors penalised more) | Reported, not optimised; H4 tests whether it predicts decision cost |
| Interval | Coverage, mean width (overall) | Nominal 90% |
| Interval | Coverage and width **per predicted-RUL stratum**, along full trajectories, engine-level bootstrap | H1, H2 |
| Decision | Failures, planned removals, mean wasted cycles per engine | Per 100 engines, so readers can plug in their own costs |
| Decision | Cost per 1,000 cycles flown at each failure : planned cost ratio | Renewal-reward cost rate |
| Decision | Break-even ratio between two rules; convert to euros by fixing the planned-removal cost | H3; bootstrap band over engines |

---

## 7. Decision layer (`src/decision.py`)

**Maintenance model.** An engine fails right after its last recorded cycle. At each check cycle a rule may raise an alert. A removal needs `horizon` cycles of notice, so the first alert at cycle *t* books a removal at *t + horizon*. If that is before failure, the removal is planned and the remaining life is wasted. Otherwise the engine fails. A planned removal costs `planned`, a failure costs `failure`. Wasted life is priced through the cost rate (total cost ÷ cycles flown), so removing every engine early cannot look cheap.

| Rule | Alert when | Role |
|---|---|---|
| `alert_age` | cycle ≥ replacement age − horizon | Time-based baseline (age tuned) |
| `alert_point` | prediction ≤ horizon + margin | Point rule; margin tuned per cost ratio, so it is a fair baseline |
| `alert_lower` | interval lower bound ≤ horizon | Interval rule (SCP, CQR, Mondrian) |
| `alert_oracle` | true RUL ≤ horizon | Perfect-information floor |

With split conformal the half-width is the same for every engine, so `alert_lower` is exactly `alert_point` with margin = half-width. This is tested in `tests/test_decision.py`. An interval can only beat a *tuned* margin if it adapts to the engine.

**Theory anchor.** Acting is cost-optimal when P(failure before the next chance to act) ≥ c_planned / (c_planned + c_failure). This is the cost-sensitive threshold (Elkan 2001) and the newsvendor critical ratio. The lower end of a central 90% interval is the 5% quantile, which is cost-optimal at a failure : planned ratio of 19 : 1 (`cost_matched_tau`, `level_for_tau`).

```python
from src.experiment import run_seed, decision_frame
from src.conformal import predict_interval
from src.decision import Costs, alert_point, alert_lower, simulate, tune, sweep, break_even

out = run_seed("FD001", seed=0, log=False)
data, cqr = out["data"], out["conformal"]["lgbm_cqr"]
H = 10                                                     # cycles of notice needed to plan a removal

def frame(name, cp):
    pred, lo, _ = predict_interval(cp, data.X(name))
    return decision_frame(data, name, pred, lo)

cal, test = frame("cal", cqr), frame("test", cqr)

def point_tuned(costs):                                    # margin re-tuned on calibration engines for each ratio
    m, _ = tune(cal, lambda m: alert_point(cal, H, m), range(0, 61), H, costs)
    return simulate(test, alert_point(test, H, m), H)

table = sweep({"point + tuned margin": point_tuned,
               "CQR lower bound": lambda c: simulate(test, alert_lower(test, H), H)},
              ratios=[1, 2, 3, 5, 10, 20, 50, 100])
break_even(table, "point + tuned margin", "CQR lower bound")  # failure:planned ratio where CQR becomes cheaper
```

The decision layer is unit-tested and has had one preliminary run on FD001 (§8). Open design choice: 15 internal-test engines per seed is small for costs. Consider evaluating decisions on every training engine once via 5-fold engine-level cross-fitting.

### Running experiments today

One experiment is one `(subset, seed)` pair, run from Python:

```python
from src.experiment import run_seed
res = run_seed("FD001", seed=0)          # linear + lgbm + SCP + CQR; appends 4 rows to results/runs.csv
res["point_table"], res["interval_table"]
```

The notebooks are the entry points: `02_baselines` and `03_conformal` run seed 0; `05_seed_variability` runs seeds 0–4. Re-running a notebook **replaces** the rows it owns (key = `subset, model, seed`), so `results/runs.csv` stays a clean snapshot of the latest code. A `src/run.py` command-line runner over the full grid is planned once the LSTM and the other subsets are in.

---

## 8. Results so far

FD001, official test (100 engines), untuned models, 90% intervals, seeds 0–4 (`results/runs.csv`):

| Model | RMSE | NASA score | Coverage | Mean width |
|---|---|---|---|---|
| Linear | 21.1 ± 0.3 | 1,711 ± 208 | n/a | n/a |
| LightGBM | 19.5 ± 0.8 | 1,593 ± 326 | n/a | n/a |
| LightGBM + SCP | 19.5 ± 0.8 | 1,593 ± 326 | 0.89 ± 0.04 | 51 ± 6 |
| LightGBM + CQR | 19.9 ± 0.7 | 1,583 ± 152 | 0.86 ± 0.05 | 49 ± 4 |

Findings from the notebooks that matter for the new direction:

- **RMSE is dominated by early-life errors**, where the model can only answer "healthy". Near failure both models track the truth closely (notebook 02). The accuracy score is driven by the part of an engine's life where no maintenance decision is taken, which is a first piece of evidence for H4.
- **Coverage is far from uniform per engine** (0.44–0.98 on internal-test engines, notebook 03), because one engine's errors are autocorrelated.
- **CQR widths shrink near failure; SCP widths are flat** (notebook 04, `figures/fd001_width_vs_rul.png`).
- **Preliminary: coverage by remaining-life range** (FD001, seed 0, 15 internal-test engines, all 3,432 cycles, `evaluate.coverage_by_stratum`). *Below* = true life shorter than the lower bound (late-alert side); *above* = longer than the upper bound.

  | Predicted RUL | SCP coverage | SCP below / above | SCP width | CQR coverage | CQR below / above | CQR width |
  |---|---|---|---|---|---|---|
  | 0–25 | 1.00 | 0.00 / 0.00 | 37 | 0.75 | 0.04 / 0.21 | 13 |
  | 25–50 | 0.95 | 0.00 / 0.05 | 50 | 0.79 | 0.05 / 0.16 | 32 |
  | 50–75 | 0.74 | 0.14 / 0.12 | 50 | 0.83 | 0.07 / 0.10 | 57 |
  | 75–100 | **0.49** | 0.14 / 0.37 | 50 | 0.80 | 0.05 / 0.15 | 70 |
  | > 100 | 0.91 | 0.09 / 0.00 | 35 | 0.91 | 0.02 / 0.07 | 46 |

  One seed and 15 engines, so these are a direction, not a result. Most SCP mid-life misses are on the *above* side (the engine is healthier than predicted), which is consistent with the RUL cap of 125, so check that before relying on it. The CQR near-failure misses include some on the late-alert side, which is where maintenance decisions are made.

- **Preliminary: first decision run** (same seed and engines, horizon 10 cycles, margin and age tuned on the 15 calibration engines). Cost per 1,000 cycles flown at a failure : planned ratio of 10 : 1:

  | Rule | Failures (of 15) | Wasted cycles per engine | Cost |
  |---|---|---|---|
  | Oracle (perfect information) | 0 | 0 | 4.37 |
  | Point + tuned margin | 0 | 6.9 | **4.51** |
  | CQR lower bound | 0 | 10.7 | 4.58 |
  | SCP lower bound | 0 | 24.5 | 4.89 |
  | Fixed age (tuned) | 0 | 78.8 | 6.67 |
  | Point, no margin | 7 | 1.9 | 22.91 |

  Acting on the raw point prediction lets about half the engines fail. A tuned safety margin is a strong baseline: on this seed it beats both interval rules at every ratio from 2 : 1 to 100 : 1 (no break-even). CQR beats SCP because it wastes less life. This is the kind of honest result H3 is designed to test with 10 seeds and more engines.

- Numbers in this section were re-run on Linux with the pinned versions; LightGBM can differ slightly from the macOS runs in `results/runs.csv` (e.g. official SCP coverage 0.80 here vs 0.83 logged).

Decisions taken in code that the thesis must state:
- Official truth is capped at 125.
- Interval bounds are clipped to [0, 125].
- Conformalisation uses all rows of the calibration engines.
- LightGBM quantile models use `boost_from_average=False`, because otherwise the 0.75 model collapsed to a constant 125.

---

## 9. Repository layout

```
TFG_BIDA_2026-27_AMA_RS/              # the raw data lives OUTSIDE the repo (see §10)
├── src/
│   ├── config.py                     # RUL_CAP=125, SEED=0, ROLL_WINDOW=10, N_REGIMES=6, CONFIDENCE_LEVEL=0.90, data lookup
│   ├── load.py                       # load_subset(name) -> train, test, rul; asserts counts; `python -m src.load --check`
│   ├── features.py                   # add_rul, FeaturePipeline (regimes, z-score, constant sensors, rolling), build_features
│   ├── split.py                      # split_engines: 70/15/15 by engine
│   ├── models.py                     # Linear, LightGBM, LightGBM quantile trio
│   ├── conformal.py                  # fit_scp / fit_cqr / fit_cqr_per_level around MAPIE v1; bounds clipped to [0, 125]
│   ├── evaluate.py                   # rmse, nasa_score, coverage, mean_width, coverage_by_stratum (NEW)
│   ├── decision.py                   # NEW: maintenance rules, simulate, cost_summary, tune, sweep, break_even
│   ├── results.py                    # append_run -> results/runs.csv
│   ├── experiment.py                 # prepare / fit / evaluate / run_seed; decision_frame (NEW)
│   └── plots.py                      # EDA and interval figures
├── notebooks/                        # 01_eda, 02_baselines, 03_conformal, 04_interval_plots, 05_seed_variability
├── tests/                            # test_labels, test_metrics (+ stratum test), test_decision (NEW)
├── docs/research_direction.md        # NEW: literature audit, gap, hypotheses, cost assumptions, references
├── results/runs.csv                  # one row per (subset, model, seed), official test
├── figures/
├── pyproject.toml · requirements.txt # pinned versions used for every logged number
└── README.md
```

Planned: `06_conditional_coverage.ipynb` (turn the preliminary table in §8 into 10 seeds with bootstrap), `07_mondrian.ipynb`, `08_decisions.ipynb`, `09_lstm.ipynb`, `10_analysis.ipynb` (statistical tests and thesis figures), and optionally `m5/` for the generalisation test.

---

## 10. Setup

```bash
git clone https://github.com/anaangelescu/TFG_BIDA_2026-27_AMA_RS.git
cd TFG_BIDA_2026-27_AMA_RS
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                       # makes `import src.…` work from notebooks and tests

# data: unzip the NASA archive into data/raw/ (git-ignored), or keep it next to the repo.
# Lookup order (src/config.py, find_data_dir): $CMAPSS_DATA_DIR -> ./data/raw -> ../data/raw -> ../data/<NASA folder>
python -m src.load --check                             # asserts row and engine counts for all subsets

pytest                                                 # 28 tests; the one that needs the NASA files is skipped without them
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

`requirements.txt` pins the exact versions used for every number in `results/` and `figures/` (Python 3.12, macOS arm64). `torch` (LSTM), `scipy` (statistical tests) and the M5 tooling are added when those phases start. This project uses the MAPIE **v1** API (`SplitConformalRegressor`, `ConformalizedQuantileRegressor`, `confidence_level=`). Tutorials with `MapieRegressor` or `alpha=` will not run.

---

## 11. Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| Pipeline and baselines (FD001) | Notebooks 01–05, 20 runs logged, tests passing | ☑ |
| Agree direction with tutor | Research question and hypotheses above, by email or meeting | ☐ |
| Conditional coverage (H1) | Coverage by predicted-RUL stratum along full trajectories, engine-level bootstrap, 10 seeds, FD001–FD004 | ◐ seed 0 only (§8) |
| Mondrian calibration (H2) | Per-stratum calibration, coverage and width cost vs SCP and CQR | ☐ |
| Decision layer (H3) | `src/decision.py` v0.1 ☑ · first FD001 run ☑ (§8) · 10 seeds + cross-fitting for more engines ☐ | ◐ |
| LSTM and model comparison (H4) | LSTM in the same pipeline; RMSE ranking vs decision-cost ranking | ☐ |
| Generalisation (H5, optional) | M5 levels 1–9, same framework and break-even plot | ☐ |
| Interviews | 4–6 practitioners to validate cost assumptions | ☐ |
| Writing and defence | Thesis in the La Salle format, code frozen at a tagged release | ☐ |

---

## 12. Key reading

1. Saxena, Goebel, Simon & Eklund (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *PHM08* (dataset paper).
2. Angelopoulos & Bates (2022). A gentle introduction to conformal prediction. arXiv 2107.07511.
3. Javanmardi & Hüllermeier (2023). Conformal prediction intervals for remaining useful lifetime estimation. *IJPHM* 14(2). https://doi.org/10.36001/ijphm.2023.v14i2.3417
4. Robinson (2026). RUL estimation for aircraft engines with risk-aware prediction intervals via CQR. *IJPHM* 17(1). https://doi.org/10.36001/ijphm.2026.v17i1.4724
5. Barber, Candès, Ramdas & Tibshirani (2019). The limits of distribution-free conditional predictive inference. arXiv 1903.04684
6. Vovk (2012). Conditional validity of inductive conformal predictors. *ACML, PMLR* 25. https://proceedings.mlr.press/v25/vovk12.html ; Boström & Johansson (2020). Mondrian conformal regressors. *PMLR* 128. https://proceedings.mlr.press/v128/bostrom20a.html
7. Gibbs, Cherian & Candès (2025). Conformal prediction with conditional guarantees. *JRSS-B* 87(4). https://doi.org/10.1093/jrsssb/qkaf008
8. Mitici, de Pater, Barros & Zeng (2023). Dynamic predictive maintenance with probabilistic RUL prognostics: the case of turbofan engines. *RESS* 234. https://doi.org/10.1016/j.ress.2023.109199
9. Lee & Mitici (2023). Deep reinforcement learning for predictive aircraft maintenance using probabilistic RUL prognostics. *RESS* 230. https://doi.org/10.1016/j.ress.2022.108908
10. Elmachtoub & Grigas (2022). Smart "predict, then optimize". *Management Science* 68(1). https://doi.org/10.1287/mnsc.2020.3922

Full reference list and cost sources: [`docs/research_direction.md`](docs/research_direction.md).

---

## 13. Acknowledgements and license

Data: NASA Ames Prognostics Center of Excellence, Prognostics Data Repository. Code and text: Ana-Maria Angelescu & Rune Snauwaert, La Salle Barcelona, 2026–2027.
