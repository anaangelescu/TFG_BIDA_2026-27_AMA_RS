# Research direction: where coverage fails, and what it costs

*Status: 14 October 2026. Combines (a) the conditional-coverage finding from our Claude Code sessions and (b) the tutor's feedback of 14 October, now built and run on FD001–FD004 × 10 seeds (README §8). To agree with Jonas Viklund before chapters are rewritten.*

## 1. The problem in one paragraph

A maintenance planner must decide when to take an engine out of service without knowing exactly how long it will last. A prediction of remaining useful life (RUL) supports that decision; it does not make it. A 90% prediction interval looks like the right tool, but its guarantee holds only *on average over all engines and all moments of their life*. Across all 709 run-to-failure engines and 10 seeds, the misses concentrate in specific remaining-life ranges. Split conformal covers 0.99 of cycles near failure (wasted engine life) and 0.70 in mid-life; CQR covers 0.92 near failure and about 0.81 in mid-life; and every method misses on the late side 11–16 % of the time when the true remaining life is 25–50 cycles (README §8.1). The business question is therefore not "how accurate is the model?" but "where is the interval wrong, what does that cost, and when is acting on an interval worth more than acting on a well-tuned point prediction?"

## 2. Two threads, one thesis

| Thread | Origin | Question | Role in the thesis |
|---|---|---|---|
| Conditional coverage | Claude Code sessions (after reading Javanmardi & Hüllermeier 2023) | Where does a 90 % interval miss, and can calibration per life stage fix it? | Technical core: measure (H1), correct with Mondrian calibration (H2) |
| Decision value | Tutor, 14 Oct | A business thesis starts from the decision: price false alarms vs missed failures, sweep cost ratios, report break-even in euros, test generalisation, let planners adjust | Main thread and framing: H3–H6, the planner prototype |

They meet in one place. Over-coverage near failure means a lower bound that is too low, so alerts come too early and engine life is wasted. Misses on the *below* side mean a lower bound that is too high, so alerts come too late and engines fail. The decision layer turns both into euros. Result (README §8.2): acting on the raw point prediction lets 53 engines in 100 fail; a safety margin tuned on the few calibration engines lets 2–3 fail; a fixed 90 % split-conformal interval lets none fail and, with 10 cycles of notice, is cheaper once a failure costs more than about €4.4 M (median of 40 runs, half between €3.2 M and €6.1 M). The over-coverage near failure that H1 flags as a calibration flaw is exactly what protects the fleet at high cost ratios.

## 3. Literature audit: what is taken

| Study | Models | Uncertainty method | Several model families? | Coverage near failure? | Decisions / costs? | Asymmetric? |
|---|---|---|---|---|---|---|
| Javanmardi & Hüllermeier (2023), IJPHM | Deep CNN, gradient boosting | Split conformal, CQR, time-weighted (non-exchangeable), normalised scores | No | No (marginal only) | No (motivation only) | No |
| Robinson (2026), IJPHM | Gradient boosting | Asymmetric CQR, 90% | No | **Yes, partly:** 5 bins of true RUL at the final window; dips to 0.77 (FD002, RUL 20–40) and 0.83 (FD001, RUL 60–80) on a few dozen engines per bin | No ("maintenance optimization models that explicitly leverage the calibrated prediction intervals" is future work) | Yes |
| BenAbdennour (2026), Machines | 65 candidates | Calibrated residual-quantile intervals | Yes | **Yes, partly:** "urgent-zone" coverage, "false-safe" rate | Alert-type misses, no costs | Yes |
| Diao et al. (2026), Sensors | LSTM quantile regression | CQR, 80% | No | Unclear | No (out of scope by its own statement) | Partly |
| Xu & Zio (2023, 2024) | 6 models | Split conformal, weighted, normalised | Yes | Described only | No | No |
| Moccardi et al. (2025), Future Internet | 6 models | Conformal margins, one-sided | Yes | Partly | Described only | One-sided |
| Noot et al. (2025), IJPHM | Transformer, LSTM | Split conformal | Yes | No | No | No |
| Mitici et al. (2023), RESS | CNN + MC dropout | Monte Carlo dropout | No | No | **Yes:** best replacement time sits near the lower bound of the 99% interval; cost ratio 5:1 | No |
| Lee & Mitici (2023), RESS | CNN + MC dropout | Monte Carlo dropout | No | No | **Yes:** 29% lower cost than replacing at the mean prediction; cost ratio 2:1 only | No |
| Kamariotis et al. (2024), RESS | 4 probabilistic models | Probabilistic outputs | Yes | No | **Yes:** cost ratios swept; no point-prediction baseline | No |
| Zhu et al. (2026), RESS | Unknown (abstract only) | Interval RUL | ? | ? | Claims interval-based beats point-based | ? |

Not peer-reviewed but relevant: a public GitHub project (HermosuraM/rul-predictive-maintenance) already schedules maintenance at a split-conformal lower bound and sweeps cost ratios against a fixed-interval baseline. Cite it and show what we add.

## 4. The gap, stated so it survives a tribunal

1. **Where coverage fails, measured properly.** Coverage stratified by *predicted* and by *true* RUL along whole trajectories of every run-to-failure engine (cross-fitted), with the miss direction, 10 seeds, four subsets. Robinson's bins use one final window per engine and a few dozen engines per bin; BenAbdennour pools an "urgent zone".
2. **A correction with a known price.** Mondrian (group-conditional) calibration over predicted-RUL strata chosen by what the maintenance decision needs, with the width it costs per stratum. Exact conditional coverage is impossible without assumptions (Barber et al., 2019), so the claim is coverage *within chosen strata*, not everywhere (Vovk, 2012; Boström & Johansson, 2020; Gibbs, Cherian & Candès, 2025).
3. **The euro value.** Interval-triggered vs *tuned* point-triggered maintenance on the same engines, swept over failure : planned cost ratios, break-even in euros. Lee & Mitici compared point and distribution at one ratio; Kamariotis et al. and Koutas & Straub swept ratios without a point baseline; conformal papers stop before the decision.

**The catch that shapes H3.** With split conformal the half-width is the same for every engine, so "act on the lower bound" is exactly "act on the point prediction plus a fixed margin". A fair comparison therefore tunes the point rule's margin. An interval can only win if it adapts (CQR) or is stratified (Mondrian). This is tested in `tests/test_decision.py::test_lower_bound_rule_equals_point_plus_constant_margin`.

**What we must not claim.**
- "Adaptive intervals were never validated." Robinson and BenAbdennour did partial checks; ours is the trajectory-level, stratified, decision-priced version.
- "Statistically guaranteed coverage." See the wording in README §5, rule 12.
- "Time-weighted methods handle our problem." Javanmardi & Hüllermeier's weighted method addresses drift between calibration and test. Within-engine autocorrelation, which shrinks the effective sample size, is a different problem; say so explicitly.
- "Adaptive width is our novelty." Their normalised scores are already adaptive. Our contribution is *checking* adaptive width and pricing it.

**Still to check: the RUL cap.** Part of the split-conformal mid-life misses (0.70 coverage at predicted RUL 75–100) are on the *above* side (the engine is healthier than predicted), which is consistent with the cap of 125: healthy-looking engines are labelled 125 and bounds are clipped at 125. Re-run with a cap of 130 or uncapped evaluation before the thesis relies on that number. The late-side misses by true RUL, which drive the decision results, are not affected by the cap (true RUL ≤ 75).

## 5. Decision economics

**Rule from decision theory.** Act when P(failure before the next chance to act) ≥ c_planned / (c_planned + c_failure). This is the cost-sensitive threshold (Elkan 2001), the newsvendor critical ratio (Arrow, Harris & Marschak 1951; Qin et al. 2011), and quantiles as cost-optimal forecasts (Gneiting 2011). The lower end of a central 90% interval is the 5% quantile, which is cost-optimal at a failure : planned ratio of 19 : 1; for a 95% interval, 39 : 1. Choosing a confidence level is an implicit statement about costs.

**Cost assumptions (C-MAPSS is synthetic, so these set a plausible range, not a measurement).**

| Item | Value | Source |
|---|---|---|
| Engines' share of global maintenance spending | 50% | IATA MCX FY2024 (2026) |
| Planned restoration shop visit, CFM56-7B | US$1.1–1.3 million | Aircraft Commerce No. 58 (2008) |
| Unscheduled heavy event | up to US$2 million | Aircraft Commerce (2008) |
| Shop visit after in-flight failure vs planned removal | US$3.61 million vs US$3.00 million | Batalha (2012), airline case |
| Lost contribution while grounded, 5–7 days | US$266,000–372,000 | Batalha (2012) |
| False alarm cleared by an inspection | US$200 | Koops (2018) |
| Spare engine lease, CFM56-7B / LEAP-1A | ~US$95,000 / 145,000–155,000 per month | IBA (2024) |
| Same-day cancellation, narrow-body / wide-body | €25,720 / €123,900 | EUROCONTROL Standard Inputs (2026) |

From these, a missed failure adds about €0.43–1.10 million over a planned removal (our own estimate from the rows above).

**How the ratio is defined in the code.** The decision model (`src/decision.py`) uses a renewal-reward cost rate: total event cost divided by cycles flown. Wasted life is therefore already priced by the denominator (an engine removed early flies fewer cycles per removal), and the ratio compares only the two *events*: the full cost of an unplanned removal over the full cost of a planned one. With a planned shop visit of about €1.0 million (`PLANNED_REMOVAL_EUR` in `src/config.py`), the extra €0.43–1.10 million gives a ratio of about **1.4 to 2.1** for the shop-visit cost alone. Adding disruption (5–7 days grounded, spare-engine lease, cancellations, secondary damage) can push a single event to several times the planned cost, so we treat **1.4–10** as the plausible band, shade it in every figure, and read ratios above 10 as stress tests. The earlier "2 : 1 to 25 : 1" figure mixed wasted life into the ratio and is superseded.

Sweep 1, 1.5, 2, 3, 5, 10, 20, 50, 100 (`COST_RATIOS`). Results are reported in units of one planned removal and converted to euros by multiplying by €1.0 million: a break-even ratio of 3 reads "the interval pays off when an unplanned failure costs more than about €3 million". If a false alarm only triggers an inspection (US$200, Koops 2018) the ratio is about 1,000 : 1 and a different model applies (alert → inspect → decide); out of scope. Do not cite the common claim that unscheduled maintenance costs "3–5×" scheduled: no credible source was found.

**Who pays.** Under power-by-the-hour contracts (Rolls-Royce TotalCare, GE TrueChoice) the engine maker bears unscheduled shop visits, while the airline bears the disruption. State the cost ratio from one party's point of view.

## 6. Behavioural design notes (tutor's suggestion)

**Measured (README §8.5):** moving a safety margin by ± 5 cycles from its optimum costs under 4 %; moving a confidence level one step riskier costs 1–29 %, one step more cautious 0.2–4 %. So planner freedom should be asymmetric: free towards caution, capped towards risk. `app/planner.html` implements a bounded adjustment of ± 2 / ± 5 steps.


- Losing trust in an algorithm after seeing it err is well documented (Dietvorst, Simmons & Massey 2015) but not universal (Logg, Minson & Moore 2019).
- People use an imperfect algorithm when they can adjust its *outputs*, even slightly (Dietvorst et al. 2018). A planner-adjustable threshold extends that finding; it does not repeat it.
- An interval alone does not improve decisions under asymmetric costs; it needs a rule that turns it into an action (Goodwin, Önkal & Thomson 2010).
- False alarms lower compliance with future true alarms (Bliss et al. 1995; LeClerc & Joslyn 2015). Consider a sensitivity scenario in which trust erodes with the false-alarm rate.
- Design implication: show "maintain now / defer" with the interval, plus the false-alarm rate, miss rate and expected cost of each threshold setting.

## 7. Generalisation: M5 Uncertainty (retail)

**Built (README §8.4):** `src/newsvendor.py`, `src/m5.py`, `src/retail.py`. On synthetic demand a constant conformal shift behaves like a tuned buffer (the retail twin of "split conformal = point + margin"), and a conformalised quantile model is cheaper from a stockout : overstock ratio of about 1–2. The M5 run needs the Kaggle files in `data/m5/`.

Same structure: missed failure ↔ stockout, false alarm ↔ overstock; order the τ-quantile of demand with τ = c_stockout / (c_stockout + c_overstock). M5 has 42,840 hierarchical Walmart series and nine quantiles, scored with pinball loss, which is itself a newsvendor cost (Makridakis et al. 2022). Reduced scope: the 154 aggregated series (levels 1–9), 28-day horizon, rolling-origin conformal calibration, newsvendor cost across the same ratio sweep. Closest precedent: Theodorou, Spiliotis & Assimakopoulos (2025), who show on M5 that the most accurate method is often not the best for inventory when lost sales cost more than holding stock.

## 8. Interview leads

IAG AI Lab (Engine Optimisation System for Aer Lingus and Iberia; Ben Dias), ITP Aero (DigitAl Aero, removal-date estimates; Pablo Fuentes), Vueling (Skywise Predictive Maintenance+), TU Delft (Mihaela Mitici), a retail planner for the M5 side. Useful question: what share of predictive alerts do planners act on? No credible published figure exists.

## 9. References

- Angelopoulos, A. N., & Bates, S. (2022). *A gentle introduction to conformal prediction and distribution-free uncertainty quantification*. arXiv:2107.07511.
- Arrow, K. J., Harris, T., & Marschak, J. (1951). Optimal inventory policy. *Econometrica, 19*(3), 250–272.
- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2019). *The limits of distribution-free conditional predictive inference*. arXiv:1903.04684.
- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. *Annals of Statistics, 51*(2). https://doi.org/10.1214/23-AOS2276
- Batalha, E. C. P. (2012). *Aircraft engines maintenance costs and reliability* [MSc dissertation, Universidade Nova de Lisboa]. https://run.unl.pt/bitstream/10362/8792/3/TEGI0308.pdf
- BenAbdennour, A. (2026). Leakage-controlled grouped validation and interval-safety calibration for RUL/SOH prediction. *Machines, 14*(10), 1124. https://doi.org/10.3390/machines14101124
- Bliss, J. P., Gilson, R. D., & Deaton, J. E. (1995). Human probability matching behaviour in response to alarms of varying reliability. *Ergonomics, 38*(11), 2300–2312. https://doi.org/10.1080/00140139508925269
- Boström, H., & Johansson, U. (2020). Mondrian conformal regressors. *PMLR, 128*, 114–133. https://proceedings.mlr.press/v128/bostrom20a.html
- Diao, R., Zhou, M., Meng, G., & Wang, S. (2026). Turbofan engine RUL prediction with reliable prediction intervals via LSTM-based quantile regression and conformal calibration. *Sensors, 26*(7), 2249. https://doi.org/10.3390/s26072249
- Dietvorst, B. J., Simmons, J. P., & Massey, C. (2015). Algorithm aversion. *Journal of Experimental Psychology: General, 144*(1), 114–126. https://doi.org/10.1037/xge0000033
- Dietvorst, B. J., Simmons, J. P., & Massey, C. (2018). Overcoming algorithm aversion. *Management Science, 64*(3), 1155–1170. https://doi.org/10.1287/mnsc.2016.2643
- Elkan, C. (2001). The foundations of cost-sensitive learning. *IJCAI*, 973–978.
- Elmachtoub, A. N., & Grigas, P. (2022). Smart "predict, then optimize". *Management Science, 68*(1), 9–26. https://doi.org/10.1287/mnsc.2020.3922
- EUROCONTROL. (2026). *Standard inputs for economic analyses*. https://ansperformance.eu/economics/cba/standard-inputs/
- Gibbs, I., Cherian, J. J., & Candès, E. J. (2025). Conformal prediction with conditional guarantees. *JRSS-B, 87*(4), 1100–1126. https://doi.org/10.1093/jrsssb/qkaf008
- Gneiting, T. (2011). Quantiles as optimal point forecasts. *International Journal of Forecasting, 27*(2), 197–207. https://doi.org/10.1016/j.ijforecast.2009.12.015
- Goodwin, P., Önkal, D., & Thomson, M. (2010). Do forecasts expressed as prediction intervals improve production planning decisions? *EJOR, 205*(1), 195–201.
- IATA. (2026). *MCX executive report FY2024 (public)*.
- IBA. (2024). *Engine values & lease rates: September 2024*.
- Javanmardi, A., & Hüllermeier, E. (2023). Conformal prediction intervals for remaining useful lifetime estimation. *IJPHM, 14*(2). https://doi.org/10.36001/ijphm.2023.v14i2.3417
- Kamariotis, A., Tatsis, K., Chatzi, E., Goebel, K., & Straub, D. (2024). A metric for assessing and optimizing data-driven prognostic algorithms for predictive maintenance. *RESS, 242*, 109723. https://doi.org/10.1016/j.ress.2023.109723
- Koops, L. (2018). ROC-based business case analysis for predictive maintenance. *PHM Society European Conference, 4*(1). https://doi.org/10.36001/phme.2018.v4i1.460
- LeClerc, J., & Joslyn, S. (2015). The cry wolf effect and weather-related decision making. *Risk Analysis, 35*(3), 385–395. https://doi.org/10.1111/risa.12336
- Lee, J., & Mitici, M. (2023). Deep reinforcement learning for predictive aircraft maintenance using probabilistic RUL prognostics. *RESS, 230*, 108908. https://doi.org/10.1016/j.ress.2022.108908
- Logg, J. M., Minson, J. A., & Moore, D. A. (2019). Algorithm appreciation. *OBHDP, 151*, 90–103. https://doi.org/10.1016/j.obhdp.2018.12.005
- Makridakis, S., Spiliotis, E., Assimakopoulos, V., Chen, Z., Gaba, A., Tsetlin, I., & Winkler, R. L. (2022). The M5 uncertainty competition. *International Journal of Forecasting, 38*(4), 1365–1385. https://doi.org/10.1016/j.ijforecast.2021.10.009
- Mitici, M., de Pater, I., Barros, A., & Zeng, Z. (2023). Dynamic predictive maintenance for multiple components using data-driven probabilistic RUL prognostics. *RESS, 234*, 109199. https://doi.org/10.1016/j.ress.2023.109199
- Moccardi, A., Conte, C., Chandra Ghosh, R., & Moscato, F. (2025). A robust conformal framework for IoT-based predictive maintenance. *Future Internet, 17*(6), 244. https://doi.org/10.3390/fi17060244
- Noot, J.-P., Martin, M., & Birmelé, E. (2025). LSTM and transformers based methods for RUL prediction considering censored data. *IJPHM, 16*(2). https://doi.org/10.36001/IJPHM.2025.v16i2.4260
- Qin, Y., Wang, R., Vakharia, A. J., Chen, Y., & Seref, M. M. H. (2011). The newsvendor problem: Review and directions for future research. *EJOR, 213*(2), 361–374.
- Robinson, C. (2026). RUL estimation for aircraft engines with risk-aware prediction intervals via conformalized quantile regression. *IJPHM, 17*(1). https://doi.org/10.36001/ijphm.2026.v17i1.4724
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *PHM 2008*.
- Theodorou, E., Spiliotis, E., & Assimakopoulos, V. (2025). Forecast accuracy and inventory performance: Insights from the M5 competition data. *EJOR, 322*(2), 414–426. https://doi.org/10.1016/j.ejor.2024.12.033
- Vovk, V. (2012). Conditional validity of inductive conformal predictors. *ACML, PMLR 25*. https://proceedings.mlr.press/v25/vovk12.html
- Xu, W., & Zio, E. (2024). Uncertainty-aware prediction of remaining useful life in complex systems. *Annual Conference of the PHM Society, 16*(1). https://doi.org/10.36001/phmconf.2024.v16i1.4178
- Zhu, J., Zhu, Z., Li, B., & Zhao, X. (2026). Predictive maintenance strategy based on critical probabilistic cost. *RESS, 267*. https://doi.org/10.1016/j.ress.2025.111823
