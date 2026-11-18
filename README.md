# Replication package

**Decision-Focused Covariance Shrinkage for Minimum-Variance Portfolios**
Varun Viswapriyan, The Ohio State University

This package reproduces every table and figure in the revised manuscript (second revision, R2).

```
code/                 all Python code
data/                 cached Yahoo Finance prices and volumes used for every result
results_original/     outputs of the main pipeline (Tables 1, 2, 6-12, Figures 1, 2, 4)
results_revision/     outputs of the first-revision analyses (Tables 3-6, 8, 13, Figure 3, Section 5.4)
results_r2/           outputs of the second-revision checks (Section 6.5 survivorship, Section 6.7 sign comparison)
run_revision.sh       one command that regenerates results_revision/
run_r2.sh             one command that regenerates the NumPy-only parts of results_r2/
requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt          # torch only needed for the gradient-trained DFS variants
bash run_revision.sh                     # ~5-15 min on a laptop, NumPy/SciPy/scikit-learn only
bash run_r2.sh                           # seconds; NumPy/pandas only
```

To regenerate the main pipeline outputs (needs torch; the gradient-trained DFS variants dominate the run time, which is considerably longer than the revision scripts):

```bash
cd code
cp ../data/prices_*.csv ../data/volume_*.csv ../results_original/   # use the cached data
python dfs_pipeline.py        --out ../results_original --part all
python dfs_pipeline_oracle.py --out ../results_original --part all
```

`dfs_shrinkage.py` is the first, superseded version of the pipeline (initial simulation and
long-only experiments). It is kept for completeness; no table in the revised paper depends on it.

## Data

* **Source.** Yahoo Finance daily adjusted close prices and volumes, 1 Dec 2004 - 31 Dec 2024,
  downloaded with `yfinance` using `auto_adjust=True` (`code/fetch_data.py` re-downloads them).
* **Dividends are included.** Yahoo's adjusted close is adjusted for stock splits *and* for dividend
  distributions (fund distributions for the ETFs): all prices before a split or ex-dividend date are
  multiplied by an adjustment factor. Daily returns computed from these prices are therefore total
  returns that include dividends, not price-only returns. The same terminology is used in the paper
  (Section 6.1, Appendix A and the limitations in Section 7).
  Yahoo revises adjusted prices retroactively, so a fresh download can differ slightly from the cache.
  The cached files in `data/` reproduce the reported numbers exactly.
* **Candidate lists.** `CFG["stocks"]` (91 large-cap US names) and `CFG["etfs"]` (12 ETFs) in `code/dfs_pipeline.py`.

### Processing steps (`dfs_pipeline.clean_universe`, `dfs_pipeline.precompute`)

1. **Inclusion.** A ticker is kept if its first valid price falls within 60 calendar days of 1 Dec 2004
   and fewer than 2% of its prices are missing. This keeps 90 stocks and 10 ETFs (DBC and HYG start later).
2. **Stock universe.** The first 60 of the retained stocks in alphabetical order (Appendix A of the paper).
   The "all stocks" universe in Section 6.7 uses all 90.
3. **Cleaning.** Prices are forward-filled across gaps of at most 5 trading days; remaining rows with any
   missing value are dropped; simple daily returns are computed and clipped to [-50%, +50%].
   The resulting panel runs from 2 Dec 2004 to 30 Dec 2024 (5,053 trading days).
4. **Windows.** Rebalancing every h = 21 trading days, starting at day L = 250. At rebalance r the
   estimation window is days [t-L, t), demeaned, with sample covariance S = X'X / L. The holding
   period is days [t, t+h). This gives 228 rebalances; the first 24 form the initial training set,
   and all methods are evaluated on the remaining 204 (30 Nov 2007 - 7 Nov 2024).
5. **No look-ahead.** A holding period's returns enter a training set only after it has ended.
   `dfs_pipeline.part_audit` asserts this at every rebalance (`results_original/audit_lookahead.json`).
6. **Evaluation.** Realized variance = 252 x variance of the concatenated daily out-of-sample portfolio
   returns, in %^2 per year. Turnover = l1 distance between consecutive target weights, annualized.
   Net returns subtract cost x turnover on the first day of each holding period.
7. **Inference.** Studentized stationary bootstrap (Politis-Romano), 1,000 replications, mean block 10,
   seed 7, HAC (Bartlett/Andrews) standard errors; Romano-Wolf stepdown.

The rotating point-in-time universe (Section 6.5) selects, at each rebalance, the 60 names with the highest
trailing dollar volume among candidate tickers with complete prices over the **estimation window only**
(`build_P(..., rotating=True)`), so constituents are determined only from information available on the
rebalancing date. The R1 code also required complete prices over the subsequent holding period and screened
candidates on their full-sample number of valid prices; both conditions used information from after the
rebalancing date and were removed in R2 (reviewer comment 1). On the cached data the corrected rule selects
identical constituents at all 228 rebalances, and the Section 6.5 results are unchanged (see `results_r2/`).

## Reproduction map

| Paper item | Script | Output file(s) |
|---|---|---|
| Table 1, 2 (factor / hetero simulations) | `dfs_pipeline.py --part sim` | `results_original/tabR12_sim_extended.*` |
| **Table 3** (sign condition vs simulated optima) | `r1_sign_theory.py 1` | `results_revision/R1_sign_theory_check_dgps.csv` |
| **Table 4** (600 random structures) | `r1_sign_theory.py 2` | Panel A: `R1_sign_random_design_summary.csv`; Panel B: `R1_sign_random_design_by_dispersion_table4B.csv`; raw: `R1_sign_random_design_raw.csv` |
| **Section 5.4** (Prop. 1 finite-difference and FOC checks) | `r1_prop1_check.py` | `R1_prop1_derivative_check_*.csv`, `R1_prop1_FOC_check_*.csv` |
| Figure 1 (intensity sweep) | `dfs_pipeline.py --part attribution` | `results_original/tabR3_delta_sweep_*.csv` |
| **Table 5** Panels A-B (gap separation) | `r1_gap_bridge.py a c` | `R1_gap_empirical_rules.csv`, `R1_gap_empirical_summary.csv`, `R1_fcv_vs_dfsg*.csv` |
| **Table 5** Panel C (known-Sigma bridge) | `r1_gap_bridge.py b` | `R1_gap_simulation_bridge.csv` |
| Table 6 (shortfall recovery) | `dfs_pipeline_oracle.py --part oracle`; FCV-G row from `r1_gap_bridge.py c` | `results_original/tabR16_gap_decomposition_*`, `R1_fcv_vs_dfsg.csv`; evaluation-period mean intensities from `results_original/deltas_*.csv` (rows 25-228) |
| Table 7, 12 (estimator comparison) | `dfs_pipeline.py --part core` | `results_original/tabR1_core_*.csv` |
| **Table 8** (inference, two Romano-Wolf families) | `r1_inference_audit.py` | `R1_romano_wolf_families.csv` |
| Section 6.3 attribution and weight diagnostics | `dfs_pipeline.py --part attribution` | `tabR4_vol_vs_corr_*`, `tabR5_weight_diag_*` |
| Section 6.3 placebos, seeds | `dfs_pipeline_oracle.py --part placebo2`; `dfs_pipeline.py --part seeds` | `tabR18*_placebo_v2*`, `tabR10*_seed*` |
| Figure 2, break-even costs | `dfs_pipeline.py --part attribution` | `tabR6_breakeven_*`, `tabR1_core_*` |
| Table 9 (L, h) | `dfs_pipeline.py --part hyper` | `tabR7_hyper_L_h.*`, `tabR7b_hyper_variance_grid.*` |
| Table 10 (subperiods) | `dfs_pipeline.py --part subperiod` (400 bootstrap replications per subperiod) | `tabR8_subperiod_*` (`var_ratio_vs_LWI`, `p_boot`) |
| Survivorship (Section 6.5) | `dfs_pipeline.py --part survivor` (R1 outputs); `r2_survivor_pit.py` (R2 recalculation, needs torch) | `results_original/tabR11*_survivorship*`; `results_r2/tabR11*_pit.*`, `R2_survivor_*` |
| delta* stability (Section 6.2) | `dfs_pipeline_oracle.py --part stability` | `tabR19_delta_star_stability.*` |
| Table 11 (ETF-calibrated prediction) | `dfs_pipeline_oracle.py --part etftie` | `tabR20_etf_hetero_prediction.*` |
| **Table 13, Figure 3** (191 universes) | `r1_universes.py` | `R1_universes_named.csv`, `R1_universes_raw.csv`, `R1_universes_summary.csv` |
| Section 6.7 sign comparison (vs. Ledoit-Wolf and vs. ex-post Frobenius optimum) | `r2_sign_comparison.py` | `results_r2/R2_sign_comparison.csv`, `R2_sign_comparison_ranges.csv` |
| Figure 4 (DFS-NN learning curve) | `dfs_pipeline.py --part sim` | `tabR14_dfsnn_learning_curve.*` |

### Notes

* `code/_suite.py` imports `dfs_pipeline.py` for its NumPy routines. If PyTorch is not installed,
  it registers a stand-in so the NumPy functions still import; calling a torch-dependent function then
  raises an error rather than producing numbers.
* The revision scripts use fixed seeds; the outputs in `results_revision/` were produced by `run_revision.sh`.
* `dfs_pipeline.foc_check` and its output `tabR13_prop1_FOC_check.*` are an early diagnostic that used the
  derivative as printed in the original submission; they are superseded and not used in the paper.
  Section 5.4 uses `r1_prop1_check.py` (and, for the original version, `dfs_pipeline_oracle.grad_decision`).
* `python r1_sign_theory.py s` re-summarises the saved 600-structure design without re-simulating it.
* DFS-I, DFS-Mix and DFS-TC are deterministic given the data (full-batch objective, fixed
  initialization); DFS-NN depends on the seed (Section 6.8).

## Second revision (R2)

The R2 changes respond to four reviewer comments. No result reported in R1 changed.

1. **Look-ahead in the rotating universe (Section 6.5).** `dfs_pipeline.build_P(rotating=True)` now selects
   constituents from estimation-window information only. `code/r2_survivor_pit.py` documents and recalculates
   this: `--check` (NumPy/pandas, seconds) compares the R1 and R2 selection rules rebalance by rebalance;
   without `--check` it re-runs the full survivorship analysis under the R2 rule (needs torch, a few minutes).
   Outputs in `results_r2/`: the selection comparison (`R2_survivor_selection_*`), the recalculated tables
   (`tabR11_survivorship_rotating_pit.*`, `tabR11b_survivorship_tests_pit.*`) and a side-by-side comparison
   with R1 (`R2_survivor_R1_vs_R2.csv`). The recalculated tables are identical to the R1 tables.
2. **Sign comparison (Section 6.7).** `code/r2_sign_comparison.py` re-tabulates the saved 191-universe results
   and compares the predicted sign of delta_D - delta_F with (a) the sign of delta* minus the mean Ledoit-Wolf
   intensity (as reported in R1) and (b) the sign of delta* minus the ex-post Frobenius-optimal constant
   intensity. No new simulation is run.
3. **Dispersion threshold (Section 5.3).** Text only; the numbers come from the existing
   `results_revision/R1_sign_random_design_*` files.
4. **Dividends.** Terminology made consistent (see "Data" above); no computation changed.

## License

Code: MIT License (see `LICENSE`). Price data are from Yahoo Finance and are provided only to
reproduce the published results; they remain subject to Yahoo's terms of use.
