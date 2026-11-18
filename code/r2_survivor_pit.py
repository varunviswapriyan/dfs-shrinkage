"""
R2, reviewer comment 1: rotating point-in-time universe without look-ahead.

The R1 rule (dfs_pipeline.build_P, rotating=True) had two uses of future information:
  (a) a name was eligible at rebalance t only if it had valid prices over the estimation
      window AND the coming holding period:      valid[t-L : t+h].all(0)
  (b) a name entered the candidate pool only if its FULL-SAMPLE count of valid prices
      exceeded L + 500, which also looks past t.

The corrected rule uses only information available at the formation date t:
  * eligibility: valid prices over the estimation window only,  valid[t-L : t].all(0)
  * ranking:     trailing dollar volume over [t-L, t), as before
  * no full-sample filter on the candidate pool
  * a name whose price stops during the holding period is held at its last price,
    i.e. it earns a zero return for the rest of that period (proceeds treated as cash).
    This is what the existing return construction (forward-fill, then NaN -> 0) does.

Everything else (estimators, DFS training, seeds, bootstrap) is unchanged, so the
new tables are directly comparable to results_original/tabR11*.

The corrected rule is also the one now implemented in dfs_pipeline.build_P(rotating=True).
This script keeps its own copy of the rule (it patches build_P at run time) so that the
outputs in results_r2/ can be reproduced exactly as they were produced, and so that
--check can reconstruct the R1 rule for comparison.

Result (R2): the corrected rule selects the same 60 constituents at all 228 rebalances,
and results_r2/tabR11*_pit.csv are byte-identical to results_original/tabR11*.csv.

Usage (from the code/ folder):
    python3 r2_survivor_pit.py --check     # selection diagnostics only; NumPy/pandas, seconds
    python3 r2_survivor_pit.py             # full rerun of Section 6.5; needs torch
Outputs go to ../results_r2/.
"""
import os, sys, json, math, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore", message="Mean of empty slice")   # MMC has no cached prices
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results_r2")
L, H, NSUB = 250, 21, 60          # same as dfs_pipeline.CFG


def load_stock_data():
    px = pd.read_csv(os.path.join(DATA, "prices_stocks.csv"), index_col=0, parse_dates=True)
    vol = pd.read_csv(os.path.join(DATA, "volume_stocks.csv"), index_col=0, parse_dates=True)
    return px, vol


def selection_inputs(px, vol, full_sample_filter):
    """Reproduces the data preparation in dfs_pipeline.build_P(rotating=True)."""
    p = px.ffill(limit=5)
    keep = [t for t in p.columns if p[t].notna().sum() > L + 500] if full_sample_filter else list(p.columns)
    p = p[keep]
    R = p.pct_change().clip(-0.5, 0.5)
    valid = p.notna().values
    dv = (p * vol[keep].reindex(p.index).ffill()).values
    return keep, R, valid, dv


def make_cols_fn(valid, dv, nsub, lookahead):
    def cols_fn(t, L=L, h=H):
        end = t + h if lookahead else t                     # <- the fix
        ok = valid[t - L:end].all(0)
        score = np.where(ok, np.nanmean(dv[t - L:t], axis=0), -np.inf)
        return np.sort(np.argsort(-score)[:nsub])
    return cols_fn


def check():
    """Compare old and new constituent lists rebalance by rebalance (no torch needed)."""
    os.makedirs(OUT, exist_ok=True)
    px, vol = load_stock_data()
    k_old, R_old, v_old, dv_old = selection_inputs(px, vol, full_sample_filter=True)
    k_new, R_new, v_new, dv_new = selection_inputs(px, vol, full_sample_filter=False)
    nsub = min(NSUB, len(k_old))
    f_old = make_cols_fn(v_old, dv_old, nsub, lookahead=True)
    f_new = make_cols_fn(v_new, dv_new, nsub, lookahead=False)
    D = len(R_old)
    starts = list(range(L, D - H + 1, H))
    rows = []
    for r, t in enumerate(starts):
        old = {k_old[i] for i in f_old(t)}
        new = {k_new[i] for i in f_new(t)}
        cols_new = [k_new.index(n) for n in sorted(new)]
        # names in the new list whose price series stops inside the holding period
        died = [k_new[i] for i in cols_new if not v_new[t:t + H, i].all()]
        rows.append(dict(rebalance=r, date=R_old.index[t].date(),
                         n_old=len(old), n_new=len(new),
                         only_old=" ".join(sorted(old - new)), only_new=" ".join(sorted(new - old)),
                         n_diff=len(old ^ new) // 2, n_missing_in_holding=len(died),
                         missing_in_holding=" ".join(died)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "R2_survivor_selection_diff.csv"), index=False)
    ev = df.iloc[24:]                                      # evaluation rebalances (first 24 = training)
    summ = dict(
        candidates_old=len(k_old), candidates_new=len(k_new),
        dropped_by_full_sample_filter=sorted(set(k_new) - set(k_old)),
        rebalances=len(df), eval_rebalances=len(ev),
        rebalances_with_different_constituents=int((df.n_diff > 0).sum()),
        eval_rebalances_with_different_constituents=int((ev.n_diff > 0).sum()),
        mean_names_swapped_per_rebalance=float(df.n_diff.mean()),
        max_names_swapped=int(df.n_diff.max()),
        holdings_with_missing_prices_in_holding_period=int(df.n_missing_in_holding.sum()),
    )
    with open(os.path.join(OUT, "R2_survivor_selection_summary.json"), "w") as f:
        json.dump(summ, f, indent=2, default=str)
    print(json.dumps(summ, indent=2, default=str))
    if summ["rebalances_with_different_constituents"] == 0:
        print("\nThe corrected rule selects exactly the same constituents at every rebalance, so the\n"
              "Section 6.5 results are unchanged; the full rerun will reproduce tabR11 exactly.")
    return summ


def full_run():
    sys.path.insert(0, HERE)
    import torch
    import dfs_pipeline as dp

    def build_P_pit(universe, cfg, out, L=None, h=None, rotating=False):
        if not rotating:
            return _orig_build_P(universe, cfg, out, L=L, h=h, rotating=False)
        c = json.loads(json.dumps(cfg)); c["dfs"] = cfg["dfs"]
        if L: c["L"] = L
        if h: c["h"] = h
        px, vol = dp.load_prices(cfg["stocks"], cfg, out, "stocks")
        keep, R, valid, dv = selection_inputs(px, vol, full_sample_filter=False)
        nsub = min(cfg["n_stocks"], len(keep))
        Rf = R.fillna(0.0)
        cols_fn = make_cols_fn(valid, dv, nsub, lookahead=False)
        P = dp.precompute(Rf, c, cols_fn=cols_fn)
        P["R_index"] = Rf.index
        return P, c, Rf

    _orig_build_P = dp.build_P
    dp.build_P = build_P_pit

    os.makedirs(OUT, exist_ok=True)
    for f in ("prices_stocks.csv", "volume_stocks.csv"):          # cached data, no download
        src, dst = os.path.join(DATA, f), os.path.join(OUT, f)
        if not os.path.exists(dst):
            pd.read_csv(src, index_col=0).to_csv(dst)
    np.random.seed(dp.CFG["seed"]); torch.manual_seed(dp.CFG["seed"])
    logf = os.path.join(OUT, "r2_survivor_run.log")

    # part_survivor saves under the old names; run it, then rename to *_pit
    summary = dp.part_survivor(OUT, dp.CFG, logf)
    for ext in ("csv", "tex"):
        for name in ("tabR11_survivorship_rotating", "tabR11b_survivorship_tests"):
            src = os.path.join(OUT, f"{name}.{ext}")
            if os.path.exists(src):
                os.replace(src, os.path.join(OUT, f"{name}_pit.{ext}"))

    # side-by-side with the R1 (look-ahead) numbers
    old = pd.read_csv(os.path.join(ROOT, "results_original", "tabR11_survivorship_rotating.csv"), index_col=0)
    new = pd.read_csv(os.path.join(OUT, "tabR11_survivorship_rotating_pit.csv"), index_col=0)
    cols = ["realized_var_ann", "turnover_ann", "sharpe_net_25bp"]
    cmp_ = old[cols].add_suffix("_R1").join(new[cols].add_suffix("_R2"))
    lw_old, lw_new = old.loc["lw_identity", "realized_var_ann"], new.loc["lw_identity", "realized_var_ann"]
    cmp_["pct_vs_LWI_R1"] = 100 * (old["realized_var_ann"] / lw_old - 1)
    cmp_["pct_vs_LWI_R2"] = 100 * (new["realized_var_ann"] / lw_new - 1)
    cmp_["rank_R1"] = old["realized_var_ann"].rank().astype(int)
    cmp_["rank_R2"] = new["realized_var_ann"].rank().astype(int)
    cmp_ = cmp_.sort_values("realized_var_ann_R2").round(4)
    cmp_.to_csv(os.path.join(OUT, "R2_survivor_R1_vs_R2.csv"))
    print(cmp_.to_string())
    with open(os.path.join(OUT, "R2_survivor_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="selection diagnostics only (no torch)")
    a = ap.parse_args()
    if a.check:
        check()
    else:
        check()
        full_run()
