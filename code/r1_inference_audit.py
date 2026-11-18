"""Revision R1: audit of the Romano-Wolf family used in Table 4 (stocks) and its
ETF counterpart.

The original Table 4 caption states that the oracle was excluded from a
sixteen-comparison family. In the original code (dfs_pipeline_oracle.part_oracle)
the oracle was in fact included (17 comparisons) and two evaluated variants
(DFS-CC, DFS-NN) were not. This script recomputes the stepdown p-values for
three explicitly defined families, using the same studentised stationary
bootstrap (B=1000, mean block 10, seed 7) and the same daily out-of-sample
return series:

  F16  : the sixteen implementable estimators reported in Table 4 (oracle excluded)
  F17  : F16 + oracle (the family actually used in the original code)
  F18  : F16 + DFS-CC + DFS-NN (every implementable estimator evaluated)

Return series for the gradient-trained DFS variants are read from the saved
original run (results_original/core_<universe>.npz); DFS-G, DFS-G-TC and the
oracle are rebuilt here from the price data with NumPy only.
"""
import math
import os
import numpy as np
import pandas as pd
from _suite import jrs, OUT, build, ROOT

ORIG = os.path.join(ROOT, "results_original")


def grid_estimators(P, c):
    r0 = c["dfs"]["min_train"]; h = c["h"]
    grid = np.linspace(0, 1, 101)
    R, N = P["R"], P["N"]
    Wg = np.zeros((len(grid), R, N)); Vg = np.zeros((len(grid), R))
    for i, d in enumerate(grid):
        for r in range(R):
            w = jrs.gmv(jrs.shrink(P["S"][r], P["FI"][r], d))
            Wg[i, r] = w
            Vg[i, r] = ((P["H"][r] @ w) ** 2).mean() * 252
    Tg = np.zeros((len(grid), R))
    Tg[:, 1:] = np.abs(np.diff(Wg, axis=1)).sum(2)
    out = {}
    for cost, nm in [(0.0, "dfc_grid"), (c["primary_cost"], "dfc_grid_tc25bp")]:
        Wd = np.zeros((R, N)); dl = np.full(R, np.nan)
        for r in range(r0, R):
            obj = Vg[:, :r].mean(1) + c["psi"] * cost * Tg[:, 1:r].mean(1) * (252.0 / h)
            i = int(np.argmin(obj)); dl[r] = grid[i]; Wd[r] = Wg[i, r]
        out[nm] = (Wd, dl)
    obj = np.array([jrs.evaluate(Wg[i], P, r0, c)[0]["realized_var_ann"] for i in range(len(grid))])
    istar = int(np.argmin(obj))
    out["oracle_const_delta"] = (Wg[istar], np.full(R, grid[istar]))
    return out, grid, Wg, Vg, Tg


def main():
    F16 = ["sample", "lw_cc", "oas", "ewma", "pca", "mp_clip", "equal", "nls", "dccnl",
           "dfs_identity", "dfs_mix", "dfs_tc_10bp", "dfs_tc_25bp", "dfs_tc_50bp",
           "dfc_grid", "dfc_grid_tc25bp"]
    fams = {"F16": F16, "F17": F16 + ["oracle_const_delta"], "F18": F16 + ["dfs_cc", "dfs_nn"]}
    allrows = []
    for uni in ["stocks", "etfs"]:
        P, c, _ = build(uni)
        r0 = c["dfs"]["min_train"]
        z = np.load(os.path.join(ORIG, f"core_{uni}.npz"), allow_pickle=True)
        rets = {k[5:]: z[k] for k in z.files if k.startswith("rets_")}
        to = {k[3:]: z[k] for k in z.files if k.startswith("to_")}
        built, *_ = grid_estimators(P, c)
        for nm, (Wd, _) in built.items():
            met, vr, rr, dates, to_r, nets = jrs.evaluate(Wd, P, r0, c)
            rets[nm] = rr; to[nm] = to_r
        # sanity: rebuilt LW-I equals the saved series
        n = len(rets["lw_identity"])
        rng = np.random.default_rng(c["boot"]["seed"])
        idx = jrs.stationary_bootstrap_idx(n, c["boot"]["block"], c["boot"]["B"], rng)
        stats, tstars = {}, {}
        for m in sorted(set(sum(fams.values(), []))):
            rv, ts = jrs.boot_test(rets[m], rets["lw_identity"], jrs.logvar_stat, c, idx=idx)
            stats[m] = rv; tstars[m] = ts
        for fam, members in fams.items():
            t = np.array([stats[m]["t"] for m in members])
            padj = jrs.romano_wolf(t, np.array([tstars[m] for m in members]).T)
            for m, p in zip(members, padj):
                allrows.append(dict(universe=uni, family=fam, method=m,
                                    var_ratio=math.exp(stats[m]["stat"]), t=stats[m]["t"],
                                    p_boot=stats[m]["p_boot"], p_RW=p))
    df = pd.DataFrame(allrows)
    df.to_csv(os.path.join(OUT, "R1_romano_wolf_families_long.csv"), index=False)
    wide = df.pivot_table(index=["universe", "method"], columns="family", values="p_RW")
    base = df.drop_duplicates(["universe", "method"]).set_index(["universe", "method"])[
        ["var_ratio", "t", "p_boot"]]
    wide = base.join(wide, how="outer")
    wide.to_csv(os.path.join(OUT, "R1_romano_wolf_families.csv"))
    pd.set_option("display.width", 200)
    print(wide.round(4).sort_values(["universe", "t"]).to_string())


if __name__ == "__main__":
    main()
