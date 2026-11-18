"""Revision R1, reviewer comment 2: separating the theoretical gap G from the
empirical comparison.

Part A (real data). For each universe, realised variance over the 204 evaluation
rebalances under:
  LW-I        estimated, time-varying Ledoit-Wolf intensity (what the paper compared)
  LW-bar      the constant intensity equal to the average LW intensity
  F-const     the ex-post Frobenius-optimal CONSTANT intensity, using the realised
              holding-period covariance C_r as an unbiased proxy for Sigma_r:
                 delta = sum_r <C_r - S_r, D_r> / sum_r ||D_r||_F^2
              Because the Frobenius loss is robust to an unbiased noisy proxy
              (Patton 2011; Laurent, Rombouts and Violante 2013), this minimises the
              expected true Frobenius loss over the evaluation sample.
  F-r         the same, rebalance by rebalance (noisy; reported for completeness)
  D-const     the ex-post decision-optimal constant intensity (the 'oracle')
Empirical gaps reported relative to D-const:
  G_hat_F   = V(F-const)/V(D-const) - 1   (empirical analogue of G: both ex post,
                                           both constant, criterion is the only difference)
  G_hat_LW  = V(LW-I)/V(D-const) - 1      (what an implementable Frobenius rule loses)
  and the split of G_hat_LW into a level part (LW-bar vs D-const) and a
  time-variation part (LW-I vs LW-bar).

Part B (simulation with known Sigma). For Sigma set to (i) the full-sample covariance
of the 60 stocks, (ii) the full-sample covariance of the 10 ETFs, (iii) the factor DGP,
with N and T=250 matched to the empirical design, compute over 400 draws:
  G        = mean_draws [V(delta_F)/V(delta_D) - 1]          (the theoretical gap)
  G_LW     = mean_draws [V(delta_LW)/V(delta_D) - 1]
  G_hat_F  = mean V(delta_F,const)/mean V(delta_D,const) - 1  (the empirical-type quantity)
  G_hat_LW = mean V(delta_LW)/mean V(delta_D,const) - 1
which shows how the empirical-type quantities relate to G when Sigma is known.
"""
import os
import numpy as np
import pandas as pd
from _suite import jrs, OUT, build

GRID = np.linspace(0, 1, 101)


def part_a():
    rows, intens = [], []
    for uni in ["stocks", "etfs"]:
        P, c, Rr = build(uni)
        r0, h, R = c["dfs"]["min_train"], c["h"], P["R"]
        S, FI, H = P["S"], P["FI"], P["H"]
        D = FI - S
        # realised holding-period covariance (mean-zero convention, as in S)
        C = np.einsum("rti,rtj->rij", H, H) / h
        ev = range(r0, R)
        num = sum(np.sum((C[r] - S[r]) * D[r]) for r in ev)
        den = sum(np.sum(D[r] * D[r]) for r in ev)
        dF_const = float(np.clip(num / den, 0, 1))
        dF_r = np.array([np.clip(np.sum((C[r] - S[r]) * D[r]) / np.sum(D[r] * D[r]), 0, 1)
                         for r in range(R)])
        dLW = np.array([jrs.lw_identity(P["X"][r]) for r in range(R)])

        def var_for(deltas):
            W = np.array([jrs.gmv(jrs.shrink(S[r], FI[r], deltas[r])) for r in range(R)])
            met = jrs.evaluate(W, P, r0, c)[0]
            return met["realized_var_ann"], met["turnover_ann"]

        Vg = np.array([var_for(np.full(R, d))[0] for d in GRID])
        iD = int(np.argmin(Vg)); dD = GRID[iD]
        res = {
            "LW-I (estimated, time-varying)": (np.nanmean(dLW[r0:]), *var_for(dLW)),
            "LW-bar (constant, mean LW)": (np.mean(dLW[r0:]), *var_for(np.full(R, np.mean(dLW[r0:])))),
            "F-const (ex-post Frobenius-optimal)": (dF_const, *var_for(np.full(R, dF_const))),
            "F-r (ex-post Frobenius-optimal, per rebalance)": (np.mean(dF_r[r0:]), *var_for(dF_r)),
            "D-const (ex-post decision-optimal)": (dD, *var_for(np.full(R, dD))),
        }
        vD = res["D-const (ex-post decision-optimal)"][1]
        for k, (dl, v, to) in res.items():
            rows.append(dict(universe=uni, rule=k, mean_delta=dl, realized_var=v, turnover=to,
                             excess_vs_Dconst_pct=100 * (v / vD - 1)))
        vLW = res["LW-I (estimated, time-varying)"][1]
        vLWb = res["LW-bar (constant, mean LW)"][1]
        vF = res["F-const (ex-post Frobenius-optimal)"][1]
        intens.append(dict(universe=uni,
                           G_hat_F_pct=100 * (vF / vD - 1),
                           G_hat_LW_pct=100 * (vLW / vD - 1),
                           level_part_pct=100 * (vLWb / vD - 1),
                           time_variation_part_pct=100 * (vLW - vLWb) / vD,
                           LW_reduction_pct_of_LW=100 * (vLW - vD) / vLW))
        pd.DataFrame(dict(date=P["dates"], delta_LW=dLW, delta_F_r=dF_r)).to_csv(
            os.path.join(OUT, f"R1_gap_intensity_paths_{uni}.csv"), index=False)
    a = pd.DataFrame(rows); b = pd.DataFrame(intens)
    a.to_csv(os.path.join(OUT, "R1_gap_empirical_rules.csv"), index=False)
    b.to_csv(os.path.join(OUT, "R1_gap_empirical_summary.csv"), index=False)
    print(a.round(3).to_string()); print(b.round(2).to_string())


def part_b(M=400, T=250, seed=11):
    rng = np.random.default_rng(seed)
    sigmas = {}
    for uni in ["stocks", "etfs"]:
        _, _, Rr = build(uni)
        X = Rr.values - Rr.values.mean(0)
        sigmas[f"{uni}_empirical"] = X.T @ X / len(X)
    sigmas["factor_DGP_N60"] = jrs.make_sigma_factor(60, 3, rng)
    rows = []
    for nm, Sigma in sigmas.items():
        N = Sigma.shape[0]
        Vmat = np.zeros((M, len(GRID))); Fmat = np.zeros((M, len(GRID)))
        dLW = np.zeros(M); VLW = np.zeros(M)
        for m in range(M):
            X = jrs.sample_returns(Sigma, T, "gauss", 5, rng); X = X - X.mean(0)
            S = X.T @ X / T; F = jrs.target_identity(S)
            for i, d in enumerate(GRID):
                Sh = jrs.shrink(S, F, d); w = jrs.gmv(Sh)
                Vmat[m, i] = w @ Sigma @ w
                Fmat[m, i] = np.linalg.norm(Sh - Sigma, "fro") ** 2
            dLW[m] = jrs.lw_identity(X)
            w = jrs.gmv(jrs.shrink(S, F, dLW[m])); VLW[m] = w @ Sigma @ w
        iD = Vmat.argmin(1); iF = Fmat.argmin(1)
        VD = Vmat[np.arange(M), iD]; VF = Vmat[np.arange(M), iF]
        iDc = int(Vmat.mean(0).argmin()); iFc = int(Fmat.mean(0).argmin())
        rows.append(dict(
            true_Sigma=nm, N=N, T=T,
            mean_delta_D=GRID[iD].mean(), mean_delta_F=GRID[iF].mean(), mean_delta_LW=dLW.mean(),
            delta_D_const=GRID[iDc], delta_F_const=GRID[iFc],
            share_dD_gt_dF=float(np.mean(GRID[iD] > GRID[iF])),
            G_pct=100 * np.mean(VF / VD - 1),
            G_LW_pct=100 * np.mean(VLW / VD - 1),
            G_hat_F_pct=100 * (Vmat[:, iFc].mean() / Vmat[:, iDc].mean() - 1),
            G_hat_LW_pct=100 * (VLW.mean() / Vmat[:, iDc].mean() - 1),
            const_vs_drawwise_pct=100 * (Vmat[:, iDc].mean() / VD.mean() - 1),
        ))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "R1_gap_simulation_bridge.csv"), index=False)
    print(df.round(3).to_string())




def part_c():
    """Same-protocol control: walk-forward selection by realised FROBENIUS loss.

    FCV-G picks, at each rebalance, the constant intensity that minimises the
    Frobenius distance between the shrunk estimate and the realised holding-period
    covariance over all completed holding periods (closed form, clipped to [0,1]).
    DFS-G does the same with realised portfolio variance. The data, window,
    grid and timing are identical; only the selection loss differs.
    """
    import math
    rows, tests = [], []
    for uni in ["stocks", "etfs"]:
        P, c, Rr = build(uni)
        r0, h, R = c["dfs"]["min_train"], c["h"], P["R"]
        S, FI, H = P["S"], P["FI"], P["H"]
        D = FI - S
        C = np.einsum("rti,rtj->rij", H, H) / h
        a = np.array([np.sum((C[r] - S[r]) * D[r]) for r in range(R)])
        b = np.array([np.sum(D[r] * D[r]) for r in range(R)])
        Wg = np.array([[jrs.gmv(jrs.shrink(S[r], FI[r], d)) for r in range(R)] for d in GRID])
        Vg = np.einsum("rti,dri->drt", H, Wg)
        Vg = (Vg ** 2).mean(2) * 252
        dF = np.full(R, np.nan); dG = np.full(R, np.nan)
        WF = np.zeros_like(Wg[0]); WG = np.zeros_like(Wg[0])
        for r in range(r0, R):
            dF[r] = float(np.clip(a[:r].sum() / b[:r].sum(), 0, 1))
            WF[r] = jrs.gmv(jrs.shrink(S[r], FI[r], dF[r]))
            i = int(np.argmin(Vg[:, :r].mean(1))); dG[r] = GRID[i]; WG[r] = Wg[i, r]
        dLW = np.array([jrs.lw_identity(P["X"][r]) for r in range(R)])
        WL = np.array([jrs.gmv(jrs.shrink(S[r], FI[r], dLW[r])) for r in range(R)])
        store = {}
        for nm, W, dl in [("LW-I", WL, dLW), ("FCV-G (Frobenius walk-forward)", WF, dF),
                          ("DFS-G (decision walk-forward)", WG, dG)]:
            met, vr, rets, dates, to_r, nets = jrs.evaluate(W, P, r0, c)
            store[nm] = rets
            rows.append(dict(universe=uni, method=nm, mean_delta=np.nanmean(dl[r0:]),
                             realized_var=met["realized_var_ann"], turnover=met["turnover_ann"],
                             sharpe_net_25bp=met["sharpe_net_25bp"]))
        n = len(store["LW-I"])
        rng = np.random.default_rng(c["boot"]["seed"])
        idx = jrs.stationary_bootstrap_idx(n, c["boot"]["block"], c["boot"]["B"], rng)
        for x, y in [("DFS-G (decision walk-forward)", "FCV-G (Frobenius walk-forward)"),
                     ("FCV-G (Frobenius walk-forward)", "LW-I")]:
            rv, _ = jrs.boot_test(store[x], store[y], jrs.logvar_stat, c, idx=idx)
            tests.append(dict(universe=uni, comparison=f"{x} vs {y}",
                              var_ratio=math.exp(rv["stat"]), t=rv["t"], p_boot=rv["p_boot"]))
        pd.DataFrame(dict(date=P["dates"], delta_FCV=dF, delta_DFSG=dG, delta_LW=dLW)).to_csv(
            os.path.join(OUT, f"R1_fcv_vs_dfsg_paths_{uni}.csv"), index=False)
    a = pd.DataFrame(rows); t = pd.DataFrame(tests)
    a.to_csv(os.path.join(OUT, "R1_fcv_vs_dfsg.csv"), index=False)
    t.to_csv(os.path.join(OUT, "R1_fcv_vs_dfsg_tests.csv"), index=False)
    print(a.round(3).to_string()); print(t.round(4).to_string())


if __name__ == "__main__":
    import sys
    parts = sys.argv[1:] or ["a", "b", "c"]
    for p_ in parts:
        {"a": part_a, "b": part_b, "c": part_c}[p_]()
