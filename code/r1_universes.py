"""Revision R1, reviewer comment 3: does covariance structure predict the
decision-optimal intensity out of sample on real data, beyond the two original
universes?

Protocol per universe is identical to the paper (L=250, h=21, 204 evaluation
rebalances from 30 Nov 2007, identity target, unconstrained GMV).

Universes
  Named (11): the two original universes plus nine designed to move volatility
  dispersion and correlation structure (all 90 stocks; low-, high- and barbell-
  volatility stock sets; stocks plus Treasury/credit ETFs; equity-only ETFs;
  equity ETFs plus gold; bonds plus gold).
  Random (180): 60 random 30-stock subsets, 90 random mixed universes (10/20/30
  random stocks plus 1-6 random ETFs), 30 random ETF subsets of size 4-8.

For each universe we record
  ex-ante structure, estimated ONLY on the pre-evaluation sample (2 Dec 2004 to
  the first evaluation date, 754 trading days): volatility dispersion, mean
  correlation, Lambda_T, and the calibrated-simulation predictions of delta_D
  and delta_F (true Sigma := pre-evaluation covariance, T=250, 60 draws);
  realised outcomes over the evaluation period: the ex-post decision-optimal
  constant intensity delta*, the mean Ledoit-Wolf intensity, the ex-post
  Frobenius-optimal constant intensity (realised-covariance proxy), and the
  realised variance of LW-I, DFS-G, FCV-G and the oracle.
"""
import os
import numpy as np
import pandas as pd
from _suite import jrs, OUT, DATA, CFG
from r1_sign_theory import theory

GRID = np.linspace(0, 1, 101)
L, h, R0 = CFG["L"], CFG["h"], CFG["dfs"]["min_train"]


def returns_panel():
    ps = pd.read_csv(os.path.join(DATA, "prices_stocks.csv"), index_col=0, parse_dates=True)
    pe = pd.read_csv(os.path.join(DATA, "prices_etfs.csv"), index_col=0, parse_dates=True)
    Rs = jrs.clean_universe(ps, CFG, None)
    Re = jrs.clean_universe(pe, CFG, None)
    R = Rs.join(Re, how="inner")
    return R, list(Rs.columns), list(Re.columns)


def gmv_grid(S, grid):
    """GMV weights for (1-d)S + d*mI for all d in grid via one eigendecomposition."""
    lam, Q = np.linalg.eigh(S)
    m = lam.mean()
    q1 = Q.T @ np.ones(S.shape[0])
    den = (1 - grid)[:, None] * lam[None, :] + grid[:, None] * m       # G x N
    U = (q1[None, :] / den) @ Q.T                                     # G x N
    return U / U.sum(1, keepdims=True)


def evaluate_universe(A, dates):
    """A: T_all x N returns. Returns dict of outcomes."""
    D_, N = A.shape
    starts = list(range(L, D_ - h + 1, h))
    Rn = len(starts)
    Wg = np.zeros((len(GRID), Rn, N)); Vsel = np.zeros((len(GRID), Rn))
    rets = np.zeros((len(GRID), Rn, h))
    dLW = np.zeros(Rn); a = np.zeros(Rn); b = np.zeros(Rn)
    WLW = np.zeros((Rn, N))
    for k, t in enumerate(starts):
        X = A[t - L:t]; X = X - X.mean(0); S = X.T @ X / L
        H = A[t:t + h]
        W = gmv_grid(S, GRID); Wg[:, k] = W
        pr = H @ W.T                                                   # h x G
        rets[:, k] = pr.T; Vsel[:, k] = (pr ** 2).mean(0) * 252
        dLW[k] = jrs.lw_identity(X)
        WLW[k] = jrs.gmv(jrs.shrink(S, jrs.target_identity(S), dLW[k]))
        C = H.T @ H / h; Dm = jrs.target_identity(S) - S
        a[k] = np.sum((C - S) * Dm); b[k] = np.sum(Dm * Dm)
    ev = slice(R0, Rn)
    allret = rets[:, ev].reshape(len(GRID), -1)
    Vfull = 1e4 * allret.var(1) * 252                                  # realised variance per constant delta
    istar = int(Vfull.argmin())
    out = dict(delta_star=GRID[istar], V_oracle=Vfull[istar], V_delta0=Vfull[0], V_delta1=Vfull[-1])
    # LW-I
    Hs = np.stack([A[t:t + h] for t in starts])
    lw_r = np.einsum("rti,ri->rt", Hs[ev], WLW[ev]).ravel()
    out["V_LWI"] = 1e4 * lw_r.var() * 252; out["mean_delta_LW"] = dLW[ev].mean()
    # ex-post Frobenius constant
    dFc = float(np.clip(a[ev].sum() / b[ev].sum(), 0, 1)); out["delta_F_expost"] = dFc
    iF = int(np.argmin(np.abs(GRID - dFc))); out["V_Fexpost"] = Vfull[iF]
    # walk-forward DFS-G and FCV-G
    dg = np.zeros(Rn); df_ = np.zeros(Rn); rg, rf = [], []
    for k in range(R0, Rn):
        i = int(Vsel[:, :k].mean(1).argmin()); dg[k] = GRID[i]; rg.append(rets[i, k])
        dfk = float(np.clip(a[:k].sum() / b[:k].sum(), 0, 1)); df_[k] = dfk
        j = int(np.argmin(np.abs(GRID - dfk))); rf.append(rets[j, k])
    out["V_DFSG"] = 1e4 * np.concatenate(rg).var() * 252; out["mean_delta_DFSG"] = dg[ev].mean()
    out["V_FCVG"] = 1e4 * np.concatenate(rf).var() * 252; out["mean_delta_FCVG"] = df_[ev].mean()
    out["first_eval_index"] = starts[R0]
    return out


def calibrated_prediction(Sigma, T=250, M=60, rng=None):
    N = Sigma.shape[0]
    Lc = np.linalg.cholesky(Sigma + 1e-14 * np.eye(N))
    dD, dF = np.zeros(M), np.zeros(M)
    for k in range(M):
        X = rng.standard_normal((T, N)) @ Lc.T; X = X - X.mean(0); S = X.T @ X / T
        W = gmv_grid(S, GRID)
        V = np.einsum("gi,ij,gj->g", W, Sigma, W)
        m = np.trace(S) / N
        Dm = m * np.eye(N) - S
        dF[k] = np.clip(np.sum((Sigma - S) * Dm) / np.sum(Dm * Dm), 0, 1)
        dD[k] = GRID[int(V.argmin())]
    return dD.mean(), dF.mean(), np.mean(dD > dF)


def structure(A_pre, T=250):
    X = A_pre - A_pre.mean(0); Sigma = X.T @ X / len(X)
    th = theory(Sigma, T)
    return Sigma, {k: th[k] for k in ["Lambda", "Lambda_T", "vol_disp", "vol_cv", "mean_corr",
                                      "top_eig_share", "dD_approx_fs", "dF_approx"]}


def universes(stocks, etfs, rng):
    sd = None
    U = {}
    U["Stocks60 (original)"] = stocks[:60]
    U["ETF10 (original)"] = etfs
    U["Stocks90 (all)"] = stocks
    return U


def main():
    rng = np.random.default_rng(31)
    R, stocks, etfs = returns_panel()
    vol_pre = R.iloc[:754].std() * np.sqrt(252)
    st_sorted = list(vol_pre[stocks].sort_values().index)
    eq_etf = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ"]
    named = {
        "Stocks60 (original)": stocks[:60],
        "ETF10 (original)": etfs,
        "Stocks90 (all stocks)": stocks,
        "Stocks30 low-vol": st_sorted[:30],
        "Stocks30 high-vol": st_sorted[-30:],
        "Stocks30 barbell (15 low + 15 high vol)": st_sorted[:15] + st_sorted[-15:],
        "Stocks60 + IEF": stocks[:60] + ["IEF"],
        "Stocks60 + TLT, IEF, LQD": stocks[:60] + ["TLT", "IEF", "LQD"],
        "Equity ETFs (6)": eq_etf,
        "Equity ETFs + GLD (7)": eq_etf + ["GLD"],
        "Bonds + gold (TLT, IEF, LQD, GLD)": ["TLT", "IEF", "LQD", "GLD"],
    }
    rand = {}
    for j in range(60):
        rand[f"rand_stocks30_{j}"] = list(rng.choice(stocks, 30, replace=False))
    for j in range(90):
        ns = int(rng.choice([10, 20, 30])); ne = int(rng.integers(1, 7))
        rand[f"rand_mixed_{j}"] = list(rng.choice(stocks, ns, replace=False)) + \
            list(rng.choice(etfs, ne, replace=False))
    for j in range(30):
        ne = int(rng.integers(4, 9))
        rand[f"rand_etfs_{j}"] = list(rng.choice(etfs, ne, replace=False))
    rows = []
    for kind, dct in [("named", named), ("random", rand)]:
        for nm, cols in dct.items():
            A = R[cols].values
            res = evaluate_universe(A, R.index)
            pre = A[:res["first_eval_index"]]           # strictly before the first evaluation date
            Sigma_pre, st = structure(pre)
            pdD, pdF, pshare = calibrated_prediction(Sigma_pre, rng=rng)
            _, st_full = structure(A)
            rows.append(dict(kind=kind, universe=nm, N=len(cols), **st,
                             Lambda_T_full=st_full["Lambda_T"], vol_disp_full=st_full["vol_disp"],
                             pred_delta_D=pdD, pred_delta_F=pdF, pred_share_dD_gt_dF=pshare, **res))
            if kind == "named" or len(rows) % 20 == 0:
                r_ = rows[-1]
                print(f"{nm[:40]:40s} N={r_['N']:3d} disp={r_['vol_disp']:.1f} LamT={r_['Lambda_T']:.2f} "
                      f"pred dD={pdD:.3f} dF={pdF:.3f} | d*={r_['delta_star']:.2f} LW={r_['mean_delta_LW']:.3f} "
                      f"dFexp={r_['delta_F_expost']:.2f} | V LW={r_['V_LWI']:.1f} DFSG={r_['V_DFSG']:.1f} "
                      f"FCVG={r_['V_FCVG']:.1f} orc={r_['V_oracle']:.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "R1_universes_raw.csv"), index=False)
    summarise(df)


def summarise(df):
    from scipy.stats import spearmanr
    d = df.copy()
    d["sign_emp"] = np.sign(d.delta_star - d.mean_delta_LW)
    d["pred_sign"] = np.where(d.Lambda_T < 1, 1, -1)
    d["pred_sign_sim"] = np.sign(d.pred_delta_D - d.pred_delta_F)
    d["DFSG_vs_LW_pct"] = 100 * (d.V_DFSG / d.V_LWI - 1)
    d["FCVG_vs_LW_pct"] = 100 * (d.V_FCVG / d.V_LWI - 1)
    d["DFSG_vs_FCVG_pct"] = 100 * (d.V_DFSG / d.V_FCVG - 1)
    d["DFSG_pct_of_oracle_gap"] = 100 * (d.V_LWI - d.V_DFSG) / (d.V_LWI - d.V_oracle).where(d.V_LWI > d.V_oracle)
    out = {}
    for nm, g in [("all", d), ("random", d[d.kind == "random"]), ("named", d[d.kind == "named"])]:
        g = g[d.sign_emp != 0] if False else g
        out[f"{nm}_n"] = len(g)
        out[f"{nm}_share_delta_star_gt_LW"] = (g.delta_star > g.mean_delta_LW).mean()
        out[f"{nm}_Lambda_rule_acc"] = (g.pred_sign == g.sign_emp).mean()
        out[f"{nm}_simpred_rule_acc"] = (g.pred_sign_sim == g.sign_emp).mean()
        out[f"{nm}_spearman_pred_dD_vs_delta_star"] = spearmanr(g.pred_delta_D, g.delta_star)[0]
        out[f"{nm}_spearman_logdisp_vs_delta_star"] = spearmanr(np.log(g.vol_disp), g.delta_star)[0]
        out[f"{nm}_spearman_LambdaT_vs_delta_star_minus_LW"] = spearmanr(
            np.log(g.Lambda_T), g.delta_star - g.mean_delta_LW)[0]
        out[f"{nm}_share_DFSG_below_LW"] = (g.V_DFSG < g.V_LWI).mean()
        out[f"{nm}_share_DFSG_below_FCVG"] = (g.V_DFSG < g.V_FCVG).mean()
        out[f"{nm}_median_DFSG_vs_LW_pct"] = g.DFSG_vs_LW_pct.median()
        out[f"{nm}_median_DFSG_vs_FCVG_pct"] = g.DFSG_vs_FCVG_pct.median()
        out[f"{nm}_share_Fexpost_gt_delta_star"] = (g.delta_F_expost > g.delta_star).mean()
    s = pd.Series(out)
    s.to_csv(os.path.join(OUT, "R1_universes_summary.csv"))
    print(s.round(3).to_string())
    # dispersion buckets (random universes)
    rnd = d[d.kind == "random"].copy()
    rnd["disp_bucket"] = pd.cut(rnd.vol_disp, [1, 2, 3, 5, 8, 100])
    tab = rnd.groupby("disp_bucket", observed=True).agg(
        n=("delta_star", "size"), mean_delta_star=("delta_star", "mean"),
        mean_delta_LW=("mean_delta_LW", "mean"), share_dstar_gt_LW=("sign_emp", lambda s: (s > 0).mean()),
        share_LamT_lt1=("Lambda_T", lambda s: (s < 1).mean()),
        median_DFSG_vs_LW=("DFSG_vs_LW_pct", "median"), median_DFSG_vs_FCVG=("DFSG_vs_FCVG_pct", "median"))
    tab.to_csv(os.path.join(OUT, "R1_universes_by_dispersion.csv"))
    print(tab.round(3).to_string())
    d[d.kind == "named"].to_csv(os.path.join(OUT, "R1_universes_named.csv"), index=False)
    return s


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "s":
        summarise(pd.read_csv(os.path.join(OUT, "R1_universes_raw.csv")))
    else:
        main()
