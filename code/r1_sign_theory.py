"""Revision R1, reviewer comment 3: a theoretical condition for the sign of
delta_D - delta_F, and a broad randomised simulation design to test it.

Second-order approximation (Gaussian returns, known mean, T large):
  S = Sigma + E,  E[E]=0,  Cov(E_ij, E_kl) = (Sigma_ik Sigma_jl + Sigma_il Sigma_jk)/T
  target F = m_hat I, m_hat = tr(S)/N = m + tau, m = tr(Sigma)/N, tau = tr(E)/N
  w* = Sigma^{-1}1/c*,  V* = 1/c*,  M* = Sigma^{-1} - Sigma^{-1}11'Sigma^{-1}/c*
  V(delta) - V* ~= w*' Delta M* Delta w*,  Delta = (1-delta)E + delta(B + tau I),  B = mI - Sigma

  E[V - V*] = (1-d)^2 a + d^2 (beta + g) + 2 d (1-d) chi, with
     a    = (N-1) V* / T                           (noise, decision metric)
     beta = m^2 w*' M* w*                          (bias,  decision metric)
     g    = 2 tr(Sigma^2)/(N^2 T) * w*' M* w*
     chi  = 2 w*' Sigma^2 M* w* / (N T)
  => delta_D^approx = (a - chi) / (a + beta + g - 2 chi)

  E||Delta||_F^2 = (1-d)^2 a_F + d^2 (beta_F + g_F) + 2 d (1-d) g_F, with
     a_F = (tr(Sigma^2) + tr(Sigma)^2)/T,  beta_F = ||Sigma - mI||_F^2,  g_F = 2 tr(Sigma^2)/(N T)
  => delta_F^approx = (a_F - g_F) / (a_F + beta_F - g_F)

  Leading order (dropping the O(1/T) tau terms) gives delta = noise/(noise+bias) under
  both criteria, so the sign of delta_D - delta_F is fixed by the T-free index
     Lambda = [beta / (a T)] / [beta_F / (a_F T)]
            = [m^2 w*'M*w* / ((N-1) V*)] * [(tr Sigma^2 + tr(Sigma)^2) / ||Sigma - mI||_F^2]
  delta_D > delta_F  <=>  Lambda < 1 (to second order).
  beta = 0 iff 1 is an eigenvector of Sigma (true GMV equally weighted) -> delta_D = 1.

Part 1 verifies the approximations on the three DGPs of Section 4.
Part 2 draws 600 random covariance structures from a broad family (number of
factors 0-5, factor strength, loading dispersion, volatility dispersion 1x-30x,
average correlation, block structure, N in {10,30,60,120}, T in {60,...,1000}),
simulates 40 samples each, and asks how well (i) Lambda and (ii) volatility
dispersion alone predict the sign of delta_D - delta_F.
"""
import os
import numpy as np
import pandas as pd
from _suite import jrs, OUT

GRID = np.linspace(0, 1, 201)


def theory(Sigma, T):
    N = Sigma.shape[0]
    one = np.ones(N)
    Si = np.linalg.inv(Sigma)
    u = Si @ one; c = one @ u; w = u / c; V = 1 / c
    M = Si - np.outer(u, u) / c
    m = np.trace(Sigma) / N
    trS2 = np.trace(Sigma @ Sigma)
    wMw = w @ M @ w
    a = (N - 1) * V / T
    beta = m ** 2 * wMw
    g = 2 * trS2 / (N ** 2 * T) * wMw
    chi = 2 * w @ Sigma @ Sigma @ M @ w / (N * T)
    aF = (trS2 + np.trace(Sigma) ** 2) / T
    betaF = np.linalg.norm(Sigma - m * np.eye(N), "fro") ** 2
    gF = 2 * trS2 / (N * T)
    dD = np.clip((a - chi) / (a + beta + g - 2 * chi), 0, 1)
    dF = np.clip((aF - gF) / (aF + betaF - gF), 0, 1)
    Lam = (beta / (a * T)) / (betaF / (aF * T)) if betaF > 0 else np.inf
    # finite-sample refinement: GMV noise inflates by ~T/(T-N) rather than 1 (Wishart moments)
    a2 = (N - 1) * V / max(T - N - 1, 1)
    dD2 = np.clip((a2 - chi) / (a2 + beta + g - 2 * chi), 0, 1)
    Lam_T = Lam * max(T - N - 1, 1) / T
    sd = np.sqrt(np.diag(Sigma))
    C = Sigma / np.outer(sd, sd)
    ev = np.linalg.eigvalsh(C)
    return dict(dD_approx=dD, dD_approx_fs=dD2, dF_approx=dF, Lambda=Lam, log_Lambda=np.log(Lam),
                Lambda_T=Lam_T,
                vol_disp=sd.max() / sd.min(), vol_cv=sd.std() / sd.mean(),
                mean_corr=(C.sum() - N) / (N * (N - 1)), top_eig_share=ev[-1] / N,
                w_eff_n=1 / np.sum(w ** 2), gross=np.abs(w).sum())


def simulate(Sigma, T, M, rng, dist="gauss"):
    N = Sigma.shape[0]
    dDs, dFs, gaps = np.zeros(M), np.zeros(M), np.zeros(M)
    for k in range(M):
        X = jrs.sample_returns(Sigma, T, dist, 5, rng); X = X - X.mean(0)
        S = X.T @ X / T; F = jrs.target_identity(S)
        V = np.empty(len(GRID)); Fr = np.empty(len(GRID))
        for i, d in enumerate(GRID):
            Sh = (1 - d) * S + d * F
            w = jrs.gmv(Sh); V[i] = w @ Sigma @ w
            Fr[i] = np.sum((Sh - Sigma) ** 2)
        iD, iF = int(V.argmin()), int(Fr.argmin())
        dDs[k], dFs[k], gaps[k] = GRID[iD], GRID[iF], V[iF] / V[iD] - 1
    return dict(dD_sim=dDs.mean(), dF_sim=dFs.mean(), share_dD_gt_dF=np.mean(dDs > dFs),
                share_dD_lt_dF=np.mean(dDs < dFs), G_pct=100 * np.mean(gaps),
                diff_sim=dDs.mean() - dFs.mean())


def random_sigma(rng, N):
    """Broad random covariance family."""
    K = int(rng.integers(0, 6))
    disp = float(np.exp(rng.uniform(0, np.log(30))))          # target vol dispersion max/min
    base = 0.01
    # correlation part: factors + optional blocks
    B = np.zeros((N, K))
    if K > 0:
        B[:, 0] = rng.normal(rng.uniform(0.0, 1.2), rng.uniform(0.05, 0.6), N)
        for k in range(1, K):
            B[:, k] = rng.normal(0, rng.uniform(0.1, 0.7), N)
    fvar = np.sort(rng.uniform(0.1, 1.0, K))[::-1] if K > 0 else np.zeros(0)
    if K > 0:
        fvar[0] *= rng.uniform(1, 4)
    C = B @ np.diag(fvar) @ B.T
    if rng.random() < 0.3:                                    # block (sector) structure
        nb = int(rng.integers(2, 6)); lab = rng.integers(0, nb, N)
        C = C + rng.uniform(0.1, 0.6) * (lab[:, None] == lab[None, :])
    idio = np.exp(rng.normal(0, rng.uniform(0.05, 0.5), N))
    C = C + np.diag(idio)
    d = np.sqrt(np.diag(C)); C = C / np.outer(d, d)            # correlation matrix
    # volatilities: log-uniform spread with given dispersion, random assignment
    sd = base * np.exp(rng.uniform(0, np.log(disp), N))
    sd[rng.integers(N)] = base; sd[rng.integers(N)] = base * disp
    return C * np.outer(sd, sd), dict(K=K)


def part1(rng, M=200):
    rows = []
    for nm, mk in [("factor", lambda N: jrs.make_sigma_factor(N, 3, rng)),
                   ("hetero", lambda N: jrs.make_sigma_hetero(N, rng)),
                   ("equicorr", lambda N: jrs.make_sigma_equicorr(N, rng))]:
        for N in [30, 60]:
            Sigma = mk(N)
            for T in [120, 250, 500]:
                th = theory(Sigma, T); sm = simulate(Sigma, T, M, rng)
                rows.append(dict(dgp=nm, N=N, T=T, **th, **sm))
                print(nm, N, T, round(th["dD_approx"], 3), round(sm["dD_sim"], 3),
                      round(th["dF_approx"], 3), round(sm["dF_sim"], 3), round(th["Lambda"], 3), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "R1_sign_theory_check_dgps.csv"), index=False)
    return df


def part2(rng, n_sigma=600, M=40):
    global GRID
    GRID = np.linspace(0, 1, 101)
    rows = []
    Ns = [10, 30, 60, 120]; Ts = [60, 120, 250, 500, 1000]
    for j in range(n_sigma):
        N = int(rng.choice(Ns))
        T = int(rng.choice([t for t in Ts if t > N]))   # N < T: sample matrix invertible
        Sigma, meta = random_sigma(rng, N)
        th = theory(Sigma, T)
        sm = simulate(Sigma, T, M, rng)
        rows.append(dict(id=j, N=N, T=T, NT=N / T, K=meta["K"], **th, **sm))
        if j % 50 == 0:
            print(j, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "R1_sign_random_design_raw.csv"), index=False)
    return df


def summarise(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    d = df.copy()
    d["sign_pos"] = (d.diff_sim > 0).astype(int)        # mean delta_D above mean delta_F
    out = {}
    out["n_structures"] = len(d)
    out["share_structures_dD_gt_dF"] = d.sign_pos.mean()
    # 1) Lambda rule
    pred = (d.Lambda < 1).astype(int)
    out["Lambda_rule_accuracy"] = (pred == d.sign_pos).mean()
    out["Lambda_rule_auc"] = roc_auc_score(d.sign_pos, -d.log_Lambda)
    # 2) approximation accuracy
    out["corr_dD_approx_sim"] = np.corrcoef(d.dD_approx, d.dD_sim)[0, 1]
    out["corr_dF_approx_sim"] = np.corrcoef(d.dF_approx, d.dF_sim)[0, 1]
    out["LambdaT_rule_accuracy"] = ((d.Lambda_T < 1).astype(int) == d.sign_pos).mean()
    out["corr_dD_approx_fs_sim"] = np.corrcoef(d.dD_approx_fs, d.dD_sim)[0, 1]
    out["approx_fs_sign_accuracy"] = ((d.dD_approx_fs > d.dF_approx).astype(int) == d.sign_pos).mean()
    out["approx_sign_accuracy"] = ((d.dD_approx > d.dF_approx).astype(int) == d.sign_pos).mean()
    # 3) single-feature AUCs (sign of association chosen in-sample, so these are upper bounds)
    for f in ["vol_disp", "vol_cv", "mean_corr", "top_eig_share", "NT"]:
        x = np.log(d[f]) if f in ["vol_disp"] else d[f]
        auc = roc_auc_score(d.sign_pos, x)
        out[f"auc_{f}"] = max(auc, 1 - auc)
    # 4) cross-validated logistic models
    feats = {"vol_disp_only": ["log_vol_disp"],
             "Lambda_only": ["log_Lambda"],
             "structure_no_Lambda": ["log_vol_disp", "mean_corr", "top_eig_share", "NT"],
             "structure_plus_Lambda": ["log_vol_disp", "mean_corr", "top_eig_share", "NT", "log_Lambda"]}
    d["log_vol_disp"] = np.log(d.vol_disp)
    for nm, cols in feats.items():
        p = cross_val_predict(LogisticRegression(max_iter=2000), d[cols].values, d.sign_pos.values,
                              cv=10, method="predict_proba")[:, 1]
        out[f"cv_auc_{nm}"] = roc_auc_score(d.sign_pos, p)
        out[f"cv_acc_{nm}"] = ((p > 0.5).astype(int) == d.sign_pos).mean()
    # 5) share by dispersion bucket
    d["disp_bucket"] = pd.cut(d.vol_disp, [1, 2, 4, 8, 16, 31])
    tab = d.groupby("disp_bucket", observed=True).agg(
        n=("sign_pos", "size"), share_dD_gt_dF=("sign_pos", "mean"),
        share_Lambda_lt_1=("Lambda", lambda s: (s < 1).mean()),
        mean_dD=("dD_sim", "mean"), mean_dF=("dF_sim", "mean"), mean_G=("G_pct", "median"))
    # 6) within low-dispersion structures, what governs the sign?
    low = d[d.vol_disp < 2]
    out["lowdisp_LambdaT_accuracy"] = ((low.Lambda_T < 1).astype(int) == low.sign_pos).mean() if len(low) else np.nan
    out["lowdisp_n"] = len(low)
    out["lowdisp_share_dD_gt_dF"] = low.sign_pos.mean()
    out["lowdisp_Lambda_accuracy"] = ((low.Lambda < 1).astype(int) == low.sign_pos).mean()
    s = pd.Series(out)
    s.to_csv(os.path.join(OUT, "R1_sign_random_design_summary.csv"))
    tab.to_csv(os.path.join(OUT, "R1_sign_random_design_by_dispersion.csv"))
    # finer bins, as reported in Table 4, Panel B of the paper
    fine = d.groupby(pd.cut(d.vol_disp, [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 31]), observed=True).agg(
        n=("sign_pos", "size"), share_dD_gt_dF=("sign_pos", "mean"),
        share_Lambda_lt_1=("Lambda", lambda s: (s < 1).mean()))
    fine.to_csv(os.path.join(OUT, "R1_sign_random_design_by_dispersion_table4B.csv"))
    print(fine.round(2).T.to_string())
    print(s.round(3).to_string()); print(tab.round(3).to_string())
    return s, tab


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(2026)
    parts = sys.argv[1:] or ["1", "2"]
    if "1" in parts:
        print(part1(rng)[["dgp", "N", "T", "dD_approx", "dD_sim", "dF_approx", "dF_sim",
                          "Lambda", "share_dD_gt_dF"]].round(3).to_string())
    if "2" in parts:
        df = part2(rng)
        summarise(df)
    if "s" in parts:
        summarise(pd.read_csv(os.path.join(OUT, "R1_sign_random_design_raw.csv")))
