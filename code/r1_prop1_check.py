"""Revision R1, reviewer comment 1: verification of the corrected Proposition 1.

Checks, on simulated covariance matrices from all three DGPs of Section 4:
  (a) the corrected weight derivative  dw/dd = -S^{-1} D w + c(d) w (w'Dw)
      against central finite differences of the GMV map;
  (b) the derivative as printed in the original submission (without c(d)),
      to document the size of the typesetting error;
  (c) that the corrected derivative satisfies the budget identity 1'dw/dd = 0;
  (d) the equivalent forms of the decision first-order condition:
         dV/dd = -2 w' Sigma M D w = 2 w' (Shat - Sigma) M D w,
      with M = Shat^{-1} - u u'/c, against finite differences of V(d);
  (e) the numerical FOC check of Section 4.3 (dV/dd at delta_D and delta_F,
      normalised by the range of V), recomputed with the corrected derivative.

Writes results_revision/R1_prop1_*.csv
"""
import numpy as np
import pandas as pd
from _suite import jrs, OUT
import os

rng = np.random.default_rng(20261112)


def pieces(S, F, d):
    D = F - S
    Sh = (1 - d) * S + d * F
    Si = np.linalg.inv(Sh)
    one = np.ones(len(S))
    u = Si @ one
    c = one @ u
    w = u / c
    M = Si - np.outer(u, u) / c
    return D, Sh, Si, u, c, w, M


def dw_corrected(S, F, d):
    D, Sh, Si, u, c, w, M = pieces(S, F, d)
    return -Si @ D @ w + c * w * (w @ D @ w)


def dw_printed(S, F, d):
    D, Sh, Si, u, c, w, M = pieces(S, F, d)
    return -Si @ D @ w + w * (w @ D @ w)


def dw_compact(S, F, d):
    D, Sh, Si, u, c, w, M = pieces(S, F, d)
    return -M @ D @ w


def w_of(S, F, d):
    return pieces(S, F, d)[5]


def main():
    specs = [("factor", lambda N: jrs.make_sigma_factor(N, 3, rng)),
             ("hetero", lambda N: jrs.make_sigma_hetero(N, rng)),
             ("equicorr", lambda N: jrs.make_sigma_equicorr(N, rng))]
    rows = []
    eps = 1e-6
    for nm, mk in specs:
        for N in [10, 30, 60]:
            for T in [120, 250]:
                Sigma = mk(N)
                for m in range(20):
                    X = jrs.sample_returns(Sigma, T, "gauss", 5, rng)
                    X = X - X.mean(0)
                    S = X.T @ X / T
                    F = jrs.target_identity(S)
                    d = rng.uniform(0.02, 0.98)
                    fd = (w_of(S, F, d + eps) - w_of(S, F, d - eps)) / (2 * eps)
                    a = dw_corrected(S, F, d)
                    b = dw_printed(S, F, d)
                    k = dw_compact(S, F, d)
                    D, Sh, Si, u, c, w, M = pieces(S, F, d)
                    V = lambda dd: (lambda ww: ww @ Sigma @ ww)(w_of(S, F, dd))
                    dV_fd = (V(d + eps) - V(d - eps)) / (2 * eps)
                    dV_1 = 2 * w @ Sigma @ a
                    dV_2 = 2 * w @ (Sh - Sigma) @ M @ D @ w
                    nfd = np.linalg.norm(fd)
                    rows.append(dict(
                        dgp=nm, N=N, T=T, delta=d, c_delta=c,
                        relerr_corrected=np.linalg.norm(a - fd) / nfd,
                        relerr_compact=np.linalg.norm(k - fd) / nfd,
                        relerr_printed=np.linalg.norm(b - fd) / nfd,
                        budget_corrected=abs(a.sum()) / nfd,
                        budget_printed=abs(b.sum()) / nfd,
                        relerr_dV_form1=abs(dV_1 - dV_fd) / max(abs(dV_fd), 1e-300),
                        relerr_dV_form2=abs(dV_2 - dV_fd) / max(abs(dV_fd), 1e-300),
                    ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "R1_prop1_derivative_check_raw.csv"), index=False)
    cols = ["relerr_corrected", "relerr_compact", "relerr_printed", "budget_corrected",
            "budget_printed", "relerr_dV_form1", "relerr_dV_form2"]
    summ = df.groupby("dgp")[cols].agg(["median", "max"])
    summ.to_csv(os.path.join(OUT, "R1_prop1_derivative_check_summary.csv"))
    print(summ.T.to_string())
    print("c(delta) range:", df.c_delta.min(), df.c_delta.max())

    # (e) FOC check at both optima, same design as Section 4.3, corrected derivative
    grid = np.linspace(0, 1, 201)
    frows = []
    for nm, mk in specs:
        for N in [30, 60]:
            Sigma = mk(N)
            for T in [120, 250]:
                for m in range(25):
                    X = jrs.sample_returns(Sigma, T, "gauss", 5, rng)
                    X = X - X.mean(0)
                    S = X.T @ X / T
                    F = jrs.target_identity(S)
                    Vg = np.array([(lambda ww: ww @ Sigma @ ww)(w_of(S, F, d)) for d in grid])
                    Fr = np.array([np.linalg.norm((1 - d) * S + d * F - Sigma, "fro") ** 2
                                   for d in grid])
                    iD, iF = int(np.argmin(Vg)), int(np.argmin(Fr))
                    sV = Vg.max() - Vg.min() + 1e-300

                    def dV(d):
                        return 2 * w_of(S, F, d) @ Sigma @ dw_corrected(S, F, d)
                    # interior optima only for the 'zero at optimum' check
                    frows.append(dict(dgp=nm, N=N, T=T, delta_D=grid[iD], delta_F=grid[iF],
                                      interior_D=0 < iD < len(grid) - 1,
                                      dV_at_D=dV(grid[iD]) / sV, dV_at_F=dV(grid[iF]) / sV))
    fdf = pd.DataFrame(frows)
    fdf.to_csv(os.path.join(OUT, "R1_prop1_FOC_check_raw.csv"), index=False)
    agg = fdf.groupby("dgp").agg(n=("dV_at_D", "size"),
                                 share_interior_D=("interior_D", "mean"),
                                 dV_at_D_mean=("dV_at_D", "mean"),
                                 dV_at_D_interior_absmax=("dV_at_D",
                                                          lambda s: s[fdf.loc[s.index, "interior_D"]].abs().max()
                                                          if fdf.loc[s.index, "interior_D"].any() else np.nan),
                                 dV_at_F_mean=("dV_at_F", "mean"),
                                 dV_at_F_median=("dV_at_F", "median"),
                                 share_dV_at_F_negative=("dV_at_F", lambda s: (s < 0).mean()))
    agg.to_csv(os.path.join(OUT, "R1_prop1_FOC_check_summary.csv"))
    print(agg.round(4).to_string())


if __name__ == "__main__":
    main()
