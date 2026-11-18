#!/usr/bin/env python3
"""
dfs_pipeline.py
Main pipeline for:
  "Decision-Focused Covariance Shrinkage for Minimum-Variance Portfolios"

All outputs are written to --out. Prices are read from <out>/prices_<universe>.csv
if present (copy data/prices_*.csv and data/volume_*.csv there), otherwise
downloaded with yfinance.

  python dfs_pipeline.py --out ../results_original --part all
  python dfs_pipeline.py --out ../results_original --part core,inference,attribution
  add --quick for a fast smoke test.

Parts
-----
 data          download + cache prices (stocks, etfs)
 core          all baselines + NONLINEAR SHRINKAGE + DCC-NL approx + DFS variants
 inference     Ledoit-Wolf(2011) HAC variance tests, LW(2008) Sharpe tests,
               stationary-bootstrap studentised p-values, Romano-Wolf stepdown
 attribution   delta-sweep oracle, vol-tilt vs correlation decomposition,
               weight diagnostics, break-even transaction cost
 hyper         sensitivity to (L, h), drift-aware turnover
 subperiod     GFC / post-crisis / COVID / rates-shock split with tests
 placebo       randomised-training-target placebo (Henide-style null)
 seeds         multi-seed dispersion of every DFS variant
 survivor      rotating point-in-time universe (survivorship-bias mitigation)
 sim           simulation extensions: SEs, reversal DGP, Prop.1 FOC check,
               DFS-NN learning curve
 audit         look-ahead / alignment assertions
 report        builds REVISION_NOTES.md with every number you need for the .tex
"""

import os, sys, json, math, time, argparse, warnings, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf, OAS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

# =============================================================================
# CONFIG  (mirrors dfs_shrinkage.py so baseline numbers reconcile)
# =============================================================================
CFG = dict(
    seed=42,
    start="2004-12-01", end="2024-12-31",
    L=250, h=21,
    costs=[0.0010, 0.0025, 0.0050], primary_cost=0.0025,
    psi=0.4, psi_grid=[0.2, 0.4, 1.0],
    pca_k=3, ewma_lambda=0.97,
    n_stocks=60,
    dfs=dict(min_train=24, retrain_every=12, steps=300, lr=0.05, lo_iters=80),
    boot=dict(B=1000, block=10, seed=7),
    seeds_list=[42, 1, 2, 3, 4, 5, 6, 7],
    stocks=["AAPL","MSFT","AMZN","GOOGL","JPM","JNJ","XOM","PG","KO","PEP","WMT","HD","CVX","MRK","PFE",
            "BAC","WFC","C","INTC","CSCO","ORCL","IBM","T","VZ","DIS","MCD","NKE","UNH","CAT","BA","MMM",
            "GE","HON","UPS","LMT","GS","MS","AXP","BLK","ABT","AMGN","GILD","LLY","BMY","MDT","TXN","QCOM",
            "ADBE","NVDA","AMAT","COST","LOW","TGT","SBUX","MO","CL","KMB","GD","RTX","DE","EMR","ITW","UNP",
            "CSX","NSC","FDX","SO","DUK","NEE","D","AEP","ADP","TMO","DHR","SYK","BDX","ISRG","SCHW","USB",
            "PNC","MET","PRU","AFL","ALL","TRV","MMC","AON","SPG","PLD","AMT","EBAY"],
    etfs=["SPY","QQQ","IWM","EFA","EEM","TLT","IEF","LQD","HYG","GLD","DBC","VNQ"],
)

QUICK = False


def log(msg, path=None):
    print(msg, flush=True)
    if path:
        with open(path, "a") as fh:
            fh.write(msg + "\n")


def save(df, out, name, index=True, tex=True):
    df.to_csv(os.path.join(out, name + ".csv"), index=index)
    if tex:
        try:
            with open(os.path.join(out, name + ".tex"), "w") as fh:
                fh.write(df.to_latex(index=index, float_format="%.4f"))
        except Exception:
            pass


# =============================================================================
# 1. ESTIMATORS
# =============================================================================
def gmv(Sig):
    N = Sig.shape[0]; one = np.ones(N)
    try:
        u = np.linalg.solve(Sig, one)
        if not np.all(np.isfinite(u)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        u = np.linalg.pinv(Sig) @ one
    return u / u.sum()


def proj_simplex(v):
    n = v.shape[-1]; u = np.sort(v)[::-1]; css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def gmv_long_only(Sig, iters=400):
    N = Sig.shape[0]; w = np.ones(N) / N
    eta = 1.0 / max(np.linalg.eigvalsh(Sig)[-1], 1e-12)
    for _ in range(iters):
        w = proj_simplex(w - eta * (Sig @ w))
    return w


def target_identity(S):
    N = S.shape[0]; return (np.trace(S) / N) * np.eye(N)


def target_cc(S):
    N = S.shape[0]; var = np.diag(S); sd = np.sqrt(var)
    corr = S / np.outer(sd, sd)
    rbar = (corr.sum() - N) / (N * (N - 1))
    F = rbar * np.outer(sd, sd); np.fill_diagonal(F, var); return F


def target_pca(S, k):
    vals, vecs = np.linalg.eigh(S); idx = np.argsort(vals)[::-1][:k]
    B = vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0)); Sf = B @ B.T
    D = np.maximum(np.diag(S) - np.diag(Sf), 1e-12)
    return Sf + np.diag(D)


def lw_identity(X):
    return float(LedoitWolf(assume_centered=True).fit(X).shrinkage_)


def lw_cc(X):
    T, N = X.shape; S = X.T @ X / T
    var = np.diag(S); sd = np.sqrt(var)
    corr = S / np.outer(sd, sd); rbar = (corr.sum() - N) / (N * (N - 1))
    F = rbar * np.outer(sd, sd); np.fill_diagonal(F, var)
    Y = X ** 2
    phi_mat = Y.T @ Y / T - S ** 2; phi = phi_mat.sum()
    term1 = (X ** 3).T @ X / T
    theta = term1 - np.outer(var, np.ones(N)) * S
    np.fill_diagonal(theta, 0.0)
    rho = np.trace(phi_mat) + rbar * (np.outer(1.0 / sd, sd) * theta).sum()
    gamma = np.linalg.norm(S - F, "fro") ** 2
    kappa = (phi - rho) / max(gamma, 1e-18)
    return float(np.clip(kappa / T, 0.0, 1.0))


def oas_cov(X):
    return OAS(assume_centered=True).fit(X).covariance_


def ewma_cov(X, lam):
    T = len(X); w = lam ** np.arange(T - 1, -1, -1); w /= w.sum()
    return (X * w[:, None]).T @ X


def mp_clip_cov(S, T):
    N = S.shape[0]; sd = np.sqrt(np.diag(S)); C = S / np.outer(sd, sd)
    vals, vecs = np.linalg.eigh(C); q = N / T
    sigma2 = max(1.0 - vals[-1] / N, 1e-6); lam_plus = sigma2 * (1 + np.sqrt(q)) ** 2
    mask = vals < lam_plus
    if mask.any():
        vals[mask] = vals[mask].mean()
    Cc = vecs @ np.diag(vals) @ vecs.T; d = np.sqrt(np.diag(Cc)); Cc = Cc / np.outer(d, d)
    return Cc * np.outer(sd, sd)


def shrink(S, F, d):
    return (1.0 - d) * S + d * F


# ---- NEW: analytical nonlinear shrinkage, Ledoit & Wolf (2020, AoS) --------
def nonlinear_shrinkage(X, demeaned=True):
    """Port of analytical_shrinkage.m. X is T x p (already demeaned)."""
    T, p = X.shape
    n = T - (0 if demeaned else 1)
    n = max(n, 2)
    S = X.T @ X / n
    lam, u = np.linalg.eigh(S)
    order = np.argsort(lam); lam = lam[order]; u = u[:, order]
    lam = lam[max(0, p - n):p]
    lam = np.maximum(lam, 1e-14)
    m = len(lam)
    L = np.tile(lam[:, None], (1, m))
    hh = n ** (-1.0 / 3.0)
    H = hh * L.T
    x = (L - L.T) / H
    ftilde = (3.0 / 4.0 / np.sqrt(5.0)) * np.mean(np.maximum(1 - x ** 2 / 5.0, 0) / H, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Hf = (-3.0 / 10.0 / np.pi) * x + (3.0 / 4.0 / np.sqrt(5.0) / np.pi) * \
             (1 - x ** 2 / 5.0) * np.log(np.abs((np.sqrt(5.0) - x) / (np.sqrt(5.0) + x)))
    bad = np.abs(np.abs(x) - np.sqrt(5.0)) < 1e-12
    Hf[bad] = (-3.0 / 10.0 / np.pi) * x[bad]
    Hf = np.nan_to_num(Hf)
    Hftilde = np.mean(Hf / H, axis=1)
    if p <= n:
        c = p / n
        dtilde = lam / ((np.pi * c * lam * ftilde) ** 2 +
                        (1 - c - np.pi * c * lam * Hftilde) ** 2)
    else:
        Hftilde0 = (1.0 / np.pi) * (3.0 / 10.0 / hh ** 2 + 3.0 / 4.0 / np.sqrt(5.0) / hh *
                    (1 - 1.0 / 5.0 / hh ** 2) *
                    np.log((1 + np.sqrt(5.0) * hh) / (1 - np.sqrt(5.0) * hh))) * np.mean(1.0 / lam)
        dtilde0 = 1.0 / (np.pi * (p - n) / n * Hftilde0)
        dtilde1 = lam / (np.pi ** 2 * lam ** 2 * (ftilde ** 2 + Hftilde ** 2))
        dtilde = np.concatenate([dtilde0 * np.ones(p - n), dtilde1])
    Sig = u @ np.diag(dtilde) @ u.T
    return 0.5 * (Sig + Sig.T)


# ---- NEW: cheap DCC-NL stand-in: EWMA de-vol + nonlinear-shrunk correlation
def dccnl_cov(X, lam=0.94):
    T, p = X.shape
    w = lam ** np.arange(T - 1, -1, -1)
    w = w / w.sum()
    sig = np.sqrt(np.maximum((X ** 2 * w[:, None]).sum(0), 1e-16))
    # rolling conditional vol path for de-volatilisation
    v = np.empty_like(X)
    s2 = (X[:60] ** 2).mean(0)
    for t in range(T):
        v[t] = np.sqrt(np.maximum(s2, 1e-16))
        s2 = lam * s2 + (1 - lam) * X[t] ** 2
    Z = X / v
    Cz = nonlinear_shrinkage(Z - Z.mean(0))
    d = np.sqrt(np.maximum(np.diag(Cz), 1e-16))
    Cz = Cz / np.outer(d, d)
    return Cz * np.outer(sig, sig)


# =============================================================================
# 2. TORCH / DFS
# =============================================================================
def proj_simplex_t(v):
    n = v.shape[-1]; u, _ = torch.sort(v, dim=-1, descending=True); css = torch.cumsum(u, -1)
    k = torch.arange(1, n + 1, dtype=v.dtype); cond = u * k > (css - 1)
    rho = cond.sum(-1, keepdim=True); theta = (css.gather(-1, rho - 1) - 1) / rho
    return torch.clamp(v - theta, min=0.0)


def gmv_t(Sig):
    B, N, _ = Sig.shape; one = torch.ones(B, N, 1)
    ridge = 1e-6 * torch.diagonal(Sig, dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(N)
    u = torch.linalg.solve(Sig + ridge, one).squeeze(-1)
    return u / u.sum(1, keepdim=True)


def gmv_lo_t(Sig, iters):
    B, N, _ = Sig.shape; w = torch.full((B, N), 1.0 / N)
    eta = 1.0 / torch.linalg.eigvalsh(Sig)[:, -1:].clamp(min=1e-12)
    for _ in range(iters):
        w = proj_simplex_t(w - eta * torch.einsum("bij,bj->bi", Sig, w))
    return w


class DFSModel(nn.Module):
    def __init__(self, variant, feat_dim=5):
        super().__init__(); self.variant = variant
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.mix = nn.Parameter(torch.zeros(3))
        self.mlp = nn.Sequential(nn.Linear(feat_dim, 16), nn.Tanh(), nn.Linear(16, 1))
        self.register_buffer("fmu", torch.zeros(feat_dim)); self.register_buffer("fsd", torch.ones(feat_dim))

    def delta(self, feat):
        if self.variant == "nn":
            return torch.sigmoid(self.mlp((feat - self.fmu) / self.fsd)).squeeze(-1)
        return torch.sigmoid(self.theta).expand(feat.shape[0])

    def cov(self, S, FI, FCC, FPCA, feat):
        d = self.delta(feat)[:, None, None]
        if self.variant == "cc":
            F = FCC
        elif self.variant == "mix":
            pi = torch.softmax(self.mix, 0); F = pi[0] * FI + pi[1] * FCC + pi[2] * FPCA
        else:
            F = FI
        return (1 - d) * S + d * F


def train_dfs(P, idx, variant, cost, psi, cfg, long_only=False, seed=None, shuffle_H=False):
    if seed is not None:
        torch.manual_seed(seed)
    S, FI, FCC, FPCA = (torch.as_tensor(P[k][idx]) for k in ["S", "FI", "FCC", "FPCA"])
    Hn = P["H"][idx]
    if shuffle_H:                      # PLACEBO: break the S_r <-> H_r link
        rs = np.random.default_rng(1234 + len(idx))
        Hn = Hn[rs.permutation(len(idx))]
    H = torch.as_tensor(Hn); feat = torch.as_tensor(P["feat"][idx]); h = cfg["h"]
    model = DFSModel(variant); model.fmu = feat.mean(0); model.fsd = feat.std(0).clamp(min=1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["dfs"]["lr"])
    steps = cfg["dfs"]["steps"] // (2 if long_only else 1)
    for _ in range(steps):
        Sig = model.cov(S, FI, FCC, FPCA, feat)
        w = gmv_lo_t(Sig, cfg["dfs"]["lo_iters"]) if long_only else gmv_t(Sig)
        port = torch.einsum("rhn,rn->rh", H, w)
        loss = port.pow(2).mean() * 252 + psi * cost * (w[1:] - w[:-1]).abs().sum(1).mean() * (252.0 / h)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def apply_dfs(model, P, r, cfg, long_only=False):
    S, FI, FCC, FPCA = (torch.as_tensor(P[k][r:r + 1]) for k in ["S", "FI", "FCC", "FPCA"])
    feat = torch.as_tensor(P["feat"][r:r + 1])
    Sig = model.cov(S, FI, FCC, FPCA, feat)
    w = gmv_lo_t(Sig, 400) if long_only else gmv_t(Sig)
    return w[0].numpy(), float(model.delta(feat)[0])


def walk_forward_dfs(P, variant, cost, psi, cfg, long_only=False, seed=None, shuffle_H=False, tag=""):
    R, N = P["R"], P["N"]; mt, re = cfg["dfs"]["min_train"], cfg["dfs"]["retrain_every"]
    W = np.zeros((R, N)); dl = np.full(R, np.nan); model = None; t0 = time.time()
    for r in range(mt, R):
        if (r - mt) % re == 0:
            model = train_dfs(P, np.arange(0, r), variant, cost, psi, cfg, long_only, seed, shuffle_H)
        W[r], dl[r] = apply_dfs(model, P, r, cfg, long_only)
    if tag:
        print(f"    DFS {tag:26s} ({time.time()-t0:.0f}s)")
    return W, dl


# =============================================================================
# 3. DATA / PRECOMPUTE / EVALUATION
# =============================================================================
def load_prices(tickers, cfg, out, tag):
    cache = os.path.join(out, f"prices_{tag}.csv")
    vcache = os.path.join(out, f"volume_{tag}.csv")
    if os.path.exists(cache):
        px = pd.read_csv(cache, index_col=0, parse_dates=True)
        vol = pd.read_csv(vcache, index_col=0, parse_dates=True) if os.path.exists(vcache) else None
        return px, vol
    import yfinance as yf
    raw = yf.download(tickers, start=cfg["start"], end=cfg["end"],
                      auto_adjust=True, progress=False, threads=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    try:
        vol = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]]
    except Exception:
        vol = None
    px.index = pd.to_datetime(px.index)
    px.to_csv(cache)
    if vol is not None:
        vol.index = pd.to_datetime(vol.index); vol.to_csv(vcache)
    return px, vol


def clean_universe(px, cfg, n_keep=None):
    tickers = list(px.columns)
    first_valid = px.apply(lambda s: s.first_valid_index())
    cutoff = pd.Timestamp(cfg["start"]) + pd.Timedelta(days=60)
    good = [t for t in tickers if first_valid[t] is not None and first_valid[t] <= cutoff
            and px[t].isna().mean() < 0.02]
    if n_keep:
        good = good[:n_keep]
    p = px[good].ffill(limit=5).dropna()
    R = p.pct_change().dropna().clip(-0.5, 0.5)
    return R


def precompute(R, cfg, cols_fn=None, n_sub=None):
    L, h, k = cfg["L"], cfg["h"], cfg["pca_k"]
    A = R.values; D, Nfull = A.shape
    starts = list(range(L, D - h + 1, h))
    P = dict(S=[], FI=[], FCC=[], FPCA=[], H=[], X=[], feat=[], cols=[], date=[], hdates=[])
    for t in starts:
        cols = np.arange(Nfull) if cols_fn is None else cols_fn(t)
        X = A[t - L:t][:, cols]; X = X - X.mean(0); N = len(cols)
        S = X.T @ X / L
        sd = np.sqrt(np.diag(S)); corr = S / np.outer(sd, sd)
        vals = np.linalg.eigvalsh(corr)
        feat = [(corr.sum() - N) / (N * (N - 1)), vals[-1] / N,
                np.log(vals[-1] / max(vals[0], 1e-9)), sd.mean() * np.sqrt(252),
                (vals < (1 + np.sqrt(N / L)) ** 2).mean()]
        P["S"].append(S); P["FI"].append(target_identity(S)); P["FCC"].append(target_cc(S))
        P["FPCA"].append(target_pca(S, k)); P["H"].append(A[t:t + h][:, cols]); P["X"].append(X)
        P["feat"].append(feat); P["cols"].append(cols); P["date"].append(R.index[t])
        P["hdates"].append(R.index[t:t + h])
    for key in ["S", "FI", "FCC", "FPCA", "H", "feat", "X", "cols"]:
        P[key] = np.array(P[key])
    P["N"] = P["S"].shape[1]; P["Nfull"] = Nfull; P["R"] = len(starts)
    P["dates"] = pd.DatetimeIndex(P["date"])
    P["oos_dates_all"] = np.concatenate([d.values for d in P["hdates"]])
    return P


def full_weights(W, P, r):
    wf = np.zeros(P["Nfull"]); wf[P["cols"][r]] = W[r]; return wf


def evaluate(W, P, r0, cfg, drift=False):
    """Returns metric dict, per-rebalance realised variance, daily OOS returns, dates."""
    H = P["H"]; h = cfg["h"]
    rets, var_r, to_r, dates = [], [], [], []
    prev = None
    for r in range(r0, P["R"]):
        pr = H[r] @ W[r]
        rets.append(pr); var_r.append(pr.var()); dates.append(P["hdates"][r].values)
        wf = full_weights(W, P, r)
        if prev is None:
            to = 0.0
        else:
            if drift:
                g = np.ones(P["Nfull"])
                g[P["cols"][r - 1]] = np.prod(1 + H[r - 1], axis=0)
                drifted = prev * g
                s = drifted.sum()
                drifted = drifted / s if abs(s) > 1e-12 else prev
                to = np.abs(wf - drifted).sum()
            else:
                to = np.abs(wf - prev).sum()
        to_r.append(to); prev = wf
    rets = np.concatenate(rets); dates = np.concatenate(dates)
    to_r = np.array(to_r); var_r = np.array(var_r)
    ann_vol = rets.std() * np.sqrt(252); ann_ret = rets.mean() * 252
    cum = np.cumprod(1 + rets); mdd = (cum / np.maximum.accumulate(cum) - 1).min()
    out = dict(ann_vol=100 * ann_vol, ann_ret=100 * ann_ret, sharpe=ann_ret / ann_vol,
               mdd=100 * mdd, turnover_ann=to_r.mean() * 252 / h,
               realized_var_ann=100 ** 2 * rets.var() * 252)
    nets = {}
    for c in cfg["costs"]:
        net = rets.copy().reshape(-1, h); net[:, 0] -= c * to_r; net = net.ravel()
        out[f"sharpe_net_{int(c*1e4)}bp"] = net.mean() * 252 / (net.std() * np.sqrt(252))
        out[f"ret_net_{int(c*1e4)}bp"] = 100 * net.mean() * 252
        nets[c] = net
    return out, var_r, rets, dates, to_r, nets


def net_returns(rets, to_r, c, h):
    net = rets.copy().reshape(-1, h); net[:, 0] -= c * to_r; return net.ravel()


# =============================================================================
# 4. HAC / BOOTSTRAP INFERENCE  (Ledoit-Wolf 2008/2011 style)
# =============================================================================
def andrews_bartlett_lrv(z):
    """Long-run variance of a mean-zero series, Bartlett kernel, Andrews(1991) bw."""
    z = np.asarray(z, float); n = len(z); z = z - z.mean()
    if n < 10:
        return z.var()
    denom = (z[:-1] @ z[:-1])
    rho = (z[1:] @ z[:-1]) / denom if denom > 0 else 0.0
    rho = np.clip(rho, -0.97, 0.97)
    alpha = 4 * rho ** 2 / (1 - rho ** 2) ** 2
    S_T = max(1, int(np.ceil(1.1447 * (alpha * n) ** (1 / 3))))
    S_T = min(S_T, n - 1)
    g = z @ z / n
    for k in range(1, S_T + 1):
        g += 2 * (1 - k / (S_T + 1)) * (z[k:] @ z[:-k]) / n
    return max(g, 1e-18)


def _moments(a, b):
    return np.column_stack([a, b, a ** 2, b ** 2])


def logvar_stat(a, b):
    mA, mB, gA, gB = a.mean(), b.mean(), (a ** 2).mean(), (b ** 2).mean()
    vA, vB = max(gA - mA ** 2, 1e-18), max(gB - mB ** 2, 1e-18)
    theta = np.log(vA) - np.log(vB)
    grad = np.array([-2 * mA / vA, 2 * mB / vB, 1.0 / vA, -1.0 / vB])
    Y = _moments(a, b); z = (Y - Y.mean(0)) @ grad
    se = np.sqrt(andrews_bartlett_lrv(z) / len(a))
    return theta, se


def sharpe_stat(a, b):
    mA, mB, gA, gB = a.mean(), b.mean(), (a ** 2).mean(), (b ** 2).mean()
    sA, sB = np.sqrt(max(gA - mA ** 2, 1e-18)), np.sqrt(max(gB - mB ** 2, 1e-18))
    theta = mA / sA - mB / sB
    grad = np.array([gA / sA ** 3, -gB / sB ** 3, -mA / (2 * sA ** 3), mB / (2 * sB ** 3)])
    Y = _moments(a, b); z = (Y - Y.mean(0)) @ grad
    se = np.sqrt(andrews_bartlett_lrv(z) / len(a))
    return theta, se


def stationary_bootstrap_idx(n, b, B, rng):
    p = 1.0 / b
    ar = np.arange(n)[None, :]
    restart = rng.random((B, n)) < p
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(B, n))
    gstart = np.maximum.accumulate(np.where(restart, ar, 0), axis=1)
    ff = np.take_along_axis(starts, gstart, axis=1)
    return ((ff + (ar - gstart)) % n).astype(np.int32)


def boot_test(a, b, stat_fn, cfg, idx=None, rng=None):
    """Studentised stationary-bootstrap two-sided p-value (Ledoit-Wolf 2008/2011)."""
    n = len(a)
    rng = rng or np.random.default_rng(cfg["boot"]["seed"])
    idx = stationary_bootstrap_idx(n, cfg["boot"]["block"], cfg["boot"]["B"], rng) if idx is None else idx
    th, se = stat_fn(a, b)
    t = th / se if se > 0 else 0.0
    cnt = 0; B = idx.shape[0]; tstars = np.empty(B)
    for j in range(B):
        ii = idx[j]
        thj, sej = stat_fn(a[ii], b[ii])
        tj = (thj - th) / sej if sej > 0 else 0.0
        tstars[j] = tj
        cnt += abs(tj) >= abs(t)
    return dict(stat=th, se=se, t=t, p_boot=(cnt + 1) / (B + 1),
                p_hac=2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))), tstars


def romano_wolf(tstats, tstar_mat):
    """Romano-Wolf stepdown adjusted p-values. tstar_mat: B x M centred bootstrap t's."""
    M = len(tstats); B = tstar_mat.shape[0]
    order = np.argsort(-np.abs(tstats)); padj = np.ones(M)
    active = list(order)
    prev = 0.0
    for step in range(M):
        m = active[0]
        mx = np.max(np.abs(tstar_mat[:, active]), axis=1)
        p = (np.sum(mx >= abs(tstats[m])) + 1) / (B + 1)
        p = max(p, prev); prev = p
        padj[m] = p
        active = active[1:]
        if not active:
            break
    return padj


# =============================================================================
# 5. CORE RUN
# =============================================================================
def build_P(universe, cfg, out, L=None, h=None, rotating=False):
    c = json.loads(json.dumps(cfg)); c["dfs"] = cfg["dfs"]
    if L: c["L"] = L
    if h: c["h"] = h
    tick = cfg["stocks"] if universe == "stocks" else cfg["etfs"]
    px, vol = load_prices(tick, cfg, out, universe)
    if rotating:
        # Rotating point-in-time universe (Section 6.5). Constituents at rebalance t are
        # chosen ONLY from information available at t: a candidate is eligible if it has
        # complete prices over the estimation window [t-L, t), and eligible names are
        # ranked by trailing dollar volume over the same window. (The R1 version also
        # required complete prices over the holding period [t, t+h) and screened
        # candidates on their full-sample number of valid prices; both used information
        # from after t and were removed in R2. On the cached data the corrected rule
        # selects identical constituents at every rebalance -- see r2_survivor_pit.py.)
        # A name whose price stops during a holding period earns a zero return for the
        # rest of that period (forward-fill, then NaN -> 0).
        p = px.ffill(limit=5)
        keep = list(p.columns)
        R = p.pct_change().clip(-0.5, 0.5)
        valid = p.notna().values
        dv = (p * (vol[keep].reindex(p.index).ffill() if vol is not None else 1.0))
        dvv = dv.values
        nsub = min(cfg["n_stocks"], len(keep))
        Rf = R.fillna(0.0)

        def cols_fn(t, L=c["L"], h=c["h"]):
            ok = valid[t - L:t].all(0)
            score = np.where(ok, np.nanmean(dvv[t - L:t], axis=0), -np.inf)
            return np.sort(np.argsort(-score)[:nsub])
        P = precompute(Rf, c, cols_fn=cols_fn)
        P["R_index"] = Rf.index
        return P, c, Rf
    Rr = clean_universe(px, cfg, cfg["n_stocks"] if universe == "stocks" else None)
    P = precompute(Rr, c)
    P["R_index"] = Rr.index
    return P, c, Rr


def run_core(P, cfg, seed=None, include_dfs=True, variants=("identity", "cc", "mix", "nn"),
             tc_variants=True, verbose=True):
    R, N, L = P["R"], P["N"], cfg["L"]
    W = {m: np.zeros((R, N)) for m in
         ["sample", "lw_identity", "lw_cc", "oas", "ewma", "pca", "mp_clip", "equal",
          "nls", "dccnl"]}
    DL = dict(lw_identity=np.zeros(R), lw_cc=np.zeros(R))
    t0 = time.time()
    for r in range(R):
        S, X = P["S"][r], P["X"][r]
        dI = lw_identity(X); dC = lw_cc(X)
        DL["lw_identity"][r] = dI; DL["lw_cc"][r] = dC
        W["sample"][r] = gmv(S)
        W["lw_identity"][r] = gmv(shrink(S, P["FI"][r], dI))
        W["lw_cc"][r] = gmv(shrink(S, P["FCC"][r], dC))
        W["oas"][r] = gmv(oas_cov(X))
        W["ewma"][r] = gmv(ewma_cov(X, cfg["ewma_lambda"]))
        W["pca"][r] = gmv(P["FPCA"][r])
        W["mp_clip"][r] = gmv(mp_clip_cov(S, L))
        W["equal"][r] = np.ones(N) / N
        W["nls"][r] = gmv(nonlinear_shrinkage(X))
        W["dccnl"][r] = gmv(dccnl_cov(X))
    if verbose:
        print(f"    baselines+NLS done ({time.time()-t0:.0f}s)")
    if include_dfs:
        for v in variants:
            W[f"dfs_{v}"], DL[f"dfs_{v}"] = walk_forward_dfs(
                P, v, 0.0, cfg["psi"], cfg, seed=seed, tag=v if verbose else "")
        if tc_variants:
            for c_ in cfg["costs"]:
                k = f"dfs_tc_{int(c_*1e4)}bp"
                W[k], DL[k] = walk_forward_dfs(P, "identity", c_, cfg["psi"], cfg,
                                               seed=seed, tag=k if verbose else "")
    return W, DL


def tabulate(W, P, cfg, drift=False):
    r0 = cfg["dfs"]["min_train"]
    rows, store = [], {}
    for m, w in W.items():
        met, vr, rets, dates, to_r, nets = evaluate(w, P, r0, cfg, drift=drift)
        rows.append(dict(method=m, **met))
        store[m] = dict(var_r=vr, rets=rets, dates=dates, to=to_r, nets=nets)
    return pd.DataFrame(rows).set_index("method"), store


# =============================================================================
# PART: core
# =============================================================================
def part_core(out, cfg, logf):
    results = {}
    for uni in (["stocks", "etfs"] if not QUICK else ["stocks"]):
        log(f"[core] universe={uni}", logf)
        P, c, Rr = build_P(uni, cfg, out)
        log(f"  N={P['N']} rebalances={P['R']} span {P['dates'][0].date()}..{P['dates'][-1].date()}", logf)
        W, DL = run_core(P, c, seed=cfg["seed"])
        res, store = tabulate(W, P, c)
        res = res.sort_values("realized_var_ann")
        save(res.round(4), out, f"tabR1_core_{uni}")
        np.savez_compressed(os.path.join(out, f"core_{uni}.npz"),
                            **{f"W_{k}": v for k, v in W.items()},
                            **{f"rets_{k}": v["rets"] for k, v in store.items()},
                            **{f"to_{k}": v["to"] for k, v in store.items()},
                            dates=store["lw_identity"]["dates"].astype("datetime64[ns]").astype(np.int64),
                            cols=P["cols"], hdates=P["oos_dates_all"].astype("datetime64[ns]").astype(np.int64))
        pd.DataFrame(DL, index=P["dates"]).to_csv(os.path.join(out, f"deltas_{uni}.csv"))
        results[uni] = res.to_dict()
        log(f"  best variance: {res.index[0]} = {res['realized_var_ann'].iloc[0]:.2f}", logf)
    return results


# =============================================================================
# PART: inference
# =============================================================================
def part_inference(out, cfg, logf):
    summary = {}
    for uni in (["stocks", "etfs"] if not QUICK else ["stocks"]):
        f = os.path.join(out, f"core_{uni}.npz")
        if not os.path.exists(f):
            log(f"[inference] missing {f}; run --part core first", logf); continue
        z = np.load(f, allow_pickle=True)
        methods = sorted([k[5:] for k in z.files if k.startswith("rets_")])
        rets = {m: z[f"rets_{m}"] for m in methods}
        to = {m: z[f"to_{m}"] for m in methods}
        n = len(rets["lw_identity"])
        rng = np.random.default_rng(cfg["boot"]["seed"])
        idx = stationary_bootstrap_idx(n, cfg["boot"]["block"], cfg["boot"]["B"], rng)
        base = "lw_identity"
        rows, tmat, order = [], [], []
        for m in methods:
            if m == base:
                continue
            rv, tsv = boot_test(rets[m], rets[base], logvar_stat, cfg, idx=idx)
            rs, _ = boot_test(rets[m], rets[base], sharpe_stat, cfg, idx=idx)
            nm = net_returns(rets[m], to[m], cfg["primary_cost"], cfg["h"])
            nb = net_returns(rets[base], to[base], cfg["primary_cost"], cfg["h"])
            rsn, _ = boot_test(nm, nb, sharpe_stat, cfg, idx=idx)
            rows.append(dict(method=m,
                             log_var_diff=rv["stat"], t_var=rv["t"],
                             p_var_hac=rv["p_hac"], p_var_boot=rv["p_boot"],
                             var_ratio=math.exp(rv["stat"]),
                             sharpe_diff=rs["stat"], p_sharpe_boot=rs["p_boot"],
                             sharpe_diff_net25=rsn["stat"], p_sharpe_net25_boot=rsn["p_boot"]))
            tmat.append(tsv); order.append(m)
        tmat = np.array(tmat).T
        tstats = np.array([r["t_var"] for r in rows])
        padj = romano_wolf(tstats, tmat)
        df = pd.DataFrame(rows).set_index("method")
        df["p_var_RomanoWolf"] = padj
        df = df.sort_values("t_var")
        save(df.round(4), out, f"tabR2_inference_{uni}")
        summary[uni] = df.to_dict()
        log(f"[inference] {uni}: Romano-Wolf survivors at 5% = "
            f"{list(df.index[df.p_var_RomanoWolf<0.05])}", logf)
    return summary


# =============================================================================
# PART: attribution  (the 'is it just more shrinkage / a vol tilt?' test)
# =============================================================================
def part_attribution(out, cfg, logf):
    summary = {}
    for uni in (["stocks", "etfs"] if not QUICK else ["stocks"]):
        P, c, Rr = build_P(uni, cfg, out)
        r0 = c["dfs"]["min_train"]
        grid = np.linspace(0, 1, 26 if QUICK else 51)

        # --- (a) ex-post optimal CONSTANT delta sweep (the key null) ---
        rows = []
        for d in grid:
            Wd = np.zeros((P["R"], P["N"]))
            for r in range(P["R"]):
                Wd[r] = gmv(shrink(P["S"][r], P["FI"][r], d))
            met, _, _, _, _, _ = evaluate(Wd, P, r0, c)
            rows.append(dict(delta=d, **met))
        sweep = pd.DataFrame(rows)
        save(sweep.round(4), out, f"tabR3_delta_sweep_{uni}", index=False)
        dstar = float(sweep.loc[sweep.realized_var_ann.idxmin(), "delta"])
        vstar = float(sweep.realized_var_ann.min())

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sweep.delta, sweep.realized_var_ann, "b-", label="OOS variance, constant $\\delta$")
        ax.axvline(dstar, ls=":", color="b")
        ax.set_xlabel("constant shrinkage intensity $\\delta$")
        ax.set_ylabel("OOS annualised variance (%$^2$)")
        ax2 = ax.twinx(); ax2.plot(sweep.delta, sweep.turnover_ann, "r--", label="turnover")
        ax2.set_ylabel("annual turnover", color="r")
        ax.set_title(f"Ex-post optimal constant intensity ({uni})")
        plt.tight_layout(); plt.savefig(os.path.join(out, f"figR1_delta_sweep_{uni}.png"), dpi=200); plt.close()

        # --- (b) volatility-tilt vs correlation decomposition ---
        W, DL = run_core(P, c, seed=cfg["seed"], variants=("identity",), tc_variants=False, verbose=False)
        Wcross = {"vol_dfs_corr_lw": np.zeros((P["R"], P["N"])),
                  "vol_lw_corr_dfs": np.zeros((P["R"], P["N"]))}
        mdl = None
        for r in range(P["R"]):
            S = P["S"][r]; FI = P["FI"][r]
            dI = DL["lw_identity"][r]
            d_dfs = DL["dfs_identity"][r]
            if not np.isfinite(d_dfs):
                d_dfs = dI
            A = shrink(S, FI, dI); B = shrink(S, FI, d_dfs)
            sa = np.sqrt(np.diag(A)); sb = np.sqrt(np.diag(B))
            Ca = A / np.outer(sa, sa); Cb = B / np.outer(sb, sb)
            Wcross["vol_dfs_corr_lw"][r] = gmv(Ca * np.outer(sb, sb))
            Wcross["vol_lw_corr_dfs"][r] = gmv(Cb * np.outer(sa, sa))
        dec = {}
        for k, w in list(Wcross.items()) + [("lw_identity", W["lw_identity"]),
                                            ("dfs_identity", W["dfs_identity"])]:
            met, _, _, _, _, _ = evaluate(w, P, r0, c)
            dec[k] = met["realized_var_ann"]
        decdf = pd.DataFrame.from_dict(dec, orient="index", columns=["realized_var_ann"])
        decdf.loc["total_effect"] = dec["dfs_identity"] - dec["lw_identity"]
        decdf.loc["vol_channel"] = dec["vol_dfs_corr_lw"] - dec["lw_identity"]
        decdf.loc["corr_channel"] = dec["vol_lw_corr_dfs"] - dec["lw_identity"]
        save(decdf.round(4), out, f"tabR4_vol_vs_corr_{uni}")

        # --- (c) weight diagnostics + break-even cost ---
        Wall, DLall = run_core(P, c, seed=cfg["seed"])
        res, store = tabulate(Wall, P, c)
        diag = []
        for m, w in Wall.items():
            ww = w[r0:]
            eff_n = 1.0 / (ww ** 2).sum(1)
            short = np.minimum(ww, 0).sum(1)
            invvar_corr = []
            for i, r in enumerate(range(r0, P["R"])):
                iv = 1.0 / np.maximum(np.diag(P["S"][r]), 1e-12)
                invvar_corr.append(np.corrcoef(ww[i], iv)[0, 1])
            diag.append(dict(method=m, eff_N=eff_n.mean(), gross=np.abs(ww).sum(1).mean(),
                             short_frac=-short.mean(), corr_w_invvar=np.nanmean(invvar_corr),
                             max_weight=ww.max()))
        save(pd.DataFrame(diag).set_index("method").round(4), out, f"tabR5_weight_diag_{uni}")

        # break-even cost of DFS-TC vs LW-I (net Sharpe and net return)
        be = []
        cs = np.linspace(0.0, 0.0150, 61)
        base = store["lw_identity"]
        for m in [k for k in store if k.startswith("dfs")] + ["nls", "lw_cc", "equal"]:
            s = store[m]
            srm = [net_returns(s["rets"], s["to"], cc, c["h"]) for cc in cs]
            srb = [net_returns(base["rets"], base["to"], cc, c["h"]) for cc in cs]
            d = np.array([x.mean() / x.std() - y.mean() / y.std() for x, y in zip(srm, srb)])
            sgn = np.sign(d)
            cross = np.where(np.diff(sgn) != 0)[0]
            be.append(dict(method=m, net_sharpe_adv_at_0=d[0],
                           net_sharpe_adv_at_25bp=d[int(np.argmin(abs(cs - 0.0025)))],
                           breakeven_cost_bp=(1e4 * cs[cross[0]] if len(cross) else np.nan)))
        save(pd.DataFrame(be).set_index("method").round(4), out, f"tabR6_breakeven_{uni}")

        summary[uni] = dict(delta_star=dstar, var_at_delta_star=vstar,
                            dfs_var=float(res.loc["dfs_identity", "realized_var_ann"]),
                            lw_var=float(res.loc["lw_identity", "realized_var_ann"]),
                            vol_channel=float(decdf.loc["vol_channel", "realized_var_ann"]),
                            corr_channel=float(decdf.loc["corr_channel", "realized_var_ann"]))
        log(f"[attribution] {uni}: delta*={dstar:.3f} V*={vstar:.2f} | "
            f"DFS={summary[uni]['dfs_var']:.2f} LW={summary[uni]['lw_var']:.2f}", logf)
    return summary


# =============================================================================
# PART: hyper (L, h, drift-aware turnover)
# =============================================================================
def part_hyper(out, cfg, logf):
    combos = [(250, 21), (125, 21), (500, 21), (250, 5), (250, 63)]
    if QUICK:
        combos = [(250, 21), (125, 21)]
    rows = []
    for (L, h) in combos:
        P, c, _ = build_P("stocks", cfg, out, L=L, h=h)
        W, DL = run_core(P, c, seed=cfg["seed"], variants=("identity", "mix"),
                         tc_variants=False, verbose=False)
        for drift in [False, True]:
            res, _ = tabulate(W, P, c, drift=drift)
            for m in ["sample", "lw_identity", "lw_cc", "nls", "dccnl", "dfs_identity", "dfs_mix", "equal"]:
                if m in res.index:
                    rows.append(dict(L=L, h=h, drift_aware_turnover=drift, method=m,
                                     **res.loc[m].to_dict()))
        log(f"[hyper] L={L} h={h} done", logf)
    df = pd.DataFrame(rows)
    save(df.round(4), out, "tabR7_hyper_L_h", index=False)
    piv = df[~df.drift_aware_turnover].pivot_table(index="method", columns=["L", "h"],
                                                   values="realized_var_ann")
    save(piv.round(2), out, "tabR7b_hyper_variance_grid")
    return dict(n_rows=len(df))


# =============================================================================
# PART: subperiod
# =============================================================================
SUBPERIODS = [("GFC 2007-2009", "2007-11-01", "2009-06-30"),
              ("Recovery 2009-2019", "2009-07-01", "2019-12-31"),
              ("COVID 2020", "2020-01-01", "2020-12-31"),
              ("Rates 2021-2024", "2021-01-01", "2024-12-31")]


def part_subperiod(out, cfg, logf):
    outp = {}
    for uni in (["stocks", "etfs"] if not QUICK else ["stocks"]):
        f = os.path.join(out, f"core_{uni}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        dates = pd.to_datetime(z["dates"])
        methods = sorted([k[5:] for k in z.files if k.startswith("rets_")])
        rng = np.random.default_rng(cfg["boot"]["seed"])
        rows = []
        for name, a, b in SUBPERIODS:
            mask = (dates >= a) & (dates <= b)
            if mask.sum() < 60:
                continue
            idx = stationary_bootstrap_idx(int(mask.sum()), cfg["boot"]["block"],
                                           min(400, cfg["boot"]["B"]), rng)
            base = z["rets_lw_identity"][mask]
            for m in methods:
                r = z[f"rets_{m}"][mask]
                rec = dict(period=name, method=m, ann_vol=100 * r.std() * np.sqrt(252),
                           ann_ret=100 * r.mean() * 252, n_days=int(mask.sum()))
                if m != "lw_identity":
                    rv, _ = boot_test(r, base, logvar_stat, cfg, idx=idx)
                    rec.update(var_ratio_vs_LWI=math.exp(rv["stat"]), p_boot=rv["p_boot"])
                rows.append(rec)
        df = pd.DataFrame(rows)
        save(df.round(4), out, f"tabR8_subperiod_{uni}", index=False)
        outp[uni] = len(df)
        log(f"[subperiod] {uni} done", logf)
    return outp


# =============================================================================
# PART: placebo
# =============================================================================
def part_placebo(out, cfg, logf):
    P, c, _ = build_P("stocks", cfg, out)
    r0 = c["dfs"]["min_train"]
    rows = []
    for shuf in [False, True]:
        for rep in range(1 if not shuf else (2 if QUICK else 5)):
            W, dl = walk_forward_dfs(P, "identity", 0.0, c["psi"], c,
                                     seed=cfg["seed"] + rep, shuffle_H=shuf, tag="")
            met, _, _, _, _, _ = evaluate(W, P, r0, c)
            rows.append(dict(shuffled=shuf, rep=rep, mean_delta=np.nanmean(dl), **met))
    base_P, _ = build_P("stocks", cfg, out)[0], None
    df = pd.DataFrame(rows)
    save(df.round(4), out, "tabR9_placebo", index=False)
    real = df[~df.shuffled].realized_var_ann.iloc[0]
    pl = df[df.shuffled].realized_var_ann
    log(f"[placebo] real DFS var={real:.2f}; shuffled mean={pl.mean():.2f} "
        f"(min {pl.min():.2f}, max {pl.max():.2f})", logf)
    return dict(real=float(real), placebo_mean=float(pl.mean()), placebo_sd=float(pl.std()))


# =============================================================================
# PART: seeds
# =============================================================================
def part_seeds(out, cfg, logf):
    seeds = cfg["seeds_list"][:3] if QUICK else cfg["seeds_list"]
    rows = []
    for uni in (["stocks"] if QUICK else ["stocks", "etfs"]):
        P, c, _ = build_P(uni, cfg, out)
        r0 = c["dfs"]["min_train"]
        for s in seeds:
            for v in ["identity", "mix", "nn"]:
                W, dl = walk_forward_dfs(P, v, 0.0, c["psi"], c, seed=s, tag="")
                met, _, _, _, _, _ = evaluate(W, P, r0, c)
                rows.append(dict(universe=uni, variant=v, seed=s,
                                 mean_delta=np.nanmean(dl), **met))
            Wt, dlt = walk_forward_dfs(P, "identity", c["primary_cost"], c["psi"], c, seed=s, tag="")
            met, _, _, _, _, _ = evaluate(Wt, P, r0, c)
            rows.append(dict(universe=uni, variant="tc25bp", seed=s,
                             mean_delta=np.nanmean(dlt), **met))
            log(f"[seeds] {uni} seed={s} done", logf)
    df = pd.DataFrame(rows)
    save(df.round(4), out, "tabR10_seed_raw", index=False)
    agg = df.groupby(["universe", "variant"]).agg(
        var_mean=("realized_var_ann", "mean"), var_sd=("realized_var_ann", "std"),
        var_min=("realized_var_ann", "min"), var_max=("realized_var_ann", "max"),
        vol_mean=("ann_vol", "mean"), vol_sd=("ann_vol", "std"),
        to_mean=("turnover_ann", "mean"), to_sd=("turnover_ann", "std"),
        delta_mean=("mean_delta", "mean"), delta_sd=("mean_delta", "std"))
    save(agg.round(4), out, "tabR10b_seed_summary")
    return agg.round(4).to_dict()


# =============================================================================
# PART: survivorship (rotating point-in-time universe)
# =============================================================================
def part_survivor(out, cfg, logf):
    P, c, _ = build_P("stocks", cfg, out, rotating=True)
    log(f"[survivor] rotating universe N={P['N']} rebalances={P['R']}", logf)
    W, DL = run_core(P, c, seed=cfg["seed"], variants=("identity", "mix"), tc_variants=True)
    res, store = tabulate(W, P, c)
    res = res.sort_values("realized_var_ann")
    save(res.round(4), out, "tabR11_survivorship_rotating")
    # inference vs LW-I on this universe
    n = len(store["lw_identity"]["rets"])
    rng = np.random.default_rng(cfg["boot"]["seed"])
    idx = stationary_bootstrap_idx(n, cfg["boot"]["block"], cfg["boot"]["B"], rng)
    rows = []
    for m in store:
        if m == "lw_identity":
            continue
        rv, _ = boot_test(store[m]["rets"], store["lw_identity"]["rets"], logvar_stat, cfg, idx=idx)
        rows.append(dict(method=m, var_ratio=math.exp(rv["stat"]), t=rv["t"], p_boot=rv["p_boot"]))
    save(pd.DataFrame(rows).set_index("method").round(4), out, "tabR11b_survivorship_tests")
    # how much the universe actually rotates
    turn = [len(set(P["cols"][r]) ^ set(P["cols"][r - 1])) / 2 for r in range(1, P["R"])]
    log(f"[survivor] avg constituent changes per rebalance: {np.mean(turn):.2f}", logf)
    return dict(rotation=float(np.mean(turn)),
                dfs=float(res.loc["dfs_identity", "realized_var_ann"]),
                lwi=float(res.loc["lw_identity", "realized_var_ann"]))


# =============================================================================
# PART: sim  (SEs, reversal DGP, FOC check, NN learning curve)
# =============================================================================
def make_sigma_factor(N, K, rng):
    B = np.zeros((N, K)); B[:, 0] = rng.normal(1.0, 0.3, N)
    for k in range(1, K):
        B[:, k] = rng.normal(0.0, 0.5, N)
    fvar = np.array([1.0e-4, 3.0e-5, 2.0e-5, 1.5e-5, 1.0e-5])[:K]
    idio = (rng.lognormal(np.log(0.012), 0.35, N)) ** 2
    return B @ np.diag(fvar) @ B.T + np.diag(idio)


def make_sigma_hetero(N, rng, spread=8.0):
    """Multi-asset-like DGP: huge vol dispersion, modest common factor.
    Here shrinking toward a SCALED IDENTITY is harmful -> delta_D may be < delta_F."""
    sd = np.exp(np.linspace(np.log(0.002), np.log(0.002 * spread), N))
    rng.shuffle(sd)
    b = rng.normal(0.6, 0.25, N)
    C = np.outer(b, b) * 0.35
    np.fill_diagonal(C, 1.0)
    C = np.clip(C, -0.9, 0.95); np.fill_diagonal(C, 1.0)
    return C * np.outer(sd, sd)


def make_sigma_equicorr(N, rng, rho=0.45, sd=0.015):
    C = np.full((N, N), rho); np.fill_diagonal(C, 1.0)
    return C * sd ** 2


def sample_returns(Sigma, T, dist, nu, rng):
    N = Sigma.shape[0]
    Lc = np.linalg.cholesky(Sigma + 1e-14 * np.eye(N))
    Z = rng.standard_normal((T, N)) @ Lc.T
    if dist == "t":
        g = rng.chisquare(nu, T) / nu
        Z = Z / np.sqrt(g)[:, None] * np.sqrt((nu - 2) / nu)
    return Z


def foc_check(S, F, Sigma, d):
    """SUPERSEDED. Early FOC diagnostic that used the derivative as printed in the
    original submission (second term without c(delta)); its output tabR13_* is not
    used in the paper. The Section 5.4 checks use the corrected derivative:
    see dfs_pipeline_oracle.grad_decision and r1_prop1_check.py."""
    D = F - S
    Sg = shrink(S, F, d)
    Sinv = np.linalg.pinv(Sg)
    w = gmv(Sg)
    dec = float(w @ Sigma @ (Sinv @ D @ w - w * (w @ D @ w)))
    fro = float(np.trace((S + d * D - Sigma).T @ D))
    return dec, fro


def part_sim(out, cfg, logf):
    rng = np.random.default_rng(cfg["seed"])
    grid = np.linspace(0, 1, 21 if QUICK else 41)
    M = 50 if QUICK else 200
    Ns = [30, 60] if QUICK else [30, 60, 120]
    Ts = [120, 250] if QUICK else [60, 120, 250, 500]
    rows, foc_rows = [], []
    dgps = {"factor": lambda N: make_sigma_factor(N, 3, rng),
            "hetero": lambda N: make_sigma_hetero(N, rng),
            "equicorr": lambda N: make_sigma_equicorr(N, rng)}
    for dgp, mk in dgps.items():
        for N in Ns:
            Sigma = mk(N)
            for T in Ts:
                for dist in (["gauss"] if dgp != "factor" else ["gauss", "t"]):
                    dF, dD, dLW, VF, VD, VLW = [], [], [], [], [], []
                    for m in range(M):
                        X = sample_returns(Sigma, T, dist, 5, rng); X = X - X.mean(0)
                        S = X.T @ X / T; F = target_identity(S)
                        Vd = np.array([(lambda w: w @ Sigma @ w)(gmv(shrink(S, F, d))) for d in grid])
                        Fd = np.array([np.linalg.norm(shrink(S, F, d) - Sigma, "fro") ** 2 for d in grid])
                        iF, iD = int(np.argmin(Fd)), int(np.argmin(Vd))
                        dlw = lw_identity(X)
                        dF.append(grid[iF]); dD.append(grid[iD]); dLW.append(dlw)
                        VF.append(Vd[iF]); VD.append(Vd[iD])
                        VLW.append((lambda w: w @ Sigma @ w)(gmv(shrink(S, F, dlw))))
                        if m < 5:
                            dec, fro = foc_check(S, F, Sigma, grid[iD])
                            foc_rows.append(dict(dgp=dgp, N=N, T=T, dist=dist,
                                                 delta=grid[iD], decision_FOC=dec, frobenius_FOC=fro))
                    VF, VD, VLW = map(np.array, (VF, VD, VLW))
                    gap = (VF - VD) / VD; gaplw = (VLW - VD) / VD
                    rows.append(dict(dgp=dgp, N=N, T=T, dist=dist, NT=N / T,
                                     dF=np.mean(dF), dF_se=np.std(dF) / np.sqrt(M),
                                     dD=np.mean(dD), dD_se=np.std(dD) / np.sqrt(M),
                                     dLW=np.mean(dLW),
                                     frac_dD_gt_dF=float(np.mean(np.array(dD) > np.array(dF))),
                                     gap_F_pct=100 * gap.mean(), gap_F_se=100 * gap.std() / np.sqrt(M),
                                     gap_LW_pct=100 * gaplw.mean(),
                                     gap_LW_se=100 * gaplw.std() / np.sqrt(M)))
        log(f"[sim] dgp={dgp} done", logf)
    df = pd.DataFrame(rows)
    save(df.round(4), out, "tabR12_sim_extended", index=False)
    save(pd.DataFrame(foc_rows).round(8), out, "tabR13_prop1_FOC_check", index=False)

    rev = df[(df.dD < df.dF)]
    log(f"[sim] cells where decision-optimal < Frobenius-optimal: {len(rev)} / {len(df)} "
        f"(dgps: {sorted(rev.dgp.unique().tolist())})", logf)

    # ---- DFS-NN learning curve: is the NN failure a sample-size issue? ----
    lc_rows = []
    Nn, Tn, h = 60, 250, 21
    reb_max = 120 if QUICK else 600
    Sig_hi = make_sigma_factor(Nn, 3, rng)
    Sig_lo = make_sigma_factor(Nn, 3, rng) * 0.35      # calm regime -> different optimal delta
    state = []
    blocks = []
    s = 0
    for r in range(reb_max + 40):
        if rng.random() < 0.06:
            s = 1 - s
        state.append(s)
        Sg = Sig_hi if s == 1 else Sig_lo
        blocks.append(sample_returns(Sg, Tn + h, "gauss", 5, rng))
    Pn = dict(S=[], FI=[], FCC=[], FPCA=[], H=[], feat=[])
    for b in blocks:
        X = b[:Tn] - b[:Tn].mean(0); S = X.T @ X / Tn
        sd = np.sqrt(np.diag(S)); corr = S / np.outer(sd, sd); vals = np.linalg.eigvalsh(corr)
        Pn["S"].append(S); Pn["FI"].append(target_identity(S)); Pn["FCC"].append(target_cc(S))
        Pn["FPCA"].append(target_pca(S, 3)); Pn["H"].append(b[Tn:Tn + h])
        Pn["feat"].append([(corr.sum() - Nn) / (Nn * (Nn - 1)), vals[-1] / Nn,
                           np.log(vals[-1] / max(vals[0], 1e-9)), sd.mean() * np.sqrt(252),
                           (vals < (1 + np.sqrt(Nn / Tn)) ** 2).mean()])
    for k in Pn:
        Pn[k] = np.array(Pn[k])
    Pn["N"] = Nn; Pn["R"] = len(blocks); Pn["Nfull"] = Nn
    Pn["cols"] = np.tile(np.arange(Nn), (Pn["R"], 1))
    Pn["hdates"] = [pd.DatetimeIndex(pd.date_range("2000-01-01", periods=h)) for _ in range(Pn["R"])]
    c = json.loads(json.dumps(cfg)); c["dfs"] = cfg["dfs"]; c["h"] = h
    test_idx = np.arange(Pn["R"] - 40, Pn["R"])
    for ntr in ([40, 80] if QUICK else [24, 48, 96, 192, 384, reb_max - 40]):
        ntr = min(ntr, Pn["R"] - 40)
        for v in ["identity", "nn"]:
            mdl = train_dfs(Pn, np.arange(0, ntr), v, 0.0, cfg["psi"], c, seed=cfg["seed"])
            W = np.zeros((Pn["R"], Nn)); dl = np.full(Pn["R"], np.nan)
            for r in test_idx:
                W[r], dl[r] = apply_dfs(mdl, Pn, r, c)
            pr = np.concatenate([Pn["H"][r] @ W[r] for r in test_idx])
            lc_rows.append(dict(n_train=ntr, variant=v, oos_var_ann=100 ** 2 * pr.var() * 252,
                                delta_sd=np.nanstd(dl[test_idx])))
    lc = pd.DataFrame(lc_rows)
    save(lc.round(4), out, "tabR14_dfsnn_learning_curve", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for v, g in lc.groupby("variant"):
        ax.plot(g.n_train, g.oos_var_ann, marker="o", label=f"DFS-{v}")
    ax.set_xscale("log"); ax.set_xlabel("number of training rebalances")
    ax.set_ylabel("OOS annualised variance (%$^2$)")
    ax.set_title("Does DFS-NN need more data? (regime-switching simulation)")
    ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "figR2_dfsnn_learning_curve.png"), dpi=200); plt.close()
    return dict(reversal_cells=int(len(rev)), total_cells=int(len(df)))


# =============================================================================
# PART: audit
# =============================================================================
def part_audit(out, cfg, logf):
    P, c, Rr = build_P("stocks", cfg, out)
    probs = []
    L, h = c["L"], c["h"]
    idx = Rr.index
    for r in range(P["R"]):
        t = L + r * h
        est_end = idx[t - 1]; hold_start = idx[t]; hold_end = idx[min(t + h - 1, len(idx) - 1)]
        if not (est_end < hold_start <= hold_end):
            probs.append(("ordering", r))
    # training set at rebalance r uses rebalances 0..r-1, whose holding windows end at t_r
    for r in range(c["dfs"]["min_train"], P["R"]):
        last_train_end = P["hdates"][r - 1][-1]
        decision_date = P["dates"][r]
        if last_train_end > decision_date:
            probs.append(("lookahead", r))
    # holding periods are non-overlapping and exactly tile the OOS sample
    allh = np.concatenate([d.values for d in P["hdates"][c["dfs"]["min_train"]:]])
    if len(allh) != len(np.unique(allh)):
        probs.append(("overlap", -1))
    res = dict(n_rebalances=P["R"], n_oos_days=int(len(allh)),
               oos_start=str(pd.Timestamp(allh[0]).date()), oos_end=str(pd.Timestamp(allh[-1]).date()),
               problems=probs, universe_tickers=list(Rr.columns))
    with open(os.path.join(out, "audit_lookahead.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    pd.Series(Rr.columns).to_csv(os.path.join(out, "tabR0_universe_stocks_ACTUAL.csv"), index=False)
    log(f"[audit] problems found: {len(probs)}; OOS {res['oos_start']}..{res['oos_end']} "
        f"({res['n_oos_days']} days). Appendix A must list exactly the {len(Rr.columns)} "
        f"tickers in tabR0_universe_stocks_ACTUAL.csv", logf)
    return res


# =============================================================================
# REPORT
# =============================================================================
def part_report(out, cfg, logf, summary):
    lines = ["# Revision results — numbers to paste into the .tex", "",
             f"_generated {time.strftime('%Y-%m-%d %H:%M')}_", ""]

    def tryread(name):
        p = os.path.join(out, name + ".csv")
        return pd.read_csv(p, index_col=0) if os.path.exists(p) else None

    for uni in ["stocks", "etfs"]:
        t = tryread(f"tabR1_core_{uni}")
        if t is None:
            continue
        lines += [f"## {uni}: core table (now including nonlinear shrinkage)", "",
                  t[["ann_vol", "realized_var_ann", "turnover_ann", "sharpe",
                     "sharpe_net_25bp"]].round(3).to_markdown(), ""]
        inf = tryread(f"tabR2_inference_{uni}")
        if inf is not None:
            lines += ["### Inference vs LW-I (studentised stationary bootstrap, Romano-Wolf adjusted)", "",
                      inf[["var_ratio", "t_var", "p_var_boot", "p_var_RomanoWolf",
                           "p_sharpe_net25_boot"]].round(4).to_markdown(), ""]
        sw = tryread(f"tabR3_delta_sweep_{uni}")
        if sw is not None:
            k = sw.reset_index()
            best = k.loc[k.realized_var_ann.idxmin()]
            lines += [f"**Ex-post optimal constant intensity**: delta* = {best['delta']:.3f}, "
                      f"variance {best['realized_var_ann']:.2f}. Compare with DFS-I. "
                      "If these are close, state explicitly that DFS attains the ex-post "
                      "optimal constant intensity out of sample, and that this -- not "
                      "state-dependence -- is the mechanism.", ""]
        dec = tryread(f"tabR4_vol_vs_corr_{uni}")
        if dec is not None:
            lines += ["### Volatility-tilt vs correlation channel", "",
                      dec.round(3).to_markdown(),
                      "", "(answers Kim et al. 2025: is the gain just a low-vol tilt?)", ""]
        be = tryread(f"tabR6_breakeven_{uni}")
        if be is not None:
            lines += ["### Break-even transaction cost", "", be.round(3).to_markdown(), ""]
    for nm, title in [("tabR7b_hyper_variance_grid", "Sensitivity to (L,h)"),
                      ("tabR10b_seed_summary", "Seed dispersion"),
                      ("tabR11_survivorship_rotating", "Rotating point-in-time universe"),
                      ("tabR12_sim_extended", "Extended simulation (3 DGPs, with SEs)"),
                      ("tabR14_dfsnn_learning_curve", "DFS-NN learning curve"),
                      ("tabR9_placebo", "Placebo (shuffled training targets)")]:
        t = tryread(nm)
        if t is not None:
            lines += [f"## {title}", "", t.round(4).to_markdown(), ""]
    lines += ["## Checklist of .tex edits", "",
              "1. Add nonlinear shrinkage (Ledoit-Wolf 2020) and the DCC-NL proxy to Tables 2 and 4; "
              "cite Ledoit-Wolf (2017, 2020) and Engle-Ledoit-Wolf (2019) as *implemented* benchmarks.",
              "2. Replace the NW t-statistic with the studentised bootstrap test and report "
              "Romano-Wolf adjusted p-values; say in Section 6.1 that you correct for multiple testing.",
              "3. Add a subsection 'Is DFS more than higher shrinkage?' built on tabR3/figR1 and tabR4.",
              "4. Reframe the ETF result: the 27% is relative to an identity target that is "
              "inappropriate for a heterogeneous-volatility universe; the correct claim is that DFS "
              "*learns not to shrink*. Compare to LW-CC and the sample matrix in the abstract.",
              "5. Replace 'doubles the net Sharpe' with the break-even cost (tabR6).",
              "6. Add the seed table (tabR10b) and delete the sentence claiming seed variation is "
              "'small by construction'.",
              "7. Add the (L,h) grid and the drift-aware turnover rows (tabR7).",
              "8. Add the subperiod table (tabR8) and point Figure 6 at it.",
              "9. Add the placebo test (tabR9) as a falsification check, as in Henide (2026).",
              "10. Add the rotating point-in-time universe (tabR11) to Section 7 as a "
              "survivorship-bias mitigation, not just a caveat.",
              "11. Report simulation standard errors and the hetero/equicorrelated DGPs "
              "(tabR12); state honestly in which DGPs the ordering reverses. "
              "Soften 'the decision-optimal intensity is larger in 23 of 24 configurations' to "
              "a statement conditional on the factor DGP + identity target.",
              "12. Add the numerical FOC check (tabR13) as a one-line footnote to Proposition 1.",
              "13. Replace the DFS-NN negative result with the learning curve (figR2) if it "
              "supports the sample-size explanation; if it does not, say so.",
              "14. Fix Appendix A to list exactly the tickers in tabR0_universe_stocks_ACTUAL.csv.", ""]
    with open(os.path.join(out, "REVISION_NOTES.md"), "w") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(out, "revision_summary.json"), "w") as fh:
        def _js(o):
            if isinstance(o, dict):
                return {str(k): _js(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_js(v) for v in o]
            return o
        json.dump(_js(summary), fh, indent=2, default=str)
    log("[report] wrote REVISION_NOTES.md", logf)


# =============================================================================
# MAIN
# =============================================================================
ALL_PARTS = ["data", "audit", "core", "inference", "attribution", "hyper",
             "subperiod", "placebo", "seeds", "survivor", "sim", "report"]


def main():
    global QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "results_original"))
    ap.add_argument("--part", default="all")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    QUICK = a.quick
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    logf = os.path.join(out, "revision_run.log")
    parts = ALL_PARTS if a.part == "all" else [p.strip() for p in a.part.split(",")]
    np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    cfg = CFG
    if QUICK:
        cfg["dfs"] = dict(cfg["dfs"]); cfg["dfs"]["steps"] = 80
        cfg["boot"] = dict(cfg["boot"]); cfg["boot"]["B"] = 250
    log(f"=== DFS pipeline | out={out} | quick={QUICK} | parts={parts} ===", logf)
    T0 = time.time(); summary = {}
    for p in parts:
        t0 = time.time()
        try:
            if p == "data":
                for u, tk in [("stocks", cfg["stocks"]), ("etfs", cfg["etfs"])]:
                    load_prices(tk, cfg, out, u); log(f"[data] cached {u}", logf)
            elif p == "audit":       summary["audit"] = part_audit(out, cfg, logf)
            elif p == "core":        summary["core"] = part_core(out, cfg, logf)
            elif p == "inference":   summary["inference"] = part_inference(out, cfg, logf)
            elif p == "attribution": summary["attribution"] = part_attribution(out, cfg, logf)
            elif p == "hyper":       summary["hyper"] = part_hyper(out, cfg, logf)
            elif p == "subperiod":   summary["subperiod"] = part_subperiod(out, cfg, logf)
            elif p == "placebo":     summary["placebo"] = part_placebo(out, cfg, logf)
            elif p == "seeds":       summary["seeds"] = part_seeds(out, cfg, logf)
            elif p == "survivor":    summary["survivor"] = part_survivor(out, cfg, logf)
            elif p == "sim":         summary["sim"] = part_sim(out, cfg, logf)
            elif p == "report":      part_report(out, cfg, logf, summary)
            log(f"--- part '{p}' finished in {(time.time()-t0)/60:.1f} min", logf)
        except Exception as e:
            import traceback
            log(f"!!! part '{p}' FAILED: {e}", logf)
            log(traceback.format_exc(), logf)
    log(f"=== total {(time.time()-T0)/60:.1f} min. Outputs in {out} ===", logf)


if __name__ == "__main__":
    main()