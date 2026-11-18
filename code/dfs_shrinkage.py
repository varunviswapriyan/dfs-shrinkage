"""
Decision-Focused Covariance Shrinkage:
  Measuring and Closing the Estimation-Decision Gap in Minimum-Variance Portfolios

Usage:
    pip install numpy pandas scipy torch yfinance scikit-learn matplotlib
    python dfs_shrinkage.py --quick      # smoke test (~4 min)
    python dfs_shrinkage.py              # full run (~20-35 min CPU)

Outputs (results_dfs/):
    tab1_sim_summary.csv/.tex      per (N,T,dist): deltas (LW, Frobenius-oracle, decision-oracle, DFS-CV), variances, gaps
    tab2_real_<universe>.csv/.tex  main real-data results, long-short GMV
    tab3_longonly_<universe>.csv   long-only results
    tab4_ablation_<universe>.csv   DFS variants and cost-weight sensitivity
    tab5_gap_map.csv               relative estimation-decision gap over (N,T)
    tab_universe_*.csv             tickers used
    fig1..fig8 *.png
    summary.json
"""
import os, json, math, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf, OAS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# =============================================================================
# 0. CONFIG
# =============================================================================
CFG = dict(
    seed=42, out="results_dfs",
    start="2004-12-01", end="2024-12-31",
    L=250, h=21,                               # estimation window, holding/rebalance period (days)
    costs=[0.0010, 0.0025, 0.0050],            # proportional transaction costs
    primary_cost=0.0025,
    psi=0.4,                                   # cost weight 2/A (risk aversion A=5)
    psi_grid=[0.2, 0.4, 1.0],
    pca_k=3, ewma_lambda=0.97,
    sim=dict(N=[30, 60, 120], T=[60, 120, 250, 500], M=200, K=3, nu=5, dgrid=41, folds=5),
    dfs=dict(min_train=24, retrain_every=12, steps=300, lr=0.05, lo_iters=80),
    n_stocks=60,
    stocks=["AAPL","MSFT","AMZN","GOOGL","JPM","JNJ","XOM","PG","KO","PEP","WMT","HD","CVX","MRK","PFE",
            "BAC","WFC","C","INTC","CSCO","ORCL","IBM","T","VZ","DIS","MCD","NKE","UNH","CAT","BA","MMM",
            "GE","HON","UPS","LMT","GS","MS","AXP","BLK","ABT","AMGN","GILD","LLY","BMY","MDT","TXN","QCOM",
            "ADBE","NVDA","AMAT","COST","LOW","TGT","SBUX","MO","CL","KMB","GD","RTX","DE","EMR","ITW","UNP",
            "CSX","NSC","FDX","SO","DUK","NEE","D","AEP","ADP","TMO","DHR","SYK","BDX","ISRG","SCHW","USB",
            "PNC","MET","PRU","AFL","ALL","TRV","MMC","AON","SPG","PLD","AMT","EBAY"],
    etfs=["SPY","QQQ","IWM","EFA","EEM","TLT","IEF","LQD","HYG","GLD","DBC","VNQ"],
)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


# =============================================================================
# 1. COVARIANCE ESTIMATORS AND PORTFOLIO RULES
# =============================================================================
def gmv(Sig):
    """Global minimum-variance weights (budget constraint only)."""
    N = Sig.shape[0]; one = np.ones(N)
    try:
        u = np.linalg.solve(Sig, one)
        if not np.all(np.isfinite(u)): raise np.linalg.LinAlgError
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
    lw = LedoitWolf(assume_centered=True).fit(X)
    return float(lw.shrinkage_)


def lw_cc(X):
    """Ledoit & Wolf (2004) constant-correlation shrinkage intensity (port of covCor.m)."""
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
    if mask.any(): vals[mask] = vals[mask].mean()
    Cc = vecs @ np.diag(vals) @ vecs.T; d = np.sqrt(np.diag(Cc)); Cc = Cc / np.outer(d, d)
    return Cc * np.outer(sd, sd)


def shrink(S, F, d):
    return (1.0 - d) * S + d * F


# =============================================================================
# 2. SIMULATION ARM
# =============================================================================
def make_sigma(N, K, rng):
    B = np.zeros((N, K)); B[:, 0] = rng.normal(1.0, 0.3, N)
    for k in range(1, K): B[:, k] = rng.normal(0.0, 0.5, N)
    fvar = np.array([1.0e-4, 3.0e-5, 2.0e-5, 1.5e-5, 1.0e-5])[:K]
    idio = (rng.lognormal(np.log(0.012), 0.35, N)) ** 2
    return B @ np.diag(fvar) @ B.T + np.diag(idio)


def sample_returns(Sigma, T, dist, nu, rng):
    N = Sigma.shape[0]; Lc = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((T, N)) @ Lc.T
    if dist == "t":
        g = rng.chisquare(nu, T) / nu
        Z = Z / np.sqrt(g)[:, None] * np.sqrt((nu - 2) / nu)
    return Z


def dfs_cv_delta(X, grid, folds, rng, reps=3):
    """Decision-focused cross-validation: delta minimizing held-out realized portfolio variance."""
    T, N = X.shape; score = np.zeros(len(grid))
    for _ in range(reps):
        fid = rng.permutation(T) % folds
        for f in range(folds):
            tr, te = X[fid != f], X[fid == f]
            S_tr = tr.T @ tr / len(tr); F = target_identity(S_tr); S_te = te.T @ te / len(te)
            for i, d in enumerate(grid):
                w = gmv(shrink(S_tr, F, d)); score[i] += w @ S_te @ w
    return float(grid[int(np.argmin(score))])


def run_simulation(cfg, rng, log):
    sc = cfg["sim"]; grid = np.linspace(0, 1, sc["dgrid"]); rows = []; curves = {}
    for N in sc["N"]:
        Sigma = make_sigma(N, sc["K"], rng); V_true = 1.0 / (np.ones(N) @ np.linalg.solve(Sigma, np.ones(N)))
        for T in sc["T"]:
            for dist in ["gauss", "t"]:
                Vc_acc = np.zeros(len(grid)); Fc_acc = np.zeros(len(grid)); t0 = time.time()
                for m in range(sc["M"]):
                    X = sample_returns(Sigma, T, dist, sc["nu"], rng); X = X - X.mean(0)
                    S = X.T @ X / T; F = target_identity(S)
                    Vd = np.array([(lambda w: w @ Sigma @ w)(gmv(shrink(S, F, d))) for d in grid])
                    Fd = np.array([np.linalg.norm(shrink(S, F, d) - Sigma, "fro") ** 2 for d in grid])
                    Vc_acc += Vd; Fc_acc += Fd
                    d_lw = lw_identity(X); d_F = grid[int(np.argmin(Fd))]; d_D = grid[int(np.argmin(Vd))]
                    d_cv = dfs_cv_delta(X, grid, sc["folds"], rng)
                    V = lambda d: (lambda w: w @ Sigma @ w)(gmv(shrink(S, F, d)))
                    rows.append(dict(N=N, T=T, dist=dist, m=m, d_lw=d_lw, d_F=float(d_F), d_D=float(d_D), d_cv=d_cv,
                                     V_sample=Vd[0], V_lw=V(d_lw), V_F=Vd[int(np.argmin(Fd))], V_D=Vd.min(),
                                     V_cv=V(d_cv), V_true=V_true))
                curves[(N, T, dist)] = dict(grid=grid, V=Vc_acc / sc["M"], Fro=Fc_acc / sc["M"])
                log(f"  sim N={N:3d} T={T:3d} {dist:5s}  ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df["gap_lw"] = df.V_lw - df.V_D; df["gap_F"] = df.V_F - df.V_D
    df["rel_gap_lw"] = df.gap_lw / df.V_D; df["rel_gap_F"] = df.gap_F / df.V_D
    df["rel_gap_cv"] = (df.V_cv - df.V_D) / df.V_D; df["NT"] = df["N"] / df["T"]
    return df, curves


# =============================================================================
# 3. REAL-DATA ARM
# =============================================================================
def load_universe(tickers, cfg, n_keep=None):
    import yfinance as yf
    raw = yf.download(tickers, start=cfg["start"], end=cfg["end"], auto_adjust=True, progress=False, threads=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    px = px.copy(); px.index = pd.to_datetime(px.index)
    first_valid = px.apply(lambda s: s.first_valid_index())
    cutoff = pd.Timestamp(cfg["start"]) + pd.Timedelta(days=60)
    good = [t for t in tickers if t in px.columns and first_valid[t] is not None and first_valid[t] <= cutoff
            and px[t].isna().mean() < 0.02]
    if n_keep: good = good[:n_keep]
    px = px[good].ffill(limit=5).dropna()
    R = px.pct_change().dropna()
    R = R.clip(-0.5, 0.5)
    return R


def precompute(R, cfg):
    L, h, k = cfg["L"], cfg["h"], cfg["pca_k"]; A = R.values; D, N = A.shape
    starts = list(range(L, D - h + 1, h)); P = dict(S=[], FI=[], FCC=[], FPCA=[], H=[], X=[], date=[], feat=[])
    for t in starts:
        X = A[t - L:t]; X = X - X.mean(0); S = X.T @ X / L
        vals = np.linalg.eigvalsh(S / np.outer(np.sqrt(np.diag(S)), np.sqrt(np.diag(S))))
        sd = np.sqrt(np.diag(S)); corr = S / np.outer(sd, sd)
        feat = [(corr.sum() - N) / (N * (N - 1)), vals[-1] / N, np.log(vals[-1] / max(vals[0], 1e-9)),
                np.sqrt(np.diag(S)).mean() * np.sqrt(252), (vals < (1 + np.sqrt(N / L)) ** 2).mean()]
        P["S"].append(S); P["FI"].append(target_identity(S)); P["FCC"].append(target_cc(S)); P["FPCA"].append(target_pca(S, k))
        P["H"].append(A[t:t + h]); P["X"].append(X); P["date"].append(R.index[t]); P["feat"].append(feat)
    for key in ["S", "FI", "FCC", "FPCA", "H", "feat"]: P[key] = np.array(P[key])
    P["N"], P["R"], P["dates"] = N, len(starts), pd.DatetimeIndex(P["date"])
    return P


def baseline_weights(P, cfg, long_only=False):
    R, N, L = P["R"], P["N"], cfg["L"]; solver = gmv_long_only if long_only else gmv
    W = {m: np.zeros((R, N)) for m in ["sample", "lw_identity", "lw_cc", "oas", "ewma", "pca", "mp_clip", "equal"]}
    deltas = dict(lw_identity=np.zeros(R), lw_cc=np.zeros(R))
    for r in range(R):
        S, X = P["S"][r], P["X"][r]
        dI = lw_identity(X); dC = lw_cc(X); deltas["lw_identity"][r] = dI; deltas["lw_cc"][r] = dC
        W["sample"][r] = solver(S)
        W["lw_identity"][r] = solver(shrink(S, P["FI"][r], dI))
        W["lw_cc"][r] = solver(shrink(S, P["FCC"][r], dC))
        W["oas"][r] = solver(oas_cov(X))
        W["ewma"][r] = solver(ewma_cov(X, cfg["ewma_lambda"]))
        W["pca"][r] = solver(P["FPCA"][r])
        W["mp_clip"][r] = solver(mp_clip_cov(S, L))
        W["equal"][r] = np.ones(N) / N
    return W, deltas


# ---- torch layers ----
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
        if self.variant == "cc": F = FCC
        elif self.variant == "mix":
            pi = torch.softmax(self.mix, 0); F = pi[0] * FI + pi[1] * FCC + pi[2] * FPCA
        else: F = FI
        return (1 - d) * S + d * F


def train_dfs(P, idx, variant, cost, psi, cfg, long_only=False):
    S, FI, FCC, FPCA = (torch.as_tensor(P[k][idx]) for k in ["S", "FI", "FCC", "FPCA"])
    H = torch.as_tensor(P["H"][idx]); feat = torch.as_tensor(P["feat"][idx]); h = cfg["h"]
    model = DFSModel(variant); model.fmu = feat.mean(0); model.fsd = feat.std(0).clamp(min=1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["dfs"]["lr"])
    steps = cfg["dfs"]["steps"] // (2 if long_only else 1)
    for _ in range(steps):
        Sig = model.cov(S, FI, FCC, FPCA, feat)
        w = gmv_lo_t(Sig, cfg["dfs"]["lo_iters"]) if long_only else gmv_t(Sig)
        port = torch.einsum("rhn,rn->rh", H, w)
        var_ann = port.pow(2).mean() * 252
        to_ann = (w[1:] - w[:-1]).abs().sum(1).mean() * (252.0 / h)
        loss = var_ann + psi * cost * to_ann
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def apply_dfs(model, P, r, cfg, long_only=False):
    S, FI, FCC, FPCA = (torch.as_tensor(P[k][r:r + 1]) for k in ["S", "FI", "FCC", "FPCA"])
    feat = torch.as_tensor(P["feat"][r:r + 1])
    Sig = model.cov(S, FI, FCC, FPCA, feat)
    w = gmv_lo_t(Sig, 400) if long_only else gmv_t(Sig)
    return w[0].numpy(), float(model.delta(feat)[0])


def walk_forward_dfs(P, variant, cost, psi, cfg, log, long_only=False, tag=""):
    R, N = P["R"], P["N"]; mt, re = cfg["dfs"]["min_train"], cfg["dfs"]["retrain_every"]
    W = np.zeros((R, N)); dl = np.full(R, np.nan); model = None; t0 = time.time()
    for r in range(mt, R):
        if (r - mt) % re == 0:
            model = train_dfs(P, np.arange(0, r), variant, cost, psi, cfg, long_only)
        W[r], dl[r] = apply_dfs(model, P, r, cfg, long_only)
    log(f"  DFS {tag:22s} done ({time.time()-t0:.0f}s)")
    return W, dl


# ---- evaluation ----
def nw_tstat(d, lag=3):
    d = np.asarray(d); n = len(d); dbar = d.mean(); e = d - dbar
    v = (e @ e) / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(dbar / np.sqrt(max(v, 1e-30) / n))


def evaluate(W, P, r0, cfg):
    H = P["H"]; h = cfg["h"]; rets, var_r, to_r, cost_hits = [], [], [], []
    for r in range(r0, P["R"]):
        pr = H[r] @ W[r]; rets.append(pr); var_r.append(pr.var())
        to = np.abs(W[r] - W[r - 1]).sum() if r > r0 else 0.0; to_r.append(to)
    rets = np.concatenate(rets); to_r = np.array(to_r); var_r = np.array(var_r)
    ann_vol = rets.std() * np.sqrt(252); ann_ret = rets.mean() * 252
    cum = np.cumprod(1 + rets); mdd = (cum / np.maximum.accumulate(cum) - 1).min()
    out = dict(ann_vol=100 * ann_vol, ann_ret=100 * ann_ret, sharpe=ann_ret / ann_vol, mdd=100 * mdd,
               turnover_ann=to_r.mean() * 252 / h, realized_var_ann=100 ** 2 * rets.var() * 252)
    for c in cfg["costs"]:
        
        net = rets.copy().reshape(-1, h); net[:, 0] -= c * to_r; net = net.ravel()
        out[f"sharpe_net_{int(c*1e4)}bp"] = net.mean() * 252 / (net.std() * np.sqrt(252))
        out[f"ret_net_{int(c*1e4)}bp"] = 100 * net.mean() * 252
    
    return out, var_r


def run_real(P, uni, cfg, log):
    r0 = cfg["dfs"]["min_train"]; k0 = cfg["primary_cost"]; psi = cfg["psi"]
    log(f"  [{uni}] N={P['N']} rebalances={P['R']} eval from {P['dates'][r0].date()} to {P['dates'][-1].date()}")
    W, deltas = baseline_weights(P, cfg)
    DL = {}
    W["dfs_identity"], DL["dfs_identity"] = walk_forward_dfs(P, "identity", 0.0, psi, cfg, log, tag="identity")
    W["dfs_cc"], DL["dfs_cc"] = walk_forward_dfs(P, "cc", 0.0, psi, cfg, log, tag="cc")
    W["dfs_mix"], DL["dfs_mix"] = walk_forward_dfs(P, "mix", 0.0, psi, cfg, log, tag="mix")
    W["dfs_nn"], DL["dfs_nn"] = walk_forward_dfs(P, "nn", 0.0, psi, cfg, log, tag="nn")
    for c in cfg["costs"]:
        W[f"dfs_tc_{int(c*1e4)}bp"], DL[f"dfs_tc_{int(c*1e4)}bp"] = walk_forward_dfs(P, "identity", c, psi, cfg, log, tag=f"tc {int(c*1e4)}bp")
    W["dfs_tc_mix_25bp"], DL["dfs_tc_mix_25bp"] = walk_forward_dfs(P, "mix", k0, psi, cfg, log, tag="tc mix 25bp")
    for ps in cfg["psi_grid"]:
        if ps == psi: continue
        W[f"dfs_tc_psi{ps}"], DL[f"dfs_tc_psi{ps}"] = walk_forward_dfs(P, "identity", k0, ps, cfg, log, tag=f"tc psi={ps}")
    rows, VR = [], {}
    for m, w in W.items():
        met, vr = evaluate(w, P, r0, cfg); VR[m] = vr; rows.append(dict(method=m, **met))
    res = pd.DataFrame(rows).set_index("method")
    res["nw_t_var_vs_lw"] = [nw_tstat(VR[m] - VR["lw_identity"]) if m != "lw_identity" else 0.0 for m in res.index]
    # long-only
    WL, _ = baseline_weights(P, cfg, long_only=True)
    WL["dfs_identity_lo"], _ = walk_forward_dfs(P, "identity", 0.0, psi, cfg, log, long_only=True, tag="identity LO")
    WL[f"dfs_tc_{int(k0*1e4)}bp_lo"], _ = walk_forward_dfs(P, "identity", k0, psi, cfg, log, long_only=True, tag="tc LO")
    rowsL = [dict(method=m, **evaluate(w, P, r0, cfg)[0]) for m, w in WL.items()]
    resL = pd.DataFrame(rowsL).set_index("method")
    DL["lw_identity"] = deltas["lw_identity"]; DL["lw_cc"] = deltas["lw_cc"]
    return res, resL, DL, W, VR


# =============================================================================
# 4. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
    cfg = json.loads(json.dumps(CFG))
    if args.quick:
        cfg["sim"].update(N=[30, 60], T=[120, 250], M=30, dgrid=21); cfg["dfs"].update(steps=80, lo_iters=40)
    out = cfg["out"]; os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "run.log"), "w")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    def tex(df, name, **kw):
        df.to_csv(os.path.join(out, name + ".csv"))
        with open(os.path.join(out, name + ".tex"), "w") as fh: fh.write(df.to_latex(**kw))
    set_seed(cfg["seed"]); rng = np.random.default_rng(cfg["seed"]); T0 = time.time()

    # ---------------- simulation ----------------
    log("Simulation arm ...")
    sim, curves = run_simulation(cfg, rng, log)
    sim.to_csv(os.path.join(out, "sim_raw.csv"), index=False)
    agg = sim.groupby(["N", "T", "dist"]).agg(NT=("NT", "first"), d_lw=("d_lw", "mean"), d_F=("d_F", "mean"), d_D=("d_D", "mean"),
                                              d_cv=("d_cv", "mean"), V_true=("V_true", "mean"), V_sample=("V_sample", "mean"),
                                              V_lw=("V_lw", "mean"), V_F=("V_F", "mean"), V_D=("V_D", "mean"), V_cv=("V_cv", "mean"),
                                              rel_gap_lw=("rel_gap_lw", "mean"), rel_gap_F=("rel_gap_F", "mean"),
                                              rel_gap_cv=("rel_gap_cv", "mean")).reset_index()
    for c in ["V_true", "V_sample", "V_lw", "V_F", "V_D", "V_cv"]: agg[c] = agg[c] * 252 * 1e4   # annualised variance in %^2
    for c in ["rel_gap_lw", "rel_gap_F", "rel_gap_cv"]: agg[c] = 100 * agg[c]
    tex(agg.round(4), "tab1_sim_summary", index=False)
    gapmap = agg[agg.dist == "gauss"].pivot(index="N", columns="T", values="rel_gap_lw").round(2)
    gapmap_t = agg[agg.dist == "t"].pivot(index="N", columns="T", values="rel_gap_lw").round(2)
    tex(gapmap, "tab5_gap_map_gauss"); tex(gapmap_t, "tab5_gap_map_t")

    # Fig 1: V(delta) and Frobenius(delta) curves
    keys = [k for k in curves if k[2] == "gauss"]; nk = len(keys)
    ncol = min(4, nk); nrow = int(math.ceil(nk / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
    for i, k in enumerate(keys):
        ax = axes[i // ncol][i % ncol]; c = curves[k]; g = c["grid"]
        ax.plot(g, c["V"] * 252 * 1e4, "b-", label="OOS portfolio variance"); ax.set_xlabel("shrinkage delta")
        ax.set_ylabel("ann. variance (%^2)", color="b"); ax2 = ax.twinx(); ax2.plot(g, c["Fro"], "r--", label="Frobenius loss")
        ax2.set_ylabel("Frobenius loss", color="r"); ax.axvline(g[np.argmin(c["V"])], color="b", ls=":"); ax.axvline(g[np.argmin(c["Fro"])], color="r", ls=":")
        ax.set_title(f"N={k[0]}, T={k[1]}")
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig1_sim_curves.png"), dpi=200); plt.close()

    # Fig 2: optimal deltas vs N/T
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, dist in enumerate(["gauss", "t"]):
        a = agg[agg.dist == dist].sort_values("NT")
        for col, lab, mk in [("d_F", "Frobenius-oracle", "s"), ("d_D", "decision-oracle", "o"), ("d_lw", "Ledoit-Wolf", "^"), ("d_cv", "DFS-CV", "d")]:
            ax[i].plot(a.NT, a[col], marker=mk, ls="-", label=lab)
        ax[i].set_xscale("log"); ax[i].set_xlabel("N / T"); ax[i].set_ylabel("shrinkage intensity"); ax[i].set_title(f"{dist} returns"); ax[i].legend(fontsize=8); ax[i].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig2_sim_deltas.png"), dpi=200); plt.close()

    # Fig 3: gap heatmap
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, (gm, ttl) in enumerate([(gapmap, "Gaussian"), (gapmap_t, "Student-t")]):
        im = ax[i].imshow(gm.values, cmap="YlOrRd"); ax[i].set_xticks(range(gm.shape[1])); ax[i].set_xticklabels(gm.columns)
        ax[i].set_yticks(range(gm.shape[0])); ax[i].set_yticklabels(gm.index); ax[i].set_xlabel("T"); ax[i].set_ylabel("N")
        for r_ in range(gm.shape[0]):
            for c_ in range(gm.shape[1]): ax[i].text(c_, r_, f"{gm.values[r_, c_]:.1f}%", ha="center", va="center", fontsize=9)
        ax[i].set_title(f"Estimation-decision gap (LW vs decision-oracle), {ttl}"); plt.colorbar(im, ax=ax[i])
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig3_gap_heatmap.png"), dpi=200); plt.close()
    log(f"  simulation done ({(time.time()-T0)/60:.1f} min)")

    # ---------------- real data ----------------
    summary = dict(quick_mode=args.quick)
    for uni, tick, nk in [("stocks", cfg["stocks"], cfg["n_stocks"]), ("etfs", cfg["etfs"], None)]:
        log(f"Real-data arm: {uni} ...")
        R = load_universe(tick, cfg, nk)
        pd.Series(R.columns).to_csv(os.path.join(out, f"tab_universe_{uni}.csv"), index=False)
        P = precompute(R, cfg)
        res, resL, DL, W, VR = run_real(P, uni, cfg, log)
        tex(res.round(4), f"tab2_real_{uni}"); tex(resL.round(4), f"tab3_longonly_{uni}")
        abl = res.loc[[m for m in res.index if m.startswith("dfs")]].round(4); tex(abl, f"tab4_ablation_{uni}")
        pd.DataFrame(DL, index=P["dates"]).to_csv(os.path.join(out, f"deltas_{uni}.csv"))
        np.savez_compressed(os.path.join(out, f"weights_{uni}.npz"), **W)
        r0 = cfg["dfs"]["min_train"]; d = P["dates"][r0:]
        # Fig 4: delta time series
        fig, ax = plt.subplots(figsize=(8, 4))
        for m in ["lw_identity", "dfs_identity", "dfs_nn", f"dfs_tc_{int(cfg['primary_cost']*1e4)}bp"]:
            ax.plot(P["dates"], DL[m], label=m)
        ax.set_ylabel("shrinkage intensity"); ax.set_title(f"Shrinkage intensity over time ({uni})"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, f"fig4_deltas_{uni}.png"), dpi=200); plt.close()
        # Fig 5: rolling realized vol
        fig, ax = plt.subplots(figsize=(8, 4))
        for m in ["sample", "lw_identity", "lw_cc", "dfs_identity", "dfs_mix", "equal"]:
            s = pd.Series(np.sqrt(VR[m] * 252) * 100, index=d).rolling(6).mean(); ax.plot(s, label=m)
        ax.set_ylabel("rolling 6-month realized vol (%)"); ax.set_title(f"Out-of-sample realized volatility ({uni})"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, f"fig5_realized_vol_{uni}.png"), dpi=200); plt.close()
        # Fig 6: net Sharpe vs cost
        fig, ax = plt.subplots(figsize=(7, 4))
        for m in ["lw_identity", "lw_cc", "dfs_identity"] + [f"dfs_tc_{int(c*1e4)}bp" for c in cfg["costs"]]:
            ax.plot([int(c * 1e4) for c in cfg["costs"]], [res.loc[m, f"sharpe_net_{int(c*1e4)}bp"] for c in cfg["costs"]], marker="o", label=m)
        ax.set_xlabel("transaction cost (bps)"); ax.set_ylabel("net Sharpe"); ax.set_title(f"Cost-adjusted Sharpe ({uni})"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, f"fig6_net_sharpe_{uni}.png"), dpi=200); plt.close()
        # Fig 7: vol vs turnover scatter
        fig, ax = plt.subplots(figsize=(7, 4))
        for m in res.index:
            ax.scatter(res.loc[m, "turnover_ann"], res.loc[m, "ann_vol"]); ax.annotate(m, (res.loc[m, "turnover_ann"], res.loc[m, "ann_vol"]), fontsize=7)
        ax.set_xlabel("annual turnover"); ax.set_ylabel("OOS annualised vol (%)"); ax.set_title(f"Variance-turnover frontier ({uni})"); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, f"fig7_frontier_{uni}.png"), dpi=200); plt.close()
        summary[uni] = dict(N=int(P["N"]), rebalances=int(P["R"]), eval_start=str(d[0].date()), eval_end=str(d[-1].date()),
                            ann_vol={m: float(res.loc[m, "ann_vol"]) for m in res.index},
                            real_gap_pct_lw_vs_dfs=float(100 * (res.loc["lw_identity", "realized_var_ann"] - res.loc["dfs_identity", "realized_var_ann"]) / res.loc["dfs_identity", "realized_var_ann"]),
                            nw_t_dfs_vs_lw=float(res.loc["dfs_identity", "nw_t_var_vs_lw"]),
                            turnover={m: float(res.loc[m, "turnover_ann"]) for m in ["lw_identity", "dfs_identity", f"dfs_tc_{int(cfg['primary_cost']*1e4)}bp"]})

    # Fig 8: sim recovery
    fig, ax = plt.subplots(figsize=(7, 4))
    a = agg[agg.dist == "gauss"].sort_values("NT")
    ax.plot(a.NT, a.rel_gap_lw, marker="o", label="Ledoit-Wolf vs decision-oracle"); ax.plot(a.NT, a.rel_gap_F, marker="s", label="Frobenius-oracle vs decision-oracle")
    ax.plot(a.NT, a.rel_gap_cv, marker="d", label="DFS-CV vs decision-oracle"); ax.set_xscale("log"); ax.set_xlabel("N / T"); ax.set_ylabel("excess OOS variance (%)")
    ax.set_title("Estimation-decision gap and DFS recovery (Gaussian)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig8_sim_recovery.png"), dpi=200); plt.close()

    summary["sim_mean_rel_gap_lw_pct"] = float(agg.rel_gap_lw.mean()); summary["sim_mean_rel_gap_cv_pct"] = float(agg.rel_gap_cv.mean())
    summary["sim_max_rel_gap_lw_pct"] = float(agg.rel_gap_lw.max()); summary["runtime_min"] = round((time.time() - T0) / 60, 1)
    summary["sim_median_rel_gap_F_pct"] = float(agg.rel_gap_F.median())
    summary["sim_rel_gap_F_by_cell"] = agg[["N", "T", "dist", "d_F", "d_D", "rel_gap_F"]].round(3).to_dict("records")
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    log(json.dumps(summary, indent=2)); log("Done. Outputs in ./results_dfs/")


if __name__ == "__main__":
    main()