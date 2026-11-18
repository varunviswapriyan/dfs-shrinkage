#!/usr/bin/env python3
"""
dfs_pipeline_oracle.py  —  run AFTER dfs_pipeline.py (same directory, same --out).

  python dfs_pipeline_oracle.py --out ../results_original --part all

Adds:
  oracle    expanding-window constant-delta estimator (the minimal honest competitor)
            + oracle gap decomposition table  -> replaces your headline table
  placebo2  three correctly-specified placebos (NN variant, phase-randomised H, iid H)
  foc       corrected numerical check of Proposition 1
  stability split-sample and expanding stability of delta*
  etftie    hetero-DGP simulation calibrated to the EMPIRICAL ETF vol dispersion
  report2   crash-free REVISION_NOTES2.md
"""
import os, sys, json, math, argparse, importlib.util, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("jrs", os.path.join(HERE, "dfs_pipeline.py"))
jrs = importlib.util.module_from_spec(spec); sys.modules["jrs"] = jrs; spec.loader.exec_module(jrs)

CFG = jrs.CFG
gmv, shrink, target_identity = jrs.gmv, jrs.shrink, jrs.target_identity


def _save(df, out, name, index=True):
    jrs.save(df, out, name, index=index)


# =============================================================================
# A. EXPANDING-WINDOW CONSTANT-DELTA ESTIMATOR + ORACLE GAP DECOMPOSITION
#    This is the simplest estimator of delta_D. If it matches DFS-I (it will),
#    say so in the paper and reposition DFS as the generalisation that handles
#    costs, target mixtures and constraints.
# =============================================================================
def part_oracle(out, cfg, logf):
    res = {}
    for uni in ["stocks", "etfs"]:
        P, c, _ = jrs.build_P(uni, cfg, out)
        r0 = c["dfs"]["min_train"]; h = c["h"]
        grid = np.linspace(0, 1, 101)
        R, N = P["R"], P["N"]

        # weights and realised holding-period variance for every (delta, rebalance)
        Wg = np.zeros((len(grid), R, N)); Vg = np.zeros((len(grid), R))
        for i, d in enumerate(grid):
            for r in range(R):
                w = gmv(shrink(P["S"][r], P["FI"][r], d))
                Wg[i, r] = w
                pr = P["H"][r] @ w
                Vg[i, r] = (pr ** 2).mean() * 252
        # turnover for every (delta, rebalance), full ticker space
        Tg = np.zeros((len(grid), R))
        for i in range(len(grid)):
            prev = None
            for r in range(R):
                wf = np.zeros(P["Nfull"]); wf[P["cols"][r]] = Wg[i, r]
                Tg[i, r] = 0.0 if prev is None else np.abs(wf - prev).sum()
                prev = wf

        built = {}
        # (i) expanding-window decision-focused grid search, no transaction cost
        for cost, nm in [(0.0, "dfc_grid"), (c["primary_cost"], "dfc_grid_tc25bp")]:
            Wd = np.zeros((R, N)); dl = np.full(R, np.nan)
            for r in range(r0, R):
                obj = Vg[:, :r].mean(1) + cfg["psi"] * cost * Tg[:, 1:r].mean(1) * (252.0 / h)
                i = int(np.argmin(obj)); dl[r] = grid[i]; Wd[r] = Wg[i, r]
            built[nm] = (Wd, dl)
        # (ii) full-sample oracle constant delta (not implementable; upper bound)
        obj = np.array([jrs.evaluate(Wg[i], P, r0, c)[0]["realized_var_ann"] for i in range(len(grid))])
        istar = int(np.argmin(obj)); dstar = grid[istar]
        built["oracle_const_delta"] = (Wg[istar], np.full(R, dstar))

        # reference methods
        W, DL = jrs.run_core(P, c, seed=cfg["seed"], variants=("identity", "mix"), tc_variants=True)
        for k, (Wd, dl) in built.items():
            W[k] = Wd; DL[k] = dl
        tab, store = jrs.tabulate(W, P, c)
        tab = tab.sort_values("realized_var_ann")
        _save(tab.round(4), out, f"tabR15_oracle_vs_estimators_{uni}")

        lw = tab.loc["lw_identity", "realized_var_ann"]
        orc = tab.loc["oracle_const_delta", "realized_var_ann"]
        rows = []
        for m in ["lw_identity", "lw_cc", "nls", "dfc_grid", "dfs_identity", "dfs_mix",
                  "dfc_grid_tc25bp", "dfs_tc_25bp", "oracle_const_delta"]:
            if m not in tab.index:
                continue
            v = tab.loc[m, "realized_var_ann"]
            rows.append(dict(method=m, mean_delta=float(np.nanmean(DL[m])) if m in DL else np.nan,
                             realized_var_ann=v, pct_vs_LWI=100 * (v - lw) / lw,
                             pct_of_oracle_gap=100 * (lw - v) / (lw - orc) if lw > orc else np.nan,
                             turnover_ann=tab.loc[m, "turnover_ann"],
                             sharpe_net_25bp=tab.loc[m, "sharpe_net_25bp"]))
        gap = pd.DataFrame(rows).set_index("method")
        _save(gap.round(4), out, f"tabR16_gap_decomposition_{uni}")

        # inference for the new estimators against LW-I
        n = len(store["lw_identity"]["rets"])
        rng = np.random.default_rng(cfg["boot"]["seed"])
        idx = jrs.stationary_bootstrap_idx(n, cfg["boot"]["block"], cfg["boot"]["B"], rng)
        irows, tmat, names = [], [], []
        for m in store:
            if m == "lw_identity":
                continue
            rv, ts = jrs.boot_test(store[m]["rets"], store["lw_identity"]["rets"],
                                   jrs.logvar_stat, cfg, idx=idx)
            nm_ = jrs.net_returns(store[m]["rets"], store[m]["to"], c["primary_cost"], h)
            nb_ = jrs.net_returns(store["lw_identity"]["rets"], store["lw_identity"]["to"],
                                  c["primary_cost"], h)
            rs, _ = jrs.boot_test(nm_, nb_, jrs.sharpe_stat, cfg, idx=idx)
            irows.append(dict(method=m, var_ratio=math.exp(rv["stat"]), t_var=rv["t"],
                              p_var_boot=rv["p_boot"], sharpe_net25_diff=rs["stat"],
                              p_sharpe_net25=rs["p_boot"]))
            tmat.append(ts); names.append(m)
        inf = pd.DataFrame(irows).set_index("method")
        inf["p_var_RomanoWolf"] = jrs.romano_wolf(inf["t_var"].values, np.array(tmat).T)
        _save(inf.sort_values("t_var").round(4), out, f"tabR17_inference_with_oracle_{uni}")

        fig, ax = plt.subplots(figsize=(8, 4))
        for m in ["lw_identity", "dfc_grid", "dfs_identity", "oracle_const_delta"]:
            if m in DL:
                ax.plot(P["dates"], DL[m], label=m)
        ax.set_ylabel("shrinkage intensity"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        ax.set_title(f"Selected intensity: matrix-loss vs decision-focused vs oracle ({uni})")
        plt.tight_layout(); plt.savefig(os.path.join(out, f"figR3_intensity_{uni}.png"), dpi=200); plt.close()

        res[uni] = dict(delta_star=float(dstar), oracle_var=float(orc), lw_var=float(lw),
                        dfs_pct_of_gap=float(gap.loc["dfs_identity", "pct_of_oracle_gap"])
                        if "dfs_identity" in gap.index else None)
        jrs.log(f"[oracle] {uni}: delta*={dstar:.3f} oracle={orc:.2f} LW={lw:.2f}", logf)
        np.savez_compressed(os.path.join(out, f"oracle_{uni}.npz"), grid=grid, Vg=Vg, Tg=Tg)
    return res


# =============================================================================
# B. CORRECTLY SPECIFIED PLACEBOS
#    (1) NN variant with shuffled targets  -> tests the CONDITIONING claim
#    (2) phase-randomised H                -> destroys cross-sectional structure
#    (3) iid resampled H across time        -> destroys the time dimension only
# =============================================================================
def _phase_randomise(H, rng):
    """Phase-randomise each asset's OOS block series, preserving marginal variance
    but destroying the covariance alignment with the estimation window."""
    out = np.empty_like(H)
    for n in range(H.shape[2]):
        x = H[:, :, n].ravel()
        F = np.fft.rfft(x)
        ph = rng.uniform(0, 2 * np.pi, len(F)); ph[0] = 0
        y = np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=len(x))
        out[:, :, n] = y.reshape(H.shape[0], H.shape[1])
    return out


def part_placebo2(out, cfg, logf):
    P, c, _ = jrs.build_P("stocks", cfg, out)
    r0 = c["dfs"]["min_train"]
    rows = []
    rng = np.random.default_rng(99)
    Horig = P["H"].copy()
    for variant in ["identity", "nn"]:
        for mode in ["real", "shuffle", "phase", "iid"]:
            for rep in range(1 if mode == "real" else 5):
                P["H"] = Horig
                shuf = (mode == "shuffle")
                if mode == "phase":
                    P["H"] = _phase_randomise(Horig, np.random.default_rng(1000 + rep))
                elif mode == "iid":
                    r2 = np.random.default_rng(2000 + rep)
                    flat = Horig.reshape(-1, Horig.shape[2])
                    P["H"] = flat[r2.integers(0, len(flat), len(flat))].reshape(Horig.shape)
                W, dl = jrs.walk_forward_dfs(P, variant, 0.0, c["psi"], c,
                                             seed=cfg["seed"] + rep, shuffle_H=shuf)
                P["H"] = Horig                        # evaluate always on REAL returns
                met, _, _, _, _, _ = jrs.evaluate(W, P, r0, c)
                rows.append(dict(variant=variant, placebo=mode, rep=rep,
                                 mean_delta=np.nanmean(dl), sd_delta=np.nanstd(dl), **met))
            jrs.log(f"[placebo2] {variant}/{mode} done", logf)
    P["H"] = Horig
    df = pd.DataFrame(rows)
    _save(df.round(4), out, "tabR18_placebo_v2", index=False)
    agg = df.groupby(["variant", "placebo"]).agg(
        var_mean=("realized_var_ann", "mean"), var_sd=("realized_var_ann", "std"),
        delta_mean=("mean_delta", "mean"), delta_disp=("sd_delta", "mean"),
        to_mean=("turnover_ann", "mean"))
    _save(agg.round(4), out, "tabR18b_placebo_v2_summary")
    jrs.log(f"[placebo2]\n{agg.round(3)}", logf)
    return agg.round(4).to_dict()


# =============================================================================
# C. CORRECTED NUMERICAL CHECK OF PROPOSITION 1
#    dV/ddelta and dFro/ddelta, each normalised, evaluated at BOTH optima.
# =============================================================================
def grad_decision(S, F, Sigma, d):
    D = F - S
    Sg = shrink(S, F, d)
    Sinv = np.linalg.pinv(Sg)
    one = np.ones(len(S))
    u = Sinv @ one; s = u.sum(); w = u / s
    g = Sinv @ (D @ w)
    dw = -(g - w * (one @ g))
    return float(2.0 * w @ Sigma @ dw)


def grad_frobenius(S, F, Sigma, d):
    D = F - S
    return float(2.0 * np.trace((shrink(S, F, d) - Sigma).T @ D))


def part_foc(out, cfg, logf):
    rng = np.random.default_rng(cfg["seed"])
    grid = np.linspace(0, 1, 201)
    rows = []
    specs = [("factor", lambda N: jrs.make_sigma_factor(N, 3, rng)),
             ("hetero", lambda N: jrs.make_sigma_hetero(N, rng)),
             ("equicorr", lambda N: jrs.make_sigma_equicorr(N, rng))]
    for nm, mk in specs:
        for N in [30, 60]:
            Sigma = mk(N)
            for T in [120, 250]:
                for m in range(25):
                    X = jrs.sample_returns(Sigma, T, "gauss", 5, rng); X = X - X.mean(0)
                    S = X.T @ X / T; F = target_identity(S)
                    V = np.array([(lambda w: w @ Sigma @ w)(gmv(shrink(S, F, d))) for d in grid])
                    Fr = np.array([np.linalg.norm(shrink(S, F, d) - Sigma, "fro") ** 2 for d in grid])
                    dD, dF = grid[int(np.argmin(V))], grid[int(np.argmin(Fr))]
                    # normalise each derivative by the scale of its own objective
                    sV, sFr = V.max() - V.min() + 1e-30, Fr.max() - Fr.min() + 1e-30
                    rows.append(dict(dgp=nm, N=N, T=T, delta_D=dD, delta_F=dF,
                                     dV_at_deltaD=grad_decision(S, F, Sigma, dD) / sV,
                                     dV_at_deltaF=grad_decision(S, F, Sigma, dF) / sV,
                                     dFro_at_deltaD=grad_frobenius(S, F, Sigma, dD) / sFr,
                                     dFro_at_deltaF=grad_frobenius(S, F, Sigma, dF) / sFr))
    df = pd.DataFrame(rows)
    _save(df.round(6), out, "tabR13b_prop1_FOC_corrected", index=False)
    agg = df.groupby("dgp")[["dV_at_deltaD", "dV_at_deltaF",
                             "dFro_at_deltaD", "dFro_at_deltaF"]].agg(["mean", "std"])
    _save(agg.round(6), out, "tabR13c_prop1_FOC_summary")
    jrs.log("[foc] |dV/dd| at the Frobenius optimum should be far from zero; "
            f"means:\n{agg.round(4)}", logf)
    return {}


# =============================================================================
# D. STABILITY OF delta*  (is the number you are estimating even stable?)
# =============================================================================
def part_stability(out, cfg, logf):
    rows = []
    for uni in ["stocks", "etfs"]:
        f = os.path.join(out, f"oracle_{uni}.npz")
        if not os.path.exists(f):
            jrs.log(f"[stability] run --part oracle first for {uni}", logf); continue
        z = np.load(f); grid, Vg = z["grid"], z["Vg"]
        R = Vg.shape[1]; r0 = CFG["dfs"]["min_train"]
        halves = [("first_half", slice(r0, r0 + (R - r0) // 2)),
                  ("second_half", slice(r0 + (R - r0) // 2, R)),
                  ("full", slice(r0, R))]
        for nm, sl in halves:
            obj = Vg[:, sl].mean(1)
            rows.append(dict(universe=uni, sample=nm, delta_star=float(grid[int(np.argmin(obj))]),
                             var_at_star=float(obj.min() * 1e4), var_at_0=float(obj[0] * 1e4)))
        # out-of-sample transfer: delta* from first half applied to second half
        i1 = int(np.argmin(Vg[:, halves[0][1]].mean(1)))
        v2 = Vg[:, halves[1][1]].mean(1)
        rows.append(dict(universe=uni, sample="first->second_transfer",
                         delta_star=float(grid[i1]), var_at_star=float(v2[i1] * 1e4),
                         var_at_0=float(v2.min() * 1e4)))
        # expanding-window path of delta*
        path = [grid[int(np.argmin(Vg[:, r0:r].mean(1)))] for r in range(r0 + 6, R)]
        pd.Series(path).to_csv(os.path.join(out, f"delta_star_path_{uni}.csv"), index=False)
        jrs.log(f"[stability] {uni}: expanding delta* ranges "
                f"{min(path):.3f}-{max(path):.3f}, final {path[-1]:.3f}", logf)
    df = pd.DataFrame(rows)
    _save(df.round(4), out, "tabR19_delta_star_stability", index=False)
    return df.round(4).to_dict("records")


# =============================================================================
# E. ETF TIE-IN: does the hetero DGP, calibrated to the REAL ETF universe,
#    predict the sign of the real-data finding?
# =============================================================================
def part_etftie(out, cfg, logf):
    px, _ = jrs.load_prices(cfg["etfs"], cfg, out, "etfs")
    R = jrs.clean_universe(px, cfg, None)
    Sig_emp = np.cov(R.values.T)
    N = Sig_emp.shape[0]
    sd = np.sqrt(np.diag(Sig_emp))
    jrs.log(f"[etftie] empirical ETF vol dispersion (max/min ann.) = "
            f"{sd.max()/sd.min():.1f}x, N={N}", logf)
    rng = np.random.default_rng(cfg["seed"])
    grid = np.linspace(0, 1, 101)
    rows = []
    for label, Sigma in [("ETF_empirical", Sig_emp),
                         ("ETF_equalised_vols",                       # counterfactual
                          (Sig_emp / np.outer(sd, sd)) * (sd.mean() ** 2))]:
        for T in [120, 250, 500]:
            dDs, dFs, gaps = [], [], []
            for m in range(400):
                X = jrs.sample_returns(Sigma, T, "gauss", 5, rng); X = X - X.mean(0)
                S = X.T @ X / T; F = target_identity(S)
                V = np.array([(lambda w: w @ Sigma @ w)(gmv(shrink(S, F, d))) for d in grid])
                Fr = np.array([np.linalg.norm(shrink(S, F, d) - Sigma, "fro") ** 2 for d in grid])
                iD, iF = int(np.argmin(V)), int(np.argmin(Fr))
                dDs.append(grid[iD]); dFs.append(grid[iF]); gaps.append((V[iF] - V[iD]) / V[iD])
            rows.append(dict(sigma=label, N=N, T=T, delta_D=np.mean(dDs), delta_F=np.mean(dFs),
                             frac_dD_gt_dF=float(np.mean(np.array(dDs) > np.array(dFs))),
                             gap_pct=100 * np.mean(gaps)))
    df = pd.DataFrame(rows)
    _save(df.round(4), out, "tabR20_etf_hetero_prediction", index=False)
    jrs.log(f"[etftie]\n{df.round(3)}", logf)
    return df.round(4).to_dict("records")


# =============================================================================
# F. CRASH-FREE REPORT
# =============================================================================
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (pd.Timestamp,)): return str(o)
    return o


def part_report2(out, cfg, logf, summary):
    def rd(n):
        p = os.path.join(out, n + ".csv")
        return pd.read_csv(p) if os.path.exists(p) else None
    L = ["# Revision notes v2 — post-patch", ""]
    for nm, title in [("tabR16_gap_decomposition_stocks", "Estimation-decision gap decomposition (stocks)"),
                      ("tabR16_gap_decomposition_etfs", "Estimation-decision gap decomposition (ETFs)"),
                      ("tabR17_inference_with_oracle_stocks", "Inference incl. oracle & grid estimators"),
                      ("tabR18b_placebo_v2_summary", "Placebos v2"),
                      ("tabR19_delta_star_stability", "Stability of delta*"),
                      ("tabR20_etf_hetero_prediction", "Hetero-DGP prediction for the ETF universe"),
                      ("tabR13c_prop1_FOC_summary", "Proposition 1 FOC check (corrected)")]:
        t = rd(nm)
        if t is not None:
            L += [f"## {title}", "", t.round(4).to_markdown(index=False), ""]
    with open(os.path.join(out, "REVISION_NOTES2.md"), "w") as fh:
        fh.write("\n".join(L))
    with open(os.path.join(out, "revision_summary2.json"), "w") as fh:
        json.dump(_jsonable(summary), fh, indent=2)
    jrs.log("[report2] wrote REVISION_NOTES2.md", logf)


PARTS = ["oracle", "placebo2", "foc", "stability", "etftie", "report2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "results_original"))
    ap.add_argument("--part", default="all")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    jrs.QUICK = a.quick
    out = os.path.expanduser(a.out); os.makedirs(out, exist_ok=True)
    logf = os.path.join(out, "revision_patch.log")
    parts = PARTS if a.part == "all" else [p.strip() for p in a.part.split(",")]
    s = {}; T0 = time.time()
    for p in parts:
        t0 = time.time()
        try:
            if p == "oracle":     s["oracle"] = part_oracle(out, CFG, logf)
            elif p == "placebo2": s["placebo2"] = part_placebo2(out, CFG, logf)
            elif p == "foc":      s["foc"] = part_foc(out, CFG, logf)
            elif p == "stability":s["stability"] = part_stability(out, CFG, logf)
            elif p == "etftie":   s["etftie"] = part_etftie(out, CFG, logf)
            elif p == "report2":  part_report2(out, CFG, logf, s)
            jrs.log(f"--- {p} finished in {(time.time()-t0)/60:.1f} min", logf)
        except Exception as e:
            import traceback; jrs.log(f"!!! {p} FAILED: {e}\n{traceback.format_exc()}", logf)
    jrs.log(f"=== patch total {(time.time()-T0)/60:.1f} min ===", logf)


if __name__ == "__main__":
    main()