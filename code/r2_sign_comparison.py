"""R2, reviewer comment 2: which empirical sign does the predicted sign of
delta_D - delta_F correspond to on real data?

Reads the saved 191-universe results (results_revision/R1_universes_raw.csv,
produced by r1_universes.py) and compares two predictors of the sign of
delta_D - delta_F with two realised signs:

  predictors (pre-evaluation data only)
    sim    sign(pred_delta_D - pred_delta_F) from the calibrated simulation
    LamT   +1 if Lambda_T < 1, else -1
  realised signs (evaluation period)
    vs_LW  sign(delta* - mean Ledoit-Wolf intensity)       [reported in R1, Section 6.7]
    vs_Fep sign(delta* - ex-post Frobenius-optimal constant intensity)
           [the more directly corresponding measure suggested by the reviewer]

No new simulation or portfolio computation is done; the script only re-tabulates
saved outputs. NumPy and pandas only.

    python3 r2_sign_comparison.py      -> results_r2/R2_sign_comparison.csv
                                          results_r2/R2_sign_comparison_ranges.csv
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "results_revision", "R1_universes_raw.csv")
OUT = os.path.join(ROOT, "results_r2")


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(SRC)
    pred = {"sim": np.sign(d.pred_delta_D - d.pred_delta_F),
            "LamT": pd.Series(np.where(d.Lambda_T < 1, 1.0, -1.0), index=d.index)}
    real = {"vs_LW": np.sign(d.delta_star - d.mean_delta_LW),
            "vs_Fep": np.sign(d.delta_star - d.delta_F_expost)}
    rows = []
    for gname, mask in [("all", d.kind.notna()), ("named", d.kind == "named"),
                        ("random", d.kind == "random")]:
        for rn, rs in real.items():
            rs_g = rs[mask]
            row = dict(group=gname, n=int(mask.sum()), realised=rn,
                       share_realised_positive=float((rs_g > 0).mean()),
                       n_realised_zero=int((rs_g == 0).sum()))
            for pn, ps in pred.items():
                row[f"share_pred_{pn}_positive"] = float((ps[mask] > 0).mean())
                row[f"agreement_{pn}"] = float((ps[mask] == rs_g).mean())
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "R2_sign_comparison.csv"), index=False)

    ranges = pd.DataFrame({
        "quantity": ["delta_star", "mean_delta_LW", "delta_F_expost", "pred_delta_F",
                     "delta_F_expost_minus_delta_star", "delta_F_expost_over_mean_delta_LW"],
        "min": [d.delta_star.min(), d.mean_delta_LW.min(), d.delta_F_expost.min(), d.pred_delta_F.min(),
                (d.delta_F_expost - d.delta_star).min(), (d.delta_F_expost / d.mean_delta_LW).min()],
        "median": [d.delta_star.median(), d.mean_delta_LW.median(), d.delta_F_expost.median(),
                   d.pred_delta_F.median(), (d.delta_F_expost - d.delta_star).median(),
                   (d.delta_F_expost / d.mean_delta_LW).median()],
        "max": [d.delta_star.max(), d.mean_delta_LW.max(), d.delta_F_expost.max(), d.pred_delta_F.max(),
                (d.delta_F_expost - d.delta_star).max(), (d.delta_F_expost / d.mean_delta_LW).max()],
    })
    ranges.to_csv(os.path.join(OUT, "R2_sign_comparison_ranges.csv"), index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(ranges.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
