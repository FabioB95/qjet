#!/usr/bin/env python3
"""
Build MAE heatmaps for (λ on x, μ on y) per γ.

Example:
  python tools/plot_sensitivity.py \
    --csv src/figures/SENS_GRID/results.csv \
    --out src/figures/SENS_GRID/heatmaps
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="results.csv from sweep_weights.py")
    ap.add_argument("--out", default=None, help="output folder for heatmaps")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.out is None:
        args.out = os.path.join(os.path.dirname(args.csv), "heatmaps")
    os.makedirs(args.out, exist_ok=True)

    gammas = sorted(df["gamma"].unique())
    for g in gammas:
        sub = df[df["gamma"] == g].copy()
        sub["mae_mean"] = 0.5 * (sub["mae_up"] + sub["mae_down"])
        lam_vals = sorted(sub["lambda"].unique())
        mu_vals  = sorted(sub["mu"].unique())

        M = np.full((len(mu_vals), len(lam_vals)), np.nan)
        for i, mu in enumerate(mu_vals):
            for j, lam in enumerate(lam_vals):
                row = sub[(sub["lambda"] == lam) & (sub["mu"] == mu)]
                if not row.empty:
                    M[i, j] = float(row["mae_mean"].values[0])

        # Raster heatmap (PNG)
        plt.figure()
        im = plt.imshow(M, origin="lower", aspect="auto",
                        extent=[0, len(lam_vals)-1, 0, len(mu_vals)-1])
        plt.colorbar(im, label="MAE (mean of up/down)")
        plt.xticks(np.arange(len(lam_vals)), [f"{v:g}" for v in lam_vals], rotation=45)
        plt.yticks(np.arange(len(mu_vals)), [f"{v:g}" for v in mu_vals])
        plt.xlabel("λ (lambda_kink)")
        plt.ylabel("μ (power well)")
        plt.title(f"Sensitivity heatmap (γ-clean={g})")
        plt.tight_layout()
        png_out = os.path.join(args.out, f"heatmap_gamma{g}.png")
        plt.savefig(png_out, dpi=150)
        plt.close()

        # Vector PDF (for the paper)
        plt.figure()
        im = plt.imshow(M, origin="lower", aspect="auto",
                        extent=[0, len(lam_vals)-1, 0, len(mu_vals)-1])
        plt.colorbar(im, label="MAE (mean of up/down)")
        plt.xticks(np.arange(len(lam_vals)), [f"{v:g}" for v in lam_vals], rotation=45)
        plt.yticks(np.arange(len(mu_vals)), [f"{v:g}" for v in mu_vals])
        plt.xlabel("λ (lambda_kink)")
        plt.ylabel("μ (power well)")
        plt.title(f"Sensitivity heatmap (γ-clean={g})")
        plt.tight_layout()
        pdf_out = os.path.join(args.out, f"heatmap_gamma{g}.pdf")
        plt.savefig(pdf_out)
        plt.close()

        print(f"[OK] Saved {png_out} and {pdf_out}")

if __name__ == "__main__":
    main()
