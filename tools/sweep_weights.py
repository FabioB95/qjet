#!/usr/bin/env python3
"""
Grid sweep over (lambda_kink, mu, gamma_clean) + benchmarking.

Example:
  python tools/sweep_weights.py \
    --grid "lambda=[1e-4,1e-3,1e-2,1e-1];mu=[0,1e-3,1e-2,1e-1];gamma=[8,15,30]" \
    --ref src/data/ref_profile.csv \
    --base_out src/figures/SENS_GRID
"""
import argparse
import os
import csv
import itertools
import subprocess
import shlex
import re

def parse_grid(spec: str):
    """
    Parse 'lambda=[1e-4,1e-3];mu=[0,1e-3,1e-2];gamma=[8,15,30]'
    -> {'lambda': [...], 'mu': [...], 'gamma': [...]}
    """
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        arr = v.strip().strip("[]")
        vals = [x.strip() for x in arr.split(",")]
        casted = []
        for x in vals:
            if re.match(r"^[0-9]+$", x):
                casted.append(int(x))
            else:
                casted.append(float(x))
        out[k] = casted
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="e.g., 'lambda=[1e-4,1e-3];mu=[0,1e-3,1e-2];gamma=[8,15,30]'")
    ap.add_argument("--ref", default="src/data/ref_profile.csv", help="Reference CSV for MAE/bias comparison")
    ap.add_argument("--base_out", default="src/figures/SENS_GRID", help="Folder for all runs")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--step", type=float, default=0.2)
    ap.add_argument("--model_dir", default="src/figures/K16_optimized")
    args = ap.parse_args()

    grid = parse_grid(args.grid)
    lambdas = grid.get("lambda", [1e-2])
    mus      = grid.get("mu", [1e-2])
    gammas   = grid.get("gamma", [15])

    os.makedirs(args.base_out, exist_ok=True)
    results_csv = os.path.join(args.base_out, "results.csv")

    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lambda","mu","gamma","mae_up","mae_down","bias_up","bias_down","outdir"]
        )
        writer.writeheader()

        for lam, mu, gam in itertools.product(lambdas, mus, gammas):
            outdir = os.path.join(args.base_out, f"l{lam:g}_m{mu:g}_g{gam}")
            # 1) Optimize field
            cmd1 = (
                f"python tools/optimize_field_cli.py "
                f"--lambda_kink {lam} --mu {mu} --gamma_clean {gam} "
                f"--iters {args.iters} --step {args.step} "
                f"--model_dir {args.model_dir} --outdir {outdir}"
            )
            print(">>", cmd1)
            subprocess.run(shlex.split(cmd1), check=True)

            # 2) Benchmark against reference
            metrics_csv = os.path.join(outdir, "metrics.csv")
            cmd2 = (
                f"python src/external_benchmark.py "
                f"--model_dir {outdir} --ref_csv {args.ref} --out {metrics_csv}"
            )
            print(">>", cmd2)
            subprocess.run(shlex.split(cmd2), check=True)

            # 3) Append to global CSV
            up = {"MAE": None, "bias": None}
            dn = {"MAE": None, "bias": None}
            with open(metrics_csv, newline="") as mf:
                r = csv.DictReader(mf)
                for row in r:
                    if row["side"].lower().startswith("up"):
                        up = row
                    else:
                        dn = row
            writer.writerow({
                "lambda": lam, "mu": mu, "gamma": gam,
                "mae_up": float(up["MAE"]), "mae_down": float(dn["MAE"]),
                "bias_up": float(up["bias"]), "bias_down": float(dn["bias"]),
                "outdir": outdir
            })
            f.flush()

    print(f"[OK] Wrote {results_csv}")

if __name__ == "__main__":
    main()
