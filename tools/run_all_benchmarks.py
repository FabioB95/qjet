#!/usr/bin/env python3
"""
Run external_benchmark on all sources in src/LATEX_IMG and build a summary CSV.

It assumes the following structure:

  src/
    LATEX_IMG/
      <SOURCE_NAME>/
        <SOURCE_NAME>_profile.csv   # normalized reference profile (already built by make_ref_profiles)

  src/figures/K16_optimized/
        Sx_opt.npy, Sy_opt.npy, Sz_opt_recon.npy ...

The script will create, for each source:

  src/LATEX_IMG/<SOURCE_NAME>/<SOURCE_NAME>_benchmark.csv

and a global summary:

  src/LATEX_IMG/benchmark_summary.csv
"""

import os
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # project root (.. from tools/, then .. from src/)
LATEX_IMG_ROOT = ROOT / "src" / "LATEX_IMG"
MODEL_DIR = ROOT / "src" / "figures" / "K16_optimized"
SUMMARY_CSV = LATEX_IMG_ROOT / "benchmark_summary.csv"

def main():
    print("\n================ RUNNING BENCHMARK ON ALL SOURCES ================")
    print(f"Root folder : {LATEX_IMG_ROOT}")
    print(f"Model dir   : {MODEL_DIR}")

    if not LATEX_IMG_ROOT.exists():
        print(f"[ERROR] Folder not found: {LATEX_IMG_ROOT}")
        sys.exit(1)

    if not MODEL_DIR.exists():
        print(f"[ERROR] Model dir not found: {MODEL_DIR}")
        sys.exit(1)

    # Each subfolder in LATEX_IMG is a source (must contain <name>_profile.csv)
    sources = []
    for child in sorted(LATEX_IMG_ROOT.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        profile_csv = child / f"{name}_profile.csv"
        if profile_csv.exists():
            sources.append(name)
        else:
            print(f"[WARN] skipping {name}: no {name}_profile.csv found")

    print(f"Found {len(sources)} source folders: {sources}\n")

    summary_rows = []

    for name in sources:
        print(f"[RUN] {name}")
        src_dir = LATEX_IMG_ROOT / name
        ref_csv = src_dir / f"{name}_profile.csv"
        out_csv = src_dir / f"{name}_benchmark.csv"

        # Build the command: SAME OPTIONS AS THE 3C_111 TEST
        cmd = [
            sys.executable,
            "-m", "src.external_benchmark",
            "--A", str(MODEL_DIR),
            "--ref", str(ref_csv),
            "--out", str(out_csv),
            "--fit_L0",
            "--z0", "0.0",
            "--nsteps", "60",
            "--seeds", "8",
            "--r_min", "0.0",
            "--r_max", "10.0",
        ]

        print("   CMD:", " ".join(cmd))
        # Run external_benchmark for this source
        subprocess.run(cmd, check=True)

        # Read its benchmark CSV and add to summary
        if out_csv.exists():
            with out_csv.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_with_source = {"source": name}
                    row_with_source.update(row)
                    summary_rows.append(row_with_source)
        else:
            print(f"[WARN] benchmark CSV not found for {name}: {out_csv}")

    # Write global summary
    if summary_rows:
        SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
        fields = list(summary_rows[0].keys())
        with SUMMARY_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
        print(f"\n[OK] Saved global summary: {SUMMARY_CSV}")
    else:
        print("[WARN] No benchmark rows produced; nothing to summarize.")

if __name__ == "__main__":
    main()
