#!/usr/bin/env python3
"""
Estrarre dalla benchmark_summary solo le righe 'up'
e salvarle in un CSV pronto per la tabella del paper.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # project root
LATEX_IMG_ROOT = ROOT / "src" / "LATEX_IMG"
SUMMARY = LATEX_IMG_ROOT / "benchmark_summary.csv"
OUT = LATEX_IMG_ROOT / "paper_table.csv"

def main():
    rows_up = []

    with SUMMARY.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["side"] != "up":
                continue
            rows_up.append({
                "source": row["source"],
                "MAE": float(row["MAE"]),
                "bias": float(row["bias"]),
                "L0": float(row["L0"]),
                "Ncuts": int(float(row["n_pts"])),
                "coverage": float(row["coverage_frac"]),
            })

    # Ordina per sorgente (facoltativo)
    rows_up.sort(key=lambda r: r["source"])

    with OUT.open("w", newline="") as f:
        fieldnames = ["source", "MAE", "bias", "L0", "Ncuts", "coverage"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_up:
            w.writerow(r)

    print(f"[OK] Saved paper table CSV to: {OUT}")

if __name__ == "__main__":
    main()
