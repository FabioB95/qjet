# multi_benchmark.py — esegue external_benchmark su più sorgenti e crea una tabella riassuntiva
import os, csv, argparse, subprocess, sys

ap=argparse.ArgumentParser()
ap.add_argument("--sources", nargs="+", required=True,
                help="nomi sorgenti (cercano data/<name>/ref_profile.csv)")
ap.add_argument("--modelA", default="figures/K16_optimized")
ap.add_argument("--out", default="figures/benchmark_summary.csv")
args=ap.parse_args()

rows=[]
for name in args.sources:
    ref = os.path.join("data", name, "ref_profile.csv")
    if not os.path.exists(ref):
        print(f"[WARN] manca {ref}, salto {name}")
        continue
    outcsv = os.path.join("figures", f"external_{name}.csv")
    cmd = [sys.executable, "external_benchmark.py", "--A", args.modelA, "--ref", ref, "--out", outcsv]
    print("[RUN]", " ".join(cmd)); subprocess.run(cmd, check=True)
    with open(outcsv,"r") as f:
        dat = list(csv.DictReader(f))
        up = dat[0]; dn = dat[1]
        rows.append({"source":name, "side":"up", **up})
        rows.append({"source":name, "side":"down", **dn})

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out,"w",newline="") as f:
    if rows:
        fields = list(rows[0].keys())
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)
print(f"[OK] Saved {args.out}")

