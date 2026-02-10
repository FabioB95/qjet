# make_m87_ref.py — profilo osservativo M87* (parabolico configurabile)
import os, csv, numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--a", type=float, default=0.56, help="indice parabolico r ~ k*|z|^a (tipico ~0.5-0.6)")
ap.add_argument("--k", type=float, default=0.18, help="scala r a |z|=1 nelle nostre unità")
ap.add_argument("--zmin", type=float, default=-4.0)
ap.add_argument("--zmax", type=float, default= 4.0)
ap.add_argument("--nz", type=int, default=136)
ap.add_argument("--z0_flat", type=float, default=0.8, help="zona interna quasi piatta (|z|<z0_flat → r≈k)")
ap.add_argument("--outfile", default="data/ref_profile.csv")
args = ap.parse_args()

z = np.linspace(args.zmin, args.zmax, args.nz)
r = np.where(np.abs(z) < args.z0_flat, args.k, args.k * (np.abs(z)/args.z0_flat)**args.a)

os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
with open(args.outfile,"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["z","r_up_ref","r_down_ref"])
    w.writeheader()
    for zi, ri in zip(z, r):
        w.writerow({"z": float(zi), "r_up_ref": float(ri), "r_down_ref": float(ri)})
print(f"[OK] Saved {args.outfile}")
print(f"[INFO] a={args.a}, k={args.k}, z0_flat={args.z0_flat}, z∈[{args.zmin},{args.zmax}], nz={args.nz}")
