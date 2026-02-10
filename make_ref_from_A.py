# make_ref_from_A.py
import os, csv, numpy as np
from benchmark import load_field, grid_like, normalize, core_profile

A = "figures/K16_optimized"

Sx, Sy, Sz = load_field(A)
x,y,z,_,_,_ = grid_like(Sx)
bx,by,bz = normalize(Sx,Sy,Sz)
r_up, r_dn = core_profile(bx,by,bz,x,y,z)

os.makedirs("data", exist_ok=True)
with open("data/ref_profile.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["z","r_up_ref","r_down_ref"])
    w.writeheader()
    for zi, rup, rdn in zip(z, r_up, r_dn):
        w.writerow({"z": float(zi),
                    "r_up_ref": float(np.nan if np.isnan(rup) else rup),
                    "r_down_ref": float(np.nan if np.isnan(rdn) else rdn)})
print("[OK] Saved data/ref_profile.csv")
