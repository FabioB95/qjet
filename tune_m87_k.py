# tune_m87_k.py — ottimizza k del profilo M87* rispetto al tuo modello
import csv, numpy as np
from benchmark import load_field, grid_like, normalize, core_profile

A = "figures/K16_optimized"
a = 0.56        # lascia questo per ora
z0_flat = 0.8   # idem

# 1) modello
Sx,Sy,Sz = load_field(A)
x,y,z,_,_,_ = grid_like(Sx)
bx,by,bz = normalize(Sx,Sy,Sz)
r_up, r_dn = core_profile(bx,by,bz,x,y,z)

# 2) costruiamo r_ref(z; k)
def r_ref(z, k):
    zz = np.abs(z)
    r = np.where(zz < z0_flat, k, k * (zz/z0_flat)**a)
    return r

# 3) trovo k che minimizza MAE medio (up & down) su |z|>=1 (evita disco)
mask = np.abs(z) >= 1.0
def mae_for(k):
    ref = r_ref(z, k)
    e_up = np.nanmean(np.abs(r_up[mask] - ref[mask]))
    e_dn = np.nanmean(np.abs(r_dn[mask] - ref[mask]))
    return 0.5*(e_up + e_dn)

ks = np.linspace(0.12, 0.30, 200)  # sweep
errs = np.array([mae_for(k) for k in ks])
k_best = ks[np.argmin(errs)]
print(f"[TUNE] a={a}, z0_flat={z0_flat} ⇒ k_best ≈ {k_best:.4f}, MAE ≈ {errs.min():.4f}")

# 4) scrivo il ref_profile.csv con k_best
import os
os.makedirs("data", exist_ok=True)
with open("data/ref_profile.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["z","r_up_ref","r_down_ref"])
    w.writeheader()
    ref = r_ref(z, k_best)
    for zi, ri in zip(z, ref):
        w.writerow({"z": float(zi), "r_up_ref": float(ri), "r_down_ref": float(ri)})
print("[OK] Saved data/ref_profile.csv")
