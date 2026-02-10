# tune_m87_grid.py — grid search su (a, z0_flat) con k ottimizzato per ciascuna coppia
import csv, numpy as np

from benchmark import load_field, grid_like, normalize, core_profile

A = "figures/K16_optimized"

# --- modello: profili r_up/r_down su griglia z ---
Sx,Sy,Sz = load_field(A)
x,y,z,_,_,_ = grid_like(Sx)
bx,by,bz = normalize(Sx,Sy,Sz)
r_up, r_dn = core_profile(bx,by,bz,x,y,z)

mask = np.abs(z) >= 1.0   # confrontiamo lontano dal disco

def r_ref(z, a, z0, k):
    zz = np.abs(z)
    f = np.where(zz < z0, 1.0, (zz/z0)**a)
    return k * f

def mae_for(a, z0, k):
    ref = r_ref(z, a, z0, k)
    e_up = np.nanmean(np.abs(r_up[mask] - ref[mask]))
    e_dn = np.nanmean(np.abs(r_dn[mask] - ref[mask]))
    return 0.5*(e_up + e_dn)

def best_k(a, z0, kmin=0.08, kmax=0.30, n=200):
    ks = np.linspace(kmin, kmax, n)
    errs = np.array([mae_for(a, z0, k) for k in ks])
    i = int(np.argmin(errs))
    return float(ks[i]), float(errs[i])

A_grid  = np.linspace(0.45, 0.65, 9)    # esponente parabolico tipico M87* ~0.5-0.6
Z0_grid = np.linspace(0.6, 1.2, 7)      # transizione piatto→parabolico

results = []
for a in A_grid:
    for z0 in Z0_grid:
        k, mae = best_k(a, z0)
        results.append((mae, a, z0, k))

# ordina per MAE crescente
results.sort(key=lambda t: t[0])

# stampa i migliori 10
print("rank, MAE, a, z0_flat, k")
for i,(mae,a,z0,k) in enumerate(results[:10], 1):
    print(f"{i:>2}, {mae:.6f}, {a:.3f}, {z0:.3f}, {k:.4f}")

# salva anche il ref migliore
best_mae, best_a, best_z0, best_k = results[0]
import os
os.makedirs("data", exist_ok=True)
with open("data/ref_profile.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["z","r_up_ref","r_down_ref"])
    w.writeheader()
    ref = r_ref(z, best_a, best_z0, best_k)
    for zi, ri in zip(z, ref):
        w.writerow({"z": float(zi), "r_up_ref": float(ri), "r_down_ref": float(ri)})

print(f"[OK] Saved data/ref_profile.csv  (best: MAE={best_mae:.6f}, a={best_a:.3f}, z0={best_z0:.3f}, k={best_k:.4f})")
