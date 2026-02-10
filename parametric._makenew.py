# make_parametric_ref.py
import os, csv, numpy as np

# profilo parabolico: r(z) = r0 * (|z|/z0)^a per |z|>=z0, altrimenti ~r0
r0 = 0.18   # raggio tipico del core vicino al disco (dalle tue misure ~0.18)
z0 = 0.8
a  = 0.50   # parabolico (tipico collimation index 0.4–0.6)

z = np.linspace(-4, 4, 136)
r = np.where(np.abs(z) < z0, r0, r0 * (np.abs(z)/z0)**a)

os.makedirs("data", exist_ok=True)
with open("data/ref_profile.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["z","r_up_ref","r_down_ref"])
    w.writeheader()
    for zi, ri in zip(z, r):
        w.writerow({"z": float(zi), "r_up_ref": float(ri), "r_down_ref": float(ri)})

print("[OK] Saved data/ref_profile.csv")
