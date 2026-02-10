import csv, numpy as np
import matplotlib.pyplot as plt
from benchmark import load_field, grid_like, normalize, core_profile

A = "figures/K16_optimized"
REF = "data/M87/ref_profile.csv"  # <-- percorso corretto
OUT = "figures/external_overlay.png"

# Carica modello
Sx, Sy, Sz = load_field(A)
x, y, z_model, _, _, _ = grid_like(Sx)
bx, by, bz = normalize(Sx, Sy, Sz)
r_up_model, r_dn_model = core_profile(bx, by, bz, x, y, z_model)

# Carica riferimento
z_ref, r_up_ref, r_dn_ref = [], [], []
with open(REF, "r") as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        z_ref.append(float(row["z"]))
        r_up_ref.append(float(row["r_up_ref"]))
        r_dn_ref.append(float(row["r_down_ref"]))
z_ref = np.array(z_ref)
r_up_ref = np.array(r_up_ref)
r_dn_ref = np.array(r_dn_ref)

# Calcola metriche per annotazione
mask = (~np.isnan(r_up_model)) & (~np.isnan(r_up_ref))
if np.any(mask):
    mae = np.mean(np.abs(r_up_model[mask] - r_up_ref[mask]))
    bias = np.mean(r_up_model[mask] - r_up_ref[mask])
else:
    mae, bias = np.nan, np.nan

# Plot
plt.figure(figsize=(8, 5))
plt.plot(z_model, r_up_model, color='tab:blue', lw=2, label="Model (up)")
plt.plot(z_model, r_dn_model, color='tab:orange', lw=2, label="Model (down)")
plt.plot(z_ref, r_up_ref, '--', color='tab:green', lw=2, label="Observations (up)")
plt.plot(z_ref, r_dn_ref, '--', color='tab:red', lw=2, label="Observations (down)")

# Zoom sulla regione fisicamente rilevante
plt.xlim(1.5, 4.1)
plt.ylim(0, max(np.nanmax(r_up_ref), np.nanmax(r_up_model)) * 1.1)

plt.xlabel(r"$z$ (model units)", fontsize=12)
plt.ylabel(r"$r(z)$ (model units)", fontsize=12)
plt.title("M87 Jet: Model vs Observations", fontsize=13, pad=15)

# Annotazione con metriche
if not np.isnan(mae):
    plt.text(0.03, 0.96, f"MAE = {mae:.3f}\nBias = {bias:.3f}",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(OUT, dpi=200)
plt.close()
print(f"[OK] Saved {OUT}")