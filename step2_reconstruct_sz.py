import os
import numpy as np
import matplotlib.pyplot as plt

# ---------- PATH ----------
OPT_DIR = "figures/K16_optimized"
SXF = os.path.join(OPT_DIR, "Sx_opt.npy")
SYF = os.path.join(OPT_DIR, "Sy_opt.npy")
os.makedirs("figures", exist_ok=True)

# ---------- LOAD ----------
Sx = np.load(SXF)
Sy = np.load(SYF)

# reshape sicuro
if Sx.ndim == 1:
    nx, ny, nz = 100, 100, 136
    Sx = Sx.reshape(nx, ny, nz)
    Sy = Sy.reshape(nx, ny, nz)
else:
    nx, ny, nz = Sx.shape

# ---------- GRIGLIA (coerente col visualizer) ----------
x = np.linspace(-2, 2, nx)
y = np.linspace(-2, 2, ny)
z = np.linspace(-4, 4, nz)
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# ---------- DERIVATE ----------
dSx_dx = np.gradient(Sx, dx, axis=0)
dSy_dy = np.gradient(Sy, dy, axis=1)

# RHS dell’equazione dSz/dz = -(dSx/dx + dSy/dy)
rhs = -(dSx_dx + dSy_dy)

# ---------- INTEGRAZIONE in z con condizione al midplane ----------
# Scegli z_mid come piano equatoriale (indice più vicino a z=0)
k0 = int(np.argmin(np.abs(z)))
Sz = np.zeros_like(Sx, dtype=np.float32)  # Sz(x,y,z0) = 0

# Integra sopra il midplane (z > z0)
for k in range(k0 + 1, nz):
    # integrazione semplice (Euler avanti); robusto e sufficiente qui
    Sz[:, :, k] = Sz[:, :, k - 1] + rhs[:, :, k - 1] * dz

# Integra sotto il midplane (z < z0)
for k in range(k0 - 1, -1, -1):
    Sz[:, :, k] = Sz[:, :, k + 1] - rhs[:, :, k + 1] * dz

# ---------- STATS ----------
def qstats(A):
    B = A[np.isfinite(A)]
    qs = np.quantile(B, [0, .01, .1, .5, .9, .99, 1])
    return dict(min=float(B.min()), max=float(B.max()), quantiles=[float(v) for v in qs])

print("== RECON Sz stats ==")
print(qstats(Sz))

# ---------- CHECK divergenza (dSx/dx + dSy/dy + dSz/dz) ----------
dSz_dz = np.gradient(Sz, dz, axis=2)
divB = dSx_dx + dSy_dy + dSz_dz
core_mask = (np.sqrt(X**2 + Y**2) < 0.7)
div_mean = np.mean(np.abs(divB[core_mask]))
print(f"[check] <|divB|>_core = {div_mean:.4e}  (più piccolo è, meglio è)")

# ---------- SALVA Sz ricostruito ----------
OUTF = os.path.join(OPT_DIR, "Sz_opt_recon.npy")
np.save(OUTF, Sz)
print(f"[OK] Saved {OUTF}")

# ---------- PLOT DIAGNOSTICA 2D ----------
# 1) Frecce nel piano equatoriale (z≈0), mostrando (Sx,Sy) e heatmap di Sz
k = k0
plt.figure(figsize=(7,6))
plt.title("Midplane (z≈0): quiver (Sx,Sy), heat Sz_recon")
step = max(1, nx//25)
plt.imshow(Sz[:, :, k].T, origin="lower",
           extent=[x.min(), x.max(), y.min(), y.max()],
           cmap="coolwarm", alpha=0.85, aspect="equal")
plt.colorbar(label="Sz_recon")
plt.quiver(x[::step], y[::step],
           Sx[::step, ::step, k].T, Sy[::step, ::step, k].T,
           scale=40, width=0.003, color="k", alpha=0.9)
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout()
plt.savefig("figures/bh_pure_recon_midplane.png", dpi=220)
print("[OK] Saved figures/bh_pure_recon_midplane.png")

# 2) Mappa |divB| su sezione y=0
j = ny//2
plt.figure(figsize=(7,6))
plt.title("|divB| on y=0 plane (diagnostic)")
plt.imshow(np.abs(divB[:, j, :]).T, origin="lower",
           extent=[x.min(), x.max(), z.min(), z.max()],
           cmap="magma", aspect="auto")
plt.colorbar(label="|divB|")
plt.xlabel("x"); plt.ylabel("z")
plt.tight_layout()
plt.savefig("figures/bh_pure_recon_divB.png", dpi=220)
print("[OK] Saved figures/bh_pure_recon_divB.png")

print("\nDONE. Usa questo Sz_opt_recon.npy per la visualizzazione 'pura' dei jet e per il benchmark.")
