# step1_pure_render.py
import os, numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt

# ---------- PATH ----------
OPT_DIR = "figures/K16_optimized"
SXF = os.path.join(OPT_DIR, "Sx_opt.npy")
SYF = os.path.join(OPT_DIR, "Sy_opt.npy")
SZF = os.path.join(OPT_DIR, "Sz_opt.npy")
os.makedirs("figures", exist_ok=True)

# ---------- LOAD ----------
Sx = np.load(SXF); Sy = np.load(SYF); Sz = np.load(SZF)

# reshape sicuro
if Sx.ndim == 1:
    nx, ny, nz = 100, 100, 136
    Sx = Sx.reshape(nx, ny, nz); Sy = Sy.reshape(nx, ny, nz); Sz = Sz.reshape(nx, ny, nz)
else:
    nx, ny, nz = Sx.shape

# grid come nel visualizer
x = np.linspace(-2, 2, nx); y = np.linspace(-2, 2, ny); z = np.linspace(-4, 4, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
R = np.sqrt(X**2 + Y**2)

# ---------- STATS ----------
def quick_stats(name, A):
    q = np.quantile(A[np.isfinite(A)], [0, .01, .1, .5, .9, .99, 1]) if np.isfinite(A).any() else None
    print(f"{name}: shape={A.shape}  min..max=({A.min():.6g},{A.max():.6g})  "
          f"zeros%={(np.isclose(A,0,atol=1e-14).mean()*100):.2f}%"
          + (f"  q=[{', '.join(f'{v:.3g}' for v in q)}]" if q is not None else ""))

print("=== PURE FIELD CHECK ===")
quick_stats("Sx", Sx); quick_stats("Sy", Sy); quick_stats("Sz", Sz)

# ---------- NORMALIZZAZIONE (PURO) ----------
B = np.sqrt(Sx**2 + Sy**2 + Sz**2) + 1e-30
bx, by, bz = Sx/B, Sy/B, Sz/B
print(f"[PURO] |bz| median={np.median(np.abs(bz)):.3e}  max={np.max(np.abs(bz)):.3e}")

# ---------- MAYAVI RENDER PURO ----------
mlab.options.offscreen = False
fig = mlab.figure(size=(1400, 1400), bgcolor=(0.02,0.02,0.08))

# disco (torus “gaussian”)
torus = np.exp(-((R - 1.0)**2) / 0.05) * np.exp(-(Z**2) / 0.10)
mlab.contour3d(X, Y, Z, torus,
               contours=[0.12, 0.25, 0.4, 0.55, 0.7],
               opacity=0.35, colormap="hot")

# buco nero (sfera scura)
mlab.points3d(0,0,0, scale_factor=0.55, color=(0.0,0.0,0.05), resolution=96)

# campo puro
src = mlab.pipeline.vector_field(bx, by, bz)

# seed: anello interno del disco (pochi = niente spaghetti)
n_seeds, radius = 12, 0.70
angles = np.linspace(0, 2*np.pi, n_seeds, endpoint=False)
for a in angles:
    x0, y0 = radius*np.cos(a), radius*np.sin(a)
    for z0, dirn in [(0.2,'forward'), (-0.2,'backward')]:
        s = mlab.pipeline.streamline(src, seedtype='point',
                                     integration_direction=dirn, colormap='autumn')
        s.seed.widget.position = (x0, y0, z0)
        s.seed.widget.enabled = False
        s.stream_tracer.maximum_propagation = 25
        mlab.pipeline.tube(s, tube_radius=0.035)

mlab.title("Black Hole Jet Field (PURE optimized field)", size=0.45, color=(0.92,0.92,1.0))
mlab.view(azimuth=40, elevation=68, distance=12, focalpoint=(0,0,0.5))

os.makedirs("figures", exist_ok=True)
mlab.savefig("figures/bh_pure.png", magnification=2)
print("[OK] Saved figures/bh_pure.png")

# ---------- SEZIONE 2D (piano equatoriale) ----------
k = nz//2  # indice piano z≈0
BX2 = bx[:,:,k]; BY2 = by[:,:,k]
BM = np.sqrt(BX2**2 + BY2**2)
plt.figure(figsize=(7,6))
plt.imshow(BM.T, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
           cmap='magma', alpha=0.85, aspect='equal')
step = max(1, nx//20)  # dirada le frecce per leggibilità
plt.quiver(x[::step], y[::step], BX2[::step,::step].T, BY2[::step,::step].T,
           scale=40, width=0.003, alpha=0.9)
plt.title("Midplane field (PURE)  z≈0")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout()
plt.savefig("figures/bh_pure_midplane.png", dpi=220)
print("[OK] Saved figures/bh_pure_midplane.png")

print("\nDONE. Questo è il riferimento 'puro'. Se i jet verticali non compaiono, è perché Sz nei dati è ~0 (diagnosi reale).")
