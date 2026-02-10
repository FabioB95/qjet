import os, numpy as np
import matplotlib.pyplot as plt

K = 16
in_dir  = f"figures/K{K}_optimized"
Sx = np.load("../data/spins/data3D0000_Sx.npy")
Sy = np.load("../data/spins/data3D0000_Sy.npy")
Sz = np.load("../data/spins/data3D0000_Sz.npy")
Sz_opt = np.load(os.path.join(in_dir, "Sz_opt.npy"))
dSz = np.load(os.path.join(in_dir, "Sz_delta.npy"))

# ---- reshape se conosci la griglia (altrimenti mostriamo istogrammi) ----
# Se non hai shape 3D, fai una mappa 2D “lineare” ordinata:
def quick_map(a):
    n = a.size
    s = int(np.ceil(np.sqrt(n)))
    pad = s*s - n
    aa = np.pad(a.ravel(), (0, pad), constant_values=np.nan)
    return aa.reshape(s, s)

fig, axs = plt.subplots(1,3, figsize=(15,4))
im0 = axs[0].imshow(quick_map(Sz),    origin="lower"); axs[0].set_title("Sz (original)")
im1 = axs[1].imshow(quick_map(Sz_opt),origin="lower"); axs[1].set_title("Sz (optimized)")
im2 = axs[2].imshow(quick_map(dSz),   origin="lower", cmap="bwr", vmin=-1, vmax=1); axs[2].set_title("ΔSz")
for ax in axs: ax.axis("off")
fig.colorbar(im0, ax=axs[0], shrink=0.7); fig.colorbar(im1, ax=axs[1], shrink=0.7); fig.colorbar(im2, ax=axs[2], shrink=0.7)
os.makedirs("figures", exist_ok=True)
plt.savefig(f"figures/K{K}_before_after.png", dpi=150, bbox_inches="tight")
print(f"Salvato: figures/K{K}_before_after.png")
