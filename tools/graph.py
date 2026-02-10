import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------
# Directory figure: src/figures (NON dentro src/tools)
# ------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent          # .../src/tools
FIG_DIR = THIS_DIR.parent / "figures"               # .../src/figures
FIG_DIR.mkdir(exist_ok=True)                        # crea se non esiste

# --- dati: MAE, Bias e L0 dalla tabella 24 sorgenti ---
sources = [
    "3C 111","3C 120","3C 264","3C 273","3C 279","3C 345","3C 3453",
    "3C 380","3C 382","3C 84","4C-06.11","BL Lac","CTA 102","CYG A",
    "M87","MRK 421","MRK 501","NGC 1052","NGC 315","NGC 6251",
    "OC 270","PKS 0528-134","PKS 1510-089","PMN J0906+0905"
]

mae = np.array([
    0.2030, 67.2517, 14.8318, 83.6987, 111.3451, 107.8753, 74.5845,
    109.7715,108.3390,22.4723,103.0433,77.4278,57.2892,14.2300,
    92.9513,16.4316,75.1852,119.4485,113.7149,111.2407,
    26.5742,0.0467,68.8252,56.8690
])

bias = np.array([
    -0.0298, -67.2446, -14.7570, -83.6058, -111.2987, -107.8753, -74.5606,
    -109.7715, -108.3390, -22.4723, -102.9504, -77.4278, -56.9794, -14.0482,
    -92.9513, -16.4316, -74.9266, -119.4485, -113.7149, -111.2097,
    -26.5690, -0.0391, -68.8252, -56.5613
])

L0 = np.array([
    0.1,0.1,0.1,0.1,0.1,0.1,0.1,
    0.1,0.1,0.1,0.1,0.1,0.1,0.219,
    0.1,0.1,0.1,0.1,0.1,0.1,
    12.0,12.0,0.1,0.1
])

# ------------------------------------------------------
# Figura 1: Histogram of MAE
# ------------------------------------------------------
plt.figure()
plt.hist(mae, bins=10)
plt.xlabel("MAE (normalized width units)")
plt.ylabel("Number of sources")
plt.title("Distribution of core MAE across 24 AGN")
plt.tight_layout()
plt.savefig(FIG_DIR / "mae_hist_24src.png", dpi=300)
plt.close()

# ------------------------------------------------------
# Figura 2: MAE vs Bias scatter
# ------------------------------------------------------
plt.figure()
plt.scatter(bias, mae)
plt.xlabel("Bias (normalized units)")
plt.ylabel("MAE (normalized units)")
plt.title("Core MAE vs bias across 24 AGN")
for s, x, y in zip(sources, bias, mae):
    plt.annotate(s, (x, y), fontsize=6, xytext=(3,3), textcoords="offset points")
plt.tight_layout()
plt.savefig(FIG_DIR / "mae_vs_bias_24src.png", dpi=300)
plt.close()

# ------------------------------------------------------
# Figura 3: MAE vs L0 (asse x log)
# ------------------------------------------------------
plt.figure()
plt.scatter(L0, mae)
plt.xscale("log")
plt.xlabel(r"$L_0$ (dimensionless scale)")
plt.ylabel("MAE (normalized units)")
plt.title("Core MAE vs fitted scale $L_0$")
for s, x, y in zip(sources, L0, mae):
    plt.annotate(s, (x, y), fontsize=6, xytext=(3,3), textcoords="offset points")
plt.tight_layout()
plt.savefig(FIG_DIR / "mae_vs_L0_24src.png", dpi=300)
plt.close()

print(f"[OK] Saved figures to: {FIG_DIR}")
