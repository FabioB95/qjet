import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange

# Funzioni già esistenti nel tuo progetto
from src.external_benchmark import load_ref, _interp_to_grid
from src.benchmark import load_field, grid_like, normalize, core_profile

# -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "src" / "LATEX_IMG" / "M87" / "M87_profile.csv"
MODEL_DIR = ROOT / "src" / "figures" / "K16_optimized"
# -------------------------------------------------------


def bootstrap_L0(n_boot=200, seed=42):
    np.random.seed(seed)

    # -------------------------------------------------------
    # 1. CARICA MODELLO UNA SOLA VOLTA
    # -------------------------------------------------------
    print("[BOOT] Loading QJET model field...")
    Sx, Sy, Sz = load_field(str(MODEL_DIR))
    x, y, z_model, _, _, _ = grid_like(Sx)
    bx, by, bz = normalize(Sx, Sy, Sz)

    # -------------------------------------------------------
    # 2. COSTRUISCI UNA SOLA VOLTA il profilo modello
    # -------------------------------------------------------
    print("[BOOT] Computing model core-profile once...")
    r_up_model, r_dn_model = core_profile(
        bx, by, bz, x, y, z_model,
        seeds=8192,
        R_MIN=0.03, R_MAX=0.35,
        Z0=0.22,
        z0_list=list(np.linspace(0.02, 3.8, 64)),
        R_CLIP=0.50,
        h=0.008, nsteps=12000, vmin=1e-8,
        agg="percentile", q=35.0
    )

    # -------------------------------------------------------
    # 3. CARICA PROFILO OSSERVATO M87
    # -------------------------------------------------------
    print("[BOOT] Loading observed M87 profile...")
    z_ref, r_up_ref, r_dn_ref = load_ref(str(PROFILE))

    # Interpoliamo il profilo osservato sulla griglia del modello
    r_up_ref_on_model = _interp_to_grid(z_model, z_ref, r_up_ref)

    # -------------------------------------------------------
    # 4. BOOTSTRAP
    # -------------------------------------------------------
    print(f"[BOOT] Running {n_boot} bootstrap samples...")

    L0_values = []

    # Griglia L0 come nel benchmark
    L0_grid = np.linspace(0.06, 0.20, 400)


    for _ in trange(n_boot):
        # resampling with replacement
        idx = np.random.randint(0, len(r_up_ref_on_model), len(r_up_ref_on_model))
        r_bs = r_up_ref_on_model[idx]

        # fit L0 by scanning
        best_L0 = None
        best_MAE = 1e99

        for L0 in L0_grid:
            # SAME rule used in benchmark: map="divide"
            r_model_scaled = r_up_model / L0
            mae = np.nanmean(np.abs(r_model_scaled[idx] - r_bs))

            if mae < best_MAE:
                best_MAE = mae
                best_L0 = L0

        L0_values.append(best_L0)

    L0_values = np.array(L0_values)

    # save .npy
    out_arr = ROOT / "src" / "LATEX_IMG" / "M87" / "L0_bootstrap.npy"
    np.save(out_arr, L0_values)
    print(f"[BOOT] Saved array at: {out_arr}")

    # save plot
    out_fig = ROOT / "src" / "LATEX_IMG" / "M87" / "L0_bootstrap_M87.png"
    plt.figure(figsize=(6,4))
    plt.hist(L0_values, bins=20, color="steelblue", alpha=0.85)
    plt.xlabel("L0 bootstrap samples")
    plt.ylabel("Count")
    plt.title("Bootstrap L0 — M87")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    print(f"[BOOT] Saved figure at: {out_fig}")


if __name__ == "__main__":
    bootstrap_L0()
