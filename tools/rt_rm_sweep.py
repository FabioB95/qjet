"""
RM sweep: generate a 3x3 EVPA panel at different external RMs.
Usage (one line):
  python -m src.tools.rt_rm_sweep --src M87 --field src/data/M87/optimized_field.npz --freq 230e9 --incl 17 --bmaj_mas 1.116 --bmin_mas 0.471 --bpa_deg -7.6 --pixscale_mas 0.10 --dl 1.0 --rm_min -50000 --rm_max 50000 --rm_n 9 --outdir src/figures
"""

import os, argparse, numpy as np, matplotlib.pyplot as plt
from src.rt.minsynch import (
    SynchParams, stokes_local, ray_integrate,
    apply_external_RM, evpa_from_QU, convolve_elliptical_gaussian
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--freq", type=float, required=True)
    ap.add_argument("--incl", type=float, default=17.0)
    ap.add_argument("--bmaj_mas", type=float, required=True)
    ap.add_argument("--bmin_mas", type=float, required=True)
    ap.add_argument("--bpa_deg", type=float, required=True)
    ap.add_argument("--pixscale_mas", type=float, required=True)
    ap.add_argument("--dl", type=float, default=1.0)
    ap.add_argument("--rm_min", type=float, default=-5e4)
    ap.add_argument("--rm_max", type=float, default=+5e4)
    ap.add_argument("--rm_n", type=int, default=9)
    ap.add_argument("--outdir", default="src/figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = np.load(args.field)
    Bx, By, Bz = data["Bx"], data["By"], data["Bz"]
    print(f"[RM-sweep] Field shape: {Bx.shape}")

    params = SynchParams(incl_deg=args.incl, dl=args.dl)
    jI, jQ, jU = stokes_local(Bx, By, Bz, params)
    I, Q, U = ray_integrate(jI, jQ, jU, dl=params.dl, axis=1)

    lam = 3e8 / args.freq
    mas2pix = 1.0 / args.pixscale_mas
    bmaj_pix = args.bmaj_mas * mas2pix
    bmin_pix = args.bmin_mas * mas2pix
    bpa_rad  = np.deg2rad(args.bpa_deg)

    rms = np.linspace(args.rm_min, args.rm_max, args.rm_n)
    n = len(rms)
    nrows = nrows_guess = int(np.ceil(np.sqrt(n)))
    ncols = int(np.ceil(n / nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 3.2*nrows), squeeze=False)
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            if k >= n:
                ax.axis("off")
                continue
            RM = rms[k]
            Qr, Ur = apply_external_RM(Q, U, lam, RM)
            I_b = convolve_elliptical_gaussian(I, bmaj_pix, bmin_pix, bpa_rad)
            Q_b = convolve_elliptical_gaussian(Qr, bmaj_pix, bmin_pix, bpa_rad)
            U_b = convolve_elliptical_gaussian(Ur, bmaj_pix, bmin_pix, bpa_rad)
            chi_deg = np.degrees(evpa_from_QU(Q_b, U_b))

            im = ax.imshow(chi_deg, origin="lower", vmin=-90, vmax=90, cmap="twilight")
            ax.set_title(f"RM = {RM:.0f} rad m$^{{-2}}$")
            ax.set_xticks([]); ax.set_yticks([])
            k += 1

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("EVPA (deg)")
    fig.suptitle(f"{args.src} — EVPA maps at 230 GHz (RM sweep)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = os.path.join(args.outdir, f"{args.src}_EVPA_RM_sweep.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[RM-sweep] Saved {out}")

if __name__ == "__main__":
    main()
