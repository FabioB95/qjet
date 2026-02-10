"""
Minimal command-line runner for the polarized synchrotron RT module.
Usage:
    python -m src.tools.rt_minimal \
        --src M87 \
        --field data/M87/optimized_field.npz \
        --freq 230e9 \
        --incl 17 \
        --rm 0 \
        --bmaj_mas 1.116 --bmin_mas 0.471 --bpa_deg -7.6 \
        --pixscale_mas 0.10 \
        --dl 1.0 \
        --outdir figures/
"""

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from src.rt.minsynch import (
    SynchParams, stokes_local, ray_integrate,
    apply_external_RM, evpa_from_QU, convolve_elliptical_gaussian
)

def main():
    p = argparse.ArgumentParser(description="Minimal polarized RT from QJET field")
    p.add_argument("--src", required=True)
    p.add_argument("--field", required=True, help="npz or npy file with Bx,By,Bz arrays")
    p.add_argument("--freq", type=float, required=True, help="frequency [Hz]")
    p.add_argument("--incl", type=float, default=17.0)
    p.add_argument("--rm", type=float, default=0.0, help="external RM [rad/m^2]")
    p.add_argument("--bmaj_mas", type=float, required=True)
    p.add_argument("--bmin_mas", type=float, required=True)
    p.add_argument("--bpa_deg", type=float, required=True)
    p.add_argument("--pixscale_mas", type=float, required=True)
    p.add_argument("--dl", type=float, default=1.0)
    p.add_argument("--outdir", default="figures/")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load magnetic field arrays
    data = np.load(args.field)
    Bx, By, Bz = data["Bx"], data["By"], data["Bz"]

    print(f"[RT] Field shape: {Bx.shape}")

    # Compute local emissivities and integrate
    params = SynchParams(incl_deg=args.incl, dl=args.dl)
    jI, jQ, jU = stokes_local(Bx, By, Bz, params)
    I, Q, U = ray_integrate(jI, jQ, jU, dl=params.dl, axis=1)

    # Apply RM rotation
    lam = 3e8 / args.freq
    Qr, Ur = apply_external_RM(Q, U, lam, args.rm)

    # Beam convolution (convert mas→pix)
    mas2pix = 1.0 / args.pixscale_mas
    I_b = convolve_elliptical_gaussian(I, args.bmaj_mas * mas2pix, args.bmin_mas * mas2pix, np.deg2rad(args.bpa_deg))
    Q_b = convolve_elliptical_gaussian(Qr, args.bmaj_mas * mas2pix, args.bmin_mas * mas2pix, np.deg2rad(args.bpa_deg))
    U_b = convolve_elliptical_gaussian(Ur, args.bmaj_mas * mas2pix, args.bmin_mas * mas2pix, np.deg2rad(args.bpa_deg))

    chi = evpa_from_QU(Q_b, U_b)
    chi_deg = np.degrees(chi)

    # Save quick plots
    plt.figure(figsize=(6, 5))
    plt.imshow(I_b, origin="lower", cmap="inferno")
    plt.title(f"{args.src} – Stokes I (beam-convolved)")
    plt.colorbar(label="I (arb.)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.src}_I.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.imshow(chi_deg, origin="lower", cmap="twilight", vmin=-90, vmax=90)
    plt.title(f"{args.src} – EVPA [deg]")
    plt.colorbar(label="EVPA (deg)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.src}_EVPA.png"), dpi=200)
    plt.close()

    np.savez(os.path.join(args.outdir, f"{args.src}_stokes.npz"), I=I_b, Q=Q_b, U=U_b, EVPA_deg=chi_deg)
    print(f"[RT] Saved results in {args.outdir}")

if __name__ == "__main__":
    main()
