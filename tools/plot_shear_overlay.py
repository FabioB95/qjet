#!/usr/bin/env python3
"""
Overlay the shear annulus (R0 ± width) on a Stokes-I image for quick sanity checks.

Usage (one line):
  python -m src.tools.plot_shear_overlay --stokes src/figures/M87_stokes.npz --R0 0.75 --shear-width 0.12 --out src/figures/M87_I_shear_overlay.png
"""
import argparse, os, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Circle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stokes", required=True, help="NPZ with I,Q,U (e.g., *_stokes.npz from rt_minimal)")
    ap.add_argument("--R0", type=float, required=True, help="Shear annulus center (0=center, 1=edge)")
    ap.add_argument("--shear-width", type=float, required=True, help="Annulus half-width (fraction of image radius)")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    data = np.load(args.stokes)
    I = data["I"]

    H, W = I.shape
    cx, cy = W/2.0, H/2.0
    # Use the smaller half-size as the reference radius
    Rpix = min(W, H) / 2.0

    r0 = args.R0 * Rpix
    w  = args.shear_width * Rpix

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(I, origin="lower", cmap="inferno")
    ax.set_axis_off()
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Stokes I (arb.)")

    # Two circles: inner and outer bounds of the annulus
    for r, ls in [(r0 - w, "--"), (r0 + w, "--")]:
        circ = Circle((cx, cy), radius=max(r, 1.0), fill=False, linewidth=1.5, linestyle=ls, edgecolor="white")
        ax.add_patch(circ)

    ttl = args.title or f"Shear annulus: R0={args.R0:.2f}, width={args.shear_width:.2f}"
    ax.set_title(ttl)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {args.out}")

if __name__ == "__main__":
    main()
