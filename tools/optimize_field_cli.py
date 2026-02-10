#!/usr/bin/env python3
"""
Lightweight optimizer wrapper for sensitivity sweeps.

Usage (example):
  python -m src.tools.optimize_field_cli ^
    --lambda_kink 1e-2 --mu 1e-2 --gamma_clean 15 ^
    --model_dir src/figures/K16_optimized ^
    --outdir src/figures/SENS_l1e-2_m1e-2_g15

If your baseline arrays are flattened 1D, you can force the shape:
  --shape 100,100,136
"""
import os
import argparse
import numpy as np
from pathlib import Path

# Make sure we can import from project root when run as a script or module
try:
    from src.hamiltonian import total_hamiltonian, compute_gradients
    from src.optimizer import riemannian_projected_descent
except Exception as e:
    # fallback if user runs from src/tools folder directly
    import sys, pathlib
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.hamiltonian import total_hamiltonian, compute_gradients
    from src.optimizer import riemannian_projected_descent

import numpy as np

def _energy_scalar(H):
    """Coerce Hamiltonian return to a single float for logging/history."""
    H_arr = np.asarray(H)
    if H_arr.size == 1:
        return float(H_arr)
    # if it’s a vector (e.g., per-term energies), sum them for a total
    return float(H_arr.sum())



def helmholtz_hodge_clean(Sx, Sy, Sz, n_iter=1):
    """
    Very lightweight proxy for divergence cleaning (γ-strength).
    Smooths and renormalizes spins; acts as a tunable γ surrogate.
    """
    for _ in range(n_iter):
        for A in (Sx, Sy, Sz):
            A[:] = (np.roll(A, 1, 0) + np.roll(A, -1, 0) +
                    np.roll(A, 1, 1) + np.roll(A, -1, 1) +
                    np.roll(A, 1, 2) + np.roll(A, -1, 2) + A) / 7.0
        mag = np.sqrt(Sx**2 + Sy**2 + Sz**2)
        mag[mag == 0] = 1e-15
        Sx[:], Sy[:], Sz[:] = Sx / mag, Sy / mag, Sz / mag
    return Sx, Sy, Sz

def _parse_shape(s):
    """Parse --shape 'Nx,Ny,Nz' into a tuple of ints."""
    try:
        Nx, Ny, Nz = [int(x.strip()) for x in s.split(",")]
        assert Nx > 0 and Ny > 0 and Nz > 0
        return (Nx, Ny, Nz)
    except Exception:
        raise ValueError(f"Invalid --shape '{s}'. Use e.g. --shape 100,100,136")

def _reshape_like(arr: np.ndarray, target_shape):
    """
    Ensure arr is 3D with target_shape. If flat and sizes match, reshape;
    if 3D but different shape with same size, reshape (warn).
    """
    size_target = int(np.prod(target_shape))
    if arr.ndim == 1:
        if arr.size != size_target:
            raise ValueError(f"Array of size {arr.size} cannot be reshaped to {target_shape} (size {size_target}).")
        return arr.reshape(target_shape)
    if arr.ndim == 3:
        if arr.shape == target_shape:
            return arr
        if int(np.prod(arr.shape)) == size_target:
            # silent reshape if same total size (saves a headache)
            return arr.reshape(target_shape)
        raise ValueError(f"Array has shape {arr.shape} incompatible with target {target_shape}.")
    # Any other ndim: try flatten then reshape
    flat = arr.ravel()
    if flat.size != size_target:
        raise ValueError(f"Array with shape {arr.shape} cannot be reshaped to {target_shape}.")
    return flat.reshape(target_shape)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda_kink", type=float, default=1e-2, help="λ: twist/current penalty")
    ap.add_argument("--mu",          type=float, default=1e-2, help="μ: power-well strength")
    ap.add_argument("--gamma_clean", type=int,   default=15,   help="γ-proxy: cleaning cadence (iterations)")
    ap.add_argument("--iters",       type=int,   default=300,  help="Riemannian iterations")
    ap.add_argument("--step",        type=float, default=0.2,  help="Initial step size η")
    ap.add_argument("--model_dir",   default="src/figures/K16_optimized", help="Baseline *.npy folder (shapes/init)")
    ap.add_argument("--outdir",      default="src/figures/SENS_run",      help="Output folder for optimized spins")
    ap.add_argument("--shape",       default=None, help="Force 3D grid shape: 'Nx,Ny,Nz' (e.g., 100,100,136)")
    ap.add_argument("--alpha-ext", type=float, default=0.0)
    ap.add_argument("--k-ext", type=float, default=2.0)
    ap.add_argument("--theta0-deg", type=float, default=8.0)
    ap.add_argument("--z0-rg", type=float, default=50.0)
    # --- Shear-layer control flags ---
    ap.add_argument("--beta-shear", type=float, default=0.0, help="β: shear-layer pitch regularization strength")
    ap.add_argument("--R0", type=float, default=0.7, help="Radial center of shear annulus (0–1)")
    ap.add_argument("--shear-width", type=float, default=0.15, help="Half-width of shear annulus in normalized radius")
    ap.add_argument("--pitch-target", type=float, default=0.85, help="Target cos(pitch) in shear layer")
    ap.add_argument("--print_shapes_only", action="store_true",
                    help="Just print inferred/forced shapes and exit (debug).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load baseline field (for shape + initializer)
    try:
        Sx0 = np.load(os.path.join(args.model_dir, "Sx_opt.npy"))
        Sy0 = np.load(os.path.join(args.model_dir, "Sy_opt.npy"))
        sz_recon = Path(args.model_dir) / "Sz_opt_recon.npy"
        Sz0 = np.load(sz_recon if sz_recon.exists() else (Path(args.model_dir) / "Sz_opt.npy"))
    except Exception as e:
        raise SystemExit(f"[ERROR] Could not load baseline arrays from {args.model_dir}: {e}")

    # Decide target shape
    target_shape = None
    if args.shape:
        target_shape = _parse_shape(args.shape)
    else:
        # If any array is already 3D, use its shape
        shapes3d = [arr.shape for arr in (Sx0, Sy0, Sz0) if getattr(arr, "ndim", 1) == 3]
        if len(shapes3d) > 0:
            target_shape = shapes3d[0]
        else:
            # No 3D shapes to infer from; user must pass --shape
            raise SystemExit(
                "[ERROR] Baseline arrays appear flattened. "
                "Please re-run with --shape Nx,Ny,Nz (e.g., --shape 100,100,136)."
            )

    # Reshape all to the target 3D shape
    Sx0 = _reshape_like(Sx0, target_shape)
    Sy0 = _reshape_like(Sy0, target_shape)
    Sz0 = _reshape_like(Sz0, target_shape)

    # --- axial coordinate in r_g (assume jet axis is z -> last dimension) ---
    Nz = Sz0.shape[2]
    z_line = np.linspace(1.0, 400.0, Nz, dtype=float)  # span in r_g; adjust later if you want
    z_rg = z_line[None, None, :].repeat(Sz0.shape[0], axis=0).repeat(Sz0.shape[1], axis=1)

    if args.print_shapes_only:
        print(f"[INFO] Using target shape: {target_shape}")
        print(f" Sx0 shape: {Sx0.shape} | Sy0 shape: {Sy0.shape} | Sz0 shape: {Sz0.shape}")
        return

    # Magnitude proxy (kept fixed)
    Br = np.sqrt(Sx0**2 + Sy0**2 + Sz0**2)
    Br[Br == 0] = 1e-15

    # Unit-spin initializer
    mag0 = np.sqrt(Sx0**2 + Sy0**2 + Sz0**2); mag0[mag0 == 0] = 1e-15
    Sx, Sy, Sz = Sx0 / mag0, Sy0 / mag0, Sz0 / mag0

    # Simple axial bias profile h_i (toy; replace if you have your own)
    h_i = np.zeros_like(Sz)
    try:
        h_i[:, :, 0] = 0.5
    except Exception:
        h_i.flat[0] = 0.5

    H_hist = []
    for t in range(1, args.iters + 1):
        H, _comps = total_hamiltonian(
            Sx, Sy, Sz, Br, h_i,
            lambda_kink=args.lambda_kink,
            mu=args.mu,
            Omega_H=0.1,
            alpha_ext=args.alpha_ext,
            k_ext=args.k_ext,
            theta0_deg=args.theta0_deg,
            z0_rg=args.z0_rg,
            z_rg=z_rg,
            beta_shear=args.beta_shear,
            R0=args.R0,
            shear_width=args.shear_width,
            pitch_target=args.pitch_target
        )
        H_val = _energy_scalar(H)

        if t % 25 == 0:
            print(f"[{t:04d}] H={H_val:.6e}  (H_ext={args.alpha_ext>0.0}, H_shear={args.beta_shear>0.0})")

        dSx, dSy, dSz = compute_gradients(
            Sx, Sy, Sz, Br, h_i,
            target_power=0.1,
            mu=args.mu,
            Omega_H=0.1,
            alpha_ext=args.alpha_ext,
            k_ext=args.k_ext,
            theta0_deg=args.theta0_deg,
            z0_rg=args.z0_rg,
            z_rg=z_rg,
            beta_shear=args.beta_shear,
            R0=args.R0,
            shear_width=args.shear_width,
            pitch_target=args.pitch_target
        )

        # Riemannian step on S^2
        Sx, Sy, Sz = riemannian_projected_descent(Sx, Sy, Sz, dSx, dSy, dSz, eta=args.step)

        # γ-proxy as cleaning cadence
        if (t % max(1, args.gamma_clean)) == 0:
            Sx, Sy, Sz = helmholtz_hodge_clean(Sx, Sy, Sz, n_iter=1)

        H_hist.append(H_val)

    # Save outputs
    np.save(os.path.join(args.outdir, "Sx_opt.npy"), Sx)
    np.save(os.path.join(args.outdir, "Sy_opt.npy"), Sy)
    np.save(os.path.join(args.outdir, "Sz_opt.npy"), Sz)
    np.save(os.path.join(args.outdir, "H_hist.npy"), np.array(H_hist, float))
    print(f"[OK] Saved optimized spins to {args.outdir}")

if __name__ == "__main__":
    main()