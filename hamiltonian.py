import numpy as np

def frame_dragging_energy(Sz, h_i):
    valid = np.isfinite(Sz) & np.isfinite(h_i)
    return -np.sum(h_i[valid] * Sz[valid])

def kink_penalty(Sx, Sy, Sz, lambda_kink=1.0):
    valid = np.isfinite(Sx) & np.isfinite(Sy) & np.isfinite(Sz)
    if np.sum(valid) < 2:
        return 0.0
    Sx_v, Sy_v, Sz_v = Sx[valid], Sy[valid], Sz[valid]
    dSx = np.gradient(Sx_v)
    dSy = np.gradient(Sy_v)
    dSz = np.gradient(Sz_v)
    return lambda_kink * np.sum(dSx**2 + dSy**2 + dSz**2)

def jet_power_constraint(Br_horizon, target_power=0.1, mu=0.1, Omega_H=0.1):
    Phi_H = Br_horizon
    P_jet = (1/(30*np.pi)) * Phi_H**2 * Omega_H**2
    return -mu * (P_jet - target_power)**2


# --- External pressure steering (opening-angle control) -----------------------
def _theta_target(z_rg: np.ndarray, k_ext: float, theta0_rad: float, z0_rg: float) -> np.ndarray:
    """
    Target opening angle profile (radians):
        theta_tgt(z) = theta0 * (z/z0)^(-(k-1)/2)
    """
    zz = np.maximum(z_rg, 1e-6)
    expo = -0.5 * (k_ext - 1.0)
    return theta0_rad * (zz / max(z0_rg, 1e-6)) ** expo

def external_pressure_energy(Sx, Sy, Sz, z_rg,
                             alpha_ext: float = 0.0,
                             k_ext: float = 2.0,
                             theta0_deg: float = 8.0,
                             z0_rg: float = 50.0) -> float:
    """
    H_ext = alpha_ext * sum (theta - theta_tgt(z))^2,
    where theta = arccos(|Sz|) is the local opening angle to the spin axis.
    """
    if alpha_ext <= 0.0:
        return 0.0
    eps = 1e-12
    # local opening angle (radians)
    cosang = np.clip(np.abs(Sz), 0.0, 1.0 - eps)
    theta = np.arccos(cosang)
    theta_tgt = _theta_target(z_rg, k_ext, np.deg2rad(theta0_deg), z0_rg)
    return float(alpha_ext) * np.sum((theta - theta_tgt) ** 2)


# --- Shear-layer control (edge pitch regularization) --------------------------
def _radius_in_cyl_coords(ix, iy, iz, shape, eps=1e-12):
    """
    Build a cylindrical radius R on the (x,y) plane assuming the jet axis is z.
    Returns an array R in [0, ~1] normalized by max radius.
    """
    Nx, Ny, Nz = shape
    x = np.linspace(-1.0, 1.0, Nx)[:, None, None]
    y = np.linspace(-1.0, 1.0, Ny)[None, :, None]
    R = np.sqrt(x**2 + y**2)
    Rmax = np.max(R) + eps
    return (R / Rmax)

def _pitch_cosine(Sx, Sy, Sz, eps=1e-12):
    """
    cos(pitch) = |B_z| / |B|. Edge wants finite pitch (not purely axial/toroidal).
    """
    B2 = Sx*Sx + Sy*Sy + Sz*Sz + eps
    return np.abs(Sz) / np.sqrt(B2)

def shear_layer_energy(Sx, Sy, Sz,
                       beta_shear: float = 0.0,
                       R0: float = 0.7,
                       width: float = 0.15,
                       pitch_target: float = 0.85):
    """
    Penalize deviation of cos(pitch) from a target in an annulus near the edge.
    H_shear = beta * sum W(R) * (cos(pitch) - pitch_target)^2
    - R0: annulus center (0=center, 1=edge)
    - width: annulus half-width (0.1-0.2 works)
    - pitch_target: target cos(pitch) in the shear (e.g. 0.8-0.9)
    """
    if beta_shear <= 0.0:
        return 0.0
    R = _radius_in_cyl_coords(0,0,0, Sx.shape)
    # smooth annulus weight
    W = 0.5 * (1.0 + np.tanh((R - (R0 - width))/1e-6)) * 0.5 * (1.0 - np.tanh((R - (R0 + width))/1e-6))
    # numerically safer bell
    W = np.exp(-((R - R0)**2) / (2.0*width*width))
    cosp = _pitch_cosine(Sx, Sy, Sz)
    return float(beta_shear) * np.sum(W * (cosp - pitch_target)**2)

def shear_layer_grad(Sx, Sy, Sz,
                     beta_shear: float = 0.0,
                     R0: float = 0.7,
                     width: float = 0.15,
                     pitch_target: float = 0.85,
                     eps: float = 1e-12):
    """
    dH_shear/dS: only Sz enters explicitly through cos(pitch) = |Sz|/|S|.
    d cosp / d Sz = sign(Sz)/|S| - |Sz| * Sz / |S|^3
    """
    if beta_shear <= 0.0:
        z = np.zeros_like(Sz)
        return np.zeros_like(Sx), np.zeros_like(Sy), z

    R = _radius_in_cyl_coords(0,0,0, Sx.shape)
    W = np.exp(-((R - R0)**2) / (2.0*width*width))

    B2 = Sx*Sx + Sy*Sy + Sz*Sz + eps
    B = np.sqrt(B2)
    cosp = np.abs(Sz) / B

    dcosp_dSz = np.sign(Sz)/B - (np.abs(Sz) * Sz) / (B**3 + eps)
    # dH/dSz = 2 * beta * W * (cosp - target) * dcosp_dSz
    dHz = 2.0 * beta_shear * W * (cosp - pitch_target) * dcosp_dSz

    # No explicit dependence on Sx,Sy in this simple form
    dHx = np.zeros_like(Sx)
    dHy = np.zeros_like(Sy)
    return dHx, dHy, dHz


def total_hamiltonian(Sx, Sy, Sz, Br, h_i,
                      target_power=0.1,
                      lambda_kink=1.0,
                      mu=0.1,
                      Omega_H=0.1,
                      # NEW: external-pressure controls
                      alpha_ext: float = 0.0,
                      k_ext: float = 2.0,
                      theta0_deg: float = 8.0,
                      z0_rg: float = 50.0,
                      z_rg: np.ndarray = None,
                      # NEW: shear-layer controls
                      beta_shear: float = 0.0,
                      R0: float = 0.7,
                      shear_width: float = 0.15,
                      pitch_target: float = 0.85):
    """
    Returns (H_total, components_dict)
    """
    H_frame = frame_dragging_energy(Sz, h_i)
    H_reconn = kink_penalty(Sx, Sy, Sz, lambda_kink)
    H_power = jet_power_constraint(Br[0], target_power, mu, Omega_H)

    # --- external pressure term (optional) ---
    if z_rg is None:
        # minimal fallback: a monotonic ramp matching array shape
        z_rg = np.linspace(1.0, 1.0 + Sz.size, Sz.size, dtype=float).reshape(Sz.shape)
    H_ext = external_pressure_energy(Sx, Sy, Sz, z_rg,
                                     alpha_ext=alpha_ext,
                                     k_ext=k_ext,
                                     theta0_deg=theta0_deg,
                                     z0_rg=z0_rg)

    # --- shear-layer term (optional) ---
    H_shear = shear_layer_energy(Sx, Sy, Sz,
                                 beta_shear=beta_shear,
                                 R0=R0,
                                 width=shear_width,
                                 pitch_target=pitch_target)

    H_total = H_frame + H_reconn + H_power + H_ext + H_shear
    return H_total, {
        'H_frame': H_frame,
        'H_reconn': H_reconn,
        'H_power': H_power,
        'H_ext': H_ext,
        'H_shear': H_shear,
    }


def compute_gradients(Sx, Sy, Sz, Br, h_i,
                      target_power: float = 0.1,
                      mu: float = 0.1,
                      Omega_H: float = 0.1,
                      # external-pressure (optional; default off)
                      alpha_ext: float = 0.0,
                      k_ext: float = 2.0,
                      theta0_deg: float = 8.0,
                      z0_rg: float = 50.0,
                      z_rg: np.ndarray | None = None,
                      # shear-layer controls (default off)
                      beta_shear: float = 0.0,
                      R0: float = 0.7,
                      shear_width: float = 0.15,
                      pitch_target: float = 0.85):
    """
    Compute dH/dSx, dH/dSy, dH/dSz for the terms we model analytically.
    - Frame-dragging: dH/dSz = -h_i
    - Kink penalty: (not implemented here)
    - Power term: no spin dependence -> zero
    - External pressure: only Sz derivative (see below)
    - Shear-layer: only Sz derivative (see below)
    """
    # init
    dH_dSx = np.zeros_like(Sx)
    dH_dSy = np.zeros_like(Sy)
    dH_dSz = np.zeros_like(Sz)

    # (1) frame-dragging: H_frame = -Σ h_i * Sz_i
    dH_dSz += -h_i

    # (2) external pressure gradient (optional)
    if alpha_ext > 0.0:
        if z_rg is None:
            z_rg = np.linspace(1.0, 1.0 + Sz.size, Sz.size, dtype=float).reshape(Sz.shape)
        eps = 1e-12
        # theta = arccos(|Sz|)
        cosang = np.clip(np.abs(Sz), 0.0, 1.0 - eps)
        theta = np.arccos(cosang)
        # theta_tgt(z)
        theta_tgt = _theta_target(z_rg, k_ext, np.deg2rad(theta0_deg), z0_rg)
        # d theta / d Sz = - sign(Sz) / sqrt(1 - Sz^2 + eps)
        dtheta_dSz = -np.sign(Sz) / np.sqrt(np.maximum(1.0 - Sz*Sz, eps))
        # chain rule
        dH_dSz += 2.0 * alpha_ext * (theta - theta_tgt) * dtheta_dSz

    # (3) shear-layer gradient (optional)
    if beta_shear > 0.0:
        dHx, dHy, dHz = shear_layer_grad(
            Sx, Sy, Sz,
            beta_shear=beta_shear,
            R0=R0,
            width=shear_width,
            pitch_target=pitch_target
        )
        dH_dSx += dHx
        dH_dSy += dHy
        dH_dSz += dHz

    return dH_dSx, dH_dSy, dH_dSz


# Optional: keep test, but load data directly
if __name__ == "__main__":
    # Load data directly (no import)
    import h5py
    import numpy as np
    
    with h5py.File('../data/grmhd/dump000.h5', 'r') as f:
        data = f['data'][:]
    Br = data[6, :]
    Bt = data[7, :]
    Bp = data[8, :]
    
    # Normalize
    mag = np.sqrt(Br**2 + Bt**2 + Bp**2)
    mag[mag == 0] = 1e-15
    Sx, Sy, Sz = Br/mag, Bt/mag, Bp/mag
    
    # Compute
    h_i = np.zeros_like(Sz)
    h_i[0] = 0.5
    
    H_total, comps = total_hamiltonian(Sx, Sy, Sz, Br, h_i)
    print(f"Total H: {H_total:.4f}")
    print(comps)