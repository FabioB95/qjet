# src/rt/minsynch.py
import numpy as np
from numpy import cos, sin
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class SynchParams:
    p: float = 2.4          # electron spectral index (2.2–2.6 typical)
    ne0: float = 1.0        # electron density scale (arbitrary units)
    incl_deg: float = 17.0  # line-of-sight inclination (deg) relative to jet axis (z)
    dl: float = 1.0         # path step (in model units)
    doppler: Optional[float] = None  # if you want a constant Doppler boost; else None

def frac_pol(p: float) -> float:
    # Optically thin synchrotron fractional polarization
    # Π0 = (p + 1) / (p + 7/3)
    return (p + 1.0) / (p + 7.0/3.0)

def project_B_to_sky(Bx, By, Bz, incl_rad: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate the volume so that the LOS is along +y' axis.
    We choose: rotate around x by incl_rad so the jet axis (z) tilts toward observer.
    Returns B_perp components (X', Z') in the sky plane.
    """
    cx, sx = np.cos(incl_rad), np.sin(incl_rad)
    # Rotation around x: (x, y, z) -> (x, y*cx - z*sx, y*sx + z*cx)
    Byp = By*cx - Bz*sx
    Bzp = By*sx + Bz*cx
    # LOS is y'; plane of sky is (x', z') = (x, z')
    return Bx, Bzp  # components in sky plane

def stokes_local(Bx, By, Bz, params: SynchParams):
    """
    Compute local emissivities j_I, j_Q, j_U for a voxel.
    We set j_I ∝ ne0 * |B_perp|^{(p+1)/2}; EVPA from projected B direction.
    """
    incl = np.deg2rad(params.incl_deg)
    Bx_sky, Bz_sky = project_B_to_sky(Bx, By, Bz, incl)
    Bperp = np.sqrt(Bx_sky**2 + Bz_sky**2) + 1e-20

    # Exponent for synchrotron emissivity
    alpha = 0.5 * (params.p + 1.0)
    jI = params.ne0 * (Bperp**alpha)

    # EVPA chi: angle of E-vector; for optically thin, E ⟂ B_perp
    # Take magnetic position angle psi_B = atan2(Bz_sky, Bx_sky)
    psiB = np.arctan2(Bz_sky, Bx_sky)
    # EVPA = psi_B + 90° -> Q,U ∝ Π0 * I * cos/sin(2 chi)
    # equivalently rotate by +pi/2: chi = psiB + pi/2
    chi = psiB + 0.5*np.pi

    Pi0 = frac_pol(params.p)
    jQ = Pi0 * jI * np.cos(2.0*chi)
    jU = Pi0 * jI * np.sin(2.0*chi)

    if params.doppler is not None:
        # simple boosting of all Stokes by delta^(3+alpha_s)
        # spectral index alpha_s = (p-1)/2 for synchrotron
        alpha_s = 0.5*(params.p - 1.0)
        boost = params.doppler ** (3.0 + alpha_s)
        jI *= boost; jQ *= boost; jU *= boost

    return jI, jQ, jU

def ray_integrate(jI, jQ, jU, dl: float, axis: int = 1):
    """
    Integrate along LOS (axis=1 => y'-like). Simple sum * dl (optically thin).
    """
    I = np.sum(jI, axis=axis) * dl
    Q = np.sum(jQ, axis=axis) * dl
    U = np.sum(jU, axis=axis) * dl
    return I, Q, U

def evpa_from_QU(Q, U):
    # EVPA in radians, in [-pi/2, pi/2)
    chi = 0.5 * np.arctan2(U, Q)
    return chi

def apply_external_RM(Q, U, lam_m: float, RM_rad_m2: float):
    """
    Rotate EVPA by Δchi = RM * λ^2 (external screen):
    (Q', U') = (Q cos2Δ - U sin2Δ, Q sin2Δ + U cos2Δ)
    """
    dchi = RM_rad_m2 * (lam_m**2)
    c, s = np.cos(2.0*dchi), np.sin(2.0*dchi)
    Qp = Q * c - U * s
    Up = Q * s + U * c
    return Qp, Up

def convolve_elliptical_gaussian(img, bmaj_pix, bmin_pix, bpa_rad):
    """
    Convolve 2D image with elliptical Gaussian beam given major/minor (in pixels)
    and position angle bpa (radians, measured from +x toward +y).
    Implemented via constructing a rotated kernel (size ~6σ).
    """
    from scipy.ndimage import gaussian_filter, rotate
    # approximate by rotating to beam frame, apply separable Gaussian, rotate back
    # Convert FWHM -> sigma: sigma = FWHM / (2 sqrt(2 ln 2))
    f = 2.0 * np.sqrt(2.0*np.log(2.0))
    sigx = bmaj_pix / f
    sigy = bmin_pix / f

    # Rotate image into beam PA frame
    im_rot = rotate(img, angle=np.degrees(bpa_rad), reshape=False, order=1, mode="nearest")
    # Apply separable Gaussian in that frame (approximate): different sigmas along axes
    im_blur_x = gaussian_filter(im_rot, sigma=(sigy, sigx))  # note order (rows, cols)
    # Rotate back
    im_out = rotate(im_blur_x, angle=-np.degrees(bpa_rad), reshape=False, order=1, mode="nearest")
    return im_out
