# fits_to_profile.py — estrai r(z) da FITS + asse DS9 (IMAGE coords)
import os, re, csv, argparse, numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from math import cos, sin, radians, sqrt

def read_ds9_polyline_image(path):
    txt = open(path, "r").read()
    m = re.search(r'polyline\(([^)]+)\)', txt, flags=re.I)
    if not m:
        raise RuntimeError("Polyline non trovata nel file DS9 (serve IMAGE coords).")
    nums = [float(s) for s in re.split(r'[ ,]+', m.group(1).strip())]
    pts = np.array(nums).reshape(-1, 2)  # (x,y) in pixel (IMAGE)
    return pts  # shape (N,2)

def gaussian(x, A, x0, sigma, C):
    return A*np.exp(-0.5*((x-x0)/sigma)**2) + C

def beam_proj_fwhm_mas(bmaj_mas, bmin_mas, bpa_deg, cut_pa_deg):
    # FWHM efficace del beam lungo la direzione del taglio
    phi = radians(cut_pa_deg - bpa_deg)
    f2 = (bmaj_mas**2)*(np.sin(phi)**2) + (bmin_mas**2)*(np.cos(phi)**2)
    return sqrt(max(f2, 0.0))

def bilinear_sample(img, xs, ys):
    coords = np.vstack([ys, xs])  # (2, M)
    return map_coordinates(img, coords, order=1, mode='nearest')

def polyline_arclength(pts):
    d = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s

def run(fits_path, reg_path, step_mas=0.05, half_cut_mas=1.0, ncut=200, out_csv="data/obs/M87_raw.csv"):
    hdul = fits.open(fits_path)
    hdr = hdul[0].header
    img = hdul[0].data.squeeze()
    # scala mas/pixel (da CDELT/CD)
    if "CDELT1" in hdr and "CDELT2" in hdr:
        cd1 = abs(hdr["CDELT1"]); cd2 = abs(hdr["CDELT2"])  # deg/pix
    else:
        cd1 = (hdr.get("CD1_1",0.0)**2 + hdr.get("CD1_2",0.0)**2)**0.5
        cd2 = (hdr.get("CD2_1",0.0)**2 + hdr.get("CD2_2",0.0)**2)**0.5
    mas_per_pix_x = cd1 * 3600.0 * 1000.0
    mas_per_pix_y = cd2 * 3600.0 * 1000.0
    mas_per_pix = 0.5*(mas_per_pix_x + mas_per_pix_y)
    # beam (deg -> mas)
    bmaj_mas = hdr.get("BMAJ", 0.0) * 3600.0 * 1000.0
    bmin_mas = hdr.get("BMIN", 0.0) * 3600.0 * 1000.0
    bpa_deg  = hdr.get("BPA", 0.0)
    # polyline DS9 in IMAGE coords
    pts = read_ds9_polyline_image(reg_path)
    # campiona lungo la polyline ogni step_mas
    s_pix = polyline_arclength(pts)
    total_mas = s_pix[-1] * mas_per_pix
    n_samples = max(2, int(total_mas / step_mas))
    s_target_mas = np.linspace(0, total_mas, n_samples)
    s_target_pix = s_target_mas / mas_per_pix
    # interp posizioni lungo l’asse
    x = np.interp(s_target_pix, s_pix, pts[:,0])
    y = np.interp(s_target_pix, s_pix, pts[:,1])
    # tangente e perpendicolare
    dx = np.gradient(x); dy = np.gradient(y)
    theta = np.arctan2(dy, dx)           # rad
    phi = theta + np.pi/2.0              # perpendicolare
    # taglio trasverso
    t = np.linspace(-half_cut_mas, +half_cut_mas, ncut)  # mas
    tx = (t / mas_per_pix)  # in pixel
    z_mas_list = []; r_mas_list = []
    for i in range(n_samples):
        xx = x[i] + tx * np.cos(phi[i])
        yy = y[i] + tx * np.sin(phi[i])
        prof = bilinear_sample(img, xx, yy)
        # fit Gaussiana 1D
        x_mas = t
        y0 = float(np.nanmax(prof)); c0 = float(np.nanmin(prof))
        try:
            popt,_ = curve_fit(gaussian, x_mas, prof, p0=[y0, 0.0, 0.1, c0], maxfev=3000)
            A, x0, sigma, C = popt
            fwhm_meas = 2.354820045 * abs(sigma)  # mas
            cut_pa_deg = np.degrees(phi[i])
            beam_eff = beam_proj_fwhm_mas(bmaj_mas, bmin_mas, bpa_deg, cut_pa_deg)
            f2 = max(fwhm_meas**2 - beam_eff**2, 0.0)
            fwhm_dec = sqrt(f2)
            r_mas = 0.5 * fwhm_dec  # half width
        except Exception:
            r_mas = np.nan
        z_mas_list.append(s_target_mas[i])
        r_mas_list.append(r_mas)
    # salva CSV
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["z_mas","r_mas"])
        w.writeheader()
        for zi,ri in zip(z_mas_list, r_mas_list):
            w.writerow({"z_mas": float(zi), "r_mas": ("" if np.isnan(ri) else float(ri))})
    print(f"[OK] Saved {out_csv}")
    print(f"[INFO] beam: BMAJ={bmaj_mas:.3f} mas, BMIN={bmin_mas:.3f} mas, BPA={bpa_deg:.1f} deg ; scale ≈ {mas_per_pix:.3f} mas/pix")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True)
    ap.add_argument("--axis_reg", required=True, help="DS9 region (IMAGE coords) polyline dell'asse")
    ap.add_argument("--step_mas", type=float, default=0.05)
    ap.add_argument("--half_cut_mas", type=float, default=1.0)
    ap.add_argument("--ncut", type=int, default=200)
    ap.add_argument("--out_csv", default="./data/obs/M87_raw.csv")
    args = ap.parse_args()
    run(args.fits, args.axis_reg, args.step_mas, args.half_cut_mas, args.ncut, args.out_csv)
