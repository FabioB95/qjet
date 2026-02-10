# external_benchmark.py — model vs reference with L0 fit, coverage, overlay
import os, csv, argparse, numpy as np

# Clean import from the local package
from src.benchmark import load_field, grid_like, normalize, core_profile

def load_ref(path):
    """
    Load reference profile CSV.
    Wide:  z, r_up_ref, r_down_ref (or r_dn_ref)
    Long:  z, r, side  with side in {up,down}
    Returns: z_ref, r_up_ref, r_dn_ref
    """
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"No header in {path}")
        name_map = {c.lower(): c for c in r.fieldnames}
        cols = set(name_map.keys())

        # Wide
        if {"z","r_up_ref"} <= cols and ({"r_down_ref"} <= cols or {"r_dn_ref"} <= cols):
            zc = name_map["z"]
            upc = name_map["r_up_ref"]
            dnc = name_map["r_down_ref"] if "r_down_ref" in name_map else name_map["r_dn_ref"]
            Z, UP, DN = [], [], []
            for row in r:
                def f(v):
                    try: return float(v) if v != "" else np.nan
                    except: return np.nan
                try: Z.append(float(row[zc]))
                except: Z.append(np.nan)
                UP.append(f(row.get(upc,"")))
                DN.append(f(row.get(dnc,"")))
            return np.array(Z,float), np.array(UP,float), np.array(DN,float)

        # Long
        if {"z","r","side"} <= cols:
            zc, rc, sc = name_map["z"], name_map["r"], name_map["side"]
            byz = {}
            for row in r:
                try: z = float(row[zc])
                except: continue
                side = (row.get(sc,"") or "").strip().lower()
                try: val = float(row[rc])
                except: val = float("nan")
                e = byz.setdefault(z, {"up": [], "down": []})
                if side in ("up","down"):
                    e[side].append(val)
                else:
                    e["up" if z >= 0 else "down"].append(val)
            z_sorted = sorted(byz.keys())
            rup, rdn = [], []
            for z in z_sorted:
                ups = [x for x in byz[z]["up"] if np.isfinite(x)]
                dns = [x for x in byz[z]["down"] if np.isfinite(x)]
                rup.append(np.median(ups) if ups else np.nan)
                rdn.append(np.median(dns) if dns else np.nan)
            return np.array(z_sorted,float), np.array(rup,float), np.array(rdn,float)

        raise ValueError(f"Unrecognized ref format in {path}. Columns: {r.fieldnames}")

def _interp_to_grid(z_grid, z, y):
    z = np.asarray(z,float); y = np.asarray(y,float); zg = np.asarray(z_grid,float)
    m = np.isfinite(z) & np.isfinite(y)
    if m.sum() < 2:
        return np.full_like(zg, np.nan, dtype=float)
    z_valid = z[m]; y_valid = y[m]
    o = np.argsort(z_valid)
    return np.interp(zg, z_valid[o], y_valid[o], left=np.nan, right=np.nan)

def metrics(y_model, y_ref):
    mask = (~np.isnan(y_model)) & (~np.isnan(y_ref))
    n = int(mask.sum())
    if n == 0:
        return np.nan, np.nan, np.nan, 0, 0.0
    e = y_model[mask] - y_ref[mask]
    mse = float(np.mean(e**2))
    mae = float(np.mean(np.abs(e)))
    bias = float(np.mean(e))
    cov = n / float(len(y_model))
    return mse, mae, bias, n, cov

def best_L0(r_model, r_ref, lo, hi, map_mode="divide", ngrid=301):
    """
    Scan L0 in [lo,hi] to minimize MAE.
    map_mode:
      - 'divide': compare (r_model / L0) vs r_ref
      - 'multiply': compare (r_model * L0) vs r_ref
    """
    zs = np.linspace(lo, hi, ngrid)
    best = (np.inf, 1.0)
    for L0 in zs:
        if map_mode == "divide":
            ym = r_model / L0
        else:
            ym = r_model * L0
        _, mae, _, n, cov = metrics(ym, r_ref)
        if n < 2:    # ignore degenerate fits
            continue
        if mae < best[0]:
            best = (mae, L0)
    return best[1]

# -------------------------------------------------------
# NEW: Minimal L0 fitter for bootstrap
# -------------------------------------------------------
def fit_L0_for_profile(z_array, r_array, L0_grid=None):
    """
    Fit L0 by scanning the same L0 grid used in external_benchmark.
    Returns the L0 that minimizes MAE.
    """

    import numpy as np

    # Default L0 grid = the same used in the benchmark scripts
    if L0_grid is None:
        L0_grid = np.logspace(-1, 1.1, 60)  # from 0.1 to about 12

    best_L0 = None
    best_MAE = 1e99

    for L0 in L0_grid:
        # Normalized z
        z_norm = z_array / L0

        # Predict r_model using your 1D geometric model
        # ------------------------------------------------
        # NOTE: this must be exactly the same function your
        # external_benchmark uses !!
        # In your code it's usually called:
        #   r_model = model.predict(z_norm)
        # or
        #   r_model = compute_model_width(z_norm)
        # ------------------------------------------------
        try:
            r_model = compute_model_width(z_norm)  # <--- MUST exist already
        except:
            raise RuntimeError(
                "ERROR: compute_model_width(z) not found.\n"
                "Send me the external_benchmark.py so I can link it properly."
            )

        # Compute MAE
        mae = np.mean(np.abs(r_model - r_array))

        if mae < best_MAE:
            best_MAE = mae
            best_L0 = L0

    return float(best_L0)


def plot_overlay(z_plot, r_up_m, r_dn_m, r_up_r, r_dn_r, L0, map_mode, outpng):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,5))
        lab_m = f"model up (L0={L0:.3f})"; lab_md = f"model down (L0={L0:.3f})"
        if map_mode == "divide":
            mu, md = r_up_m / L0, r_dn_m / L0
        else:
            mu, md = r_up_m * L0, r_dn_m * L0
        plt.plot(z_plot, mu, label=lab_m)
        plt.plot(z_plot, md, label=lab_md)
        plt.plot(z_plot, r_up_r, "--", label="ref up")
        plt.plot(z_plot, r_dn_r, "--", label="ref down")
        plt.xlabel("z (ref grid)"); plt.ylabel("half-width r(z)"); plt.legend()
        os.makedirs(os.path.dirname(outpng) or ".", exist_ok=True)
        plt.tight_layout(); plt.savefig(outpng, dpi=200); plt.close()
        print(f"[OK] Saved overlay {outpng}")
    except Exception as e:
        print("[WARN] overlay skipped:", e)

def _parse_z0s(s):
    if not s: return None
    vals = []
    for t in s.split(","):
        t = t.strip()
        if not t: continue
        try:
            v = float(t)
            if v >= 0: vals.append(v)
        except:
            pass
    return vals or None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", required=True, help="model folder with Sx/Sy/Sz .npy")
    ap.add_argument("--ref", required=True, help="reference CSV (wide or long)")
    ap.add_argument("--out", default="figures/external_benchmark.csv")
    # L0 fitting & plotting
    ap.add_argument("--fit_L0", action="store_true", help="scan L0 to minimize MAE")
    ap.add_argument("--L0_lo", type=float, default=0.1)
    ap.add_argument("--L0_hi", type=float, default=12.0)
    ap.add_argument("--map", choices=["divide","multiply"], default="divide")
    ap.add_argument("--min_cov", type=float, default=0.50, help="min coverage fraction required")
    ap.add_argument("--overlay_png", default=None)
    # seeding/geometry controls
    ap.add_argument("--seeds", type=int, default=8192)
    ap.add_argument("--z0", type=float, default=0.22)
    ap.add_argument("--r_min", type=float, default=0.03)
    ap.add_argument("--r_max", type=float, default=0.35)
    ap.add_argument("--r_clip", type=float, default=0.50)
    # NEW: tracer controls
    ap.add_argument("--h", type=float, default=0.008)
    ap.add_argument("--nsteps", type=int, default=12000)
    ap.add_argument("--vmin", type=float, default=1e-8)
    # NEW: multi-z0 and robust width
    ap.add_argument("--agg", choices=["median","percentile","min"], default="percentile")
    ap.add_argument("--q", type=float, default=35.0, help="percentile for --agg=percentile")
    ap.add_argument("--z0s", type=str, default="", help="comma-separated list of nonnegative z0 planes")
    ap.add_argument("--z0s_mode", choices=["list","auto64","all"], default="auto64",
                    help="auto64: 64 |z0| in [0.02,3.8]; all: every |z| grid; list: use --z0s")
    args = ap.parse_args()

    # --- Load model field
    Sx, Sy, Sz = load_field(args.A)
    x, y, z_model, _, _, _ = grid_like(Sx)
    bx, by, bz = normalize(Sx, Sy, Sz)

    # --- Build z0_list
    z0_list = None
    if args.z0s_mode == "list":
        z0_list = _parse_z0s(args.z0s)
    elif args.z0s_mode == "auto64":
        z0_list = list(np.linspace(0.02, 3.8, 64))
    elif args.z0s_mode == "all":
        # use every |z| bin (clipped to >0 to avoid exactly 0)
        z0_list = list(np.unique(np.clip(np.abs(z_model), 1e-6, None)))

    # --- Build model core profiles
    r_up_model, r_dn_model = core_profile(
        bx, by, bz, x, y, z_model,
        seeds=args.seeds,
        R_MIN=args.r_min, R_MAX=args.r_max,
        Z0=args.z0, z0_list=z0_list,
        R_CLIP=args.r_clip,
        h=args.h, nsteps=args.nsteps, vmin=args.vmin,
        agg=args.agg, q=args.q
    )

    # --- Load reference and interpolate ONTO MODEL GRID for fitting
    z_ref, r_up_ref, r_dn_ref = load_ref(args.ref)
    r_up_ref_on_model = _interp_to_grid(z_model, z_ref, r_up_ref)
    r_dn_ref_on_model = _interp_to_grid(z_model, z_ref, r_dn_ref)

    # --- Optionally fit L0 (independently on 'up' only; apply same L0 to 'down')
    L0 = 1.0
    if args.fit_L0:
        L0 = best_L0(r_up_model, r_up_ref_on_model, args.L0_lo, args.L0_hi, map_mode=args.map)
        print(f"[FIT] Best L0 ≈ {L0:.6f} (map={args.map})")

    # Scale model for metrics (on model grid)
    if args.map == "divide":
        r_up_m = r_up_model / L0
        r_dn_m = r_dn_model / L0
    else:
        r_up_m = r_up_model * L0
        r_dn_m = r_dn_model * L0

    # --- Metrics on model grid
    mse_up, mae_up, bias_up, n_up, cov_up = metrics(r_up_m, r_up_ref_on_model)
    mse_dn, mae_dn, bias_dn, n_dn, cov_dn = metrics(r_dn_m, r_dn_ref_on_model)

    # Guard on coverage if requested
    if args.fit_L0 and (cov_up < args.min_cov and cov_dn < args.min_cov):
        print(f"[WARN] coverage too low (up={cov_up:.2f}, down={cov_dn:.2f}) "
              f"< min_cov={args.min_cov:.2f}. Consider increasing --seeds or using --z0s_mode all.")
    # --- Save CSV
    rows = [
        {"side":"up","MSE":mse_up,"MAE":mae_up,"bias":bias_up,
         "L0":L0,"n_pts":n_up,"coverage_frac":cov_up},
        {"side":"down","MSE":mse_dn,"MAE":mae_dn,"bias":bias_dn,
         "L0":L0,"n_pts":n_dn,"coverage_frac":cov_dn},
    ]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[OK] Saved {args.out}")

    # --- Make overlay on the REFERENCE GRID (nicer to inspect)
    if args.overlay_png:
        # bring model (already scaled) onto ref grid for plotting
        r_up_m_on_ref = _interp_to_grid(z_ref, z_model, r_up_m)
        r_dn_m_on_ref = _interp_to_grid(z_ref, z_model, r_dn_m)
        plot_overlay(z_ref, r_up_m_on_ref, r_dn_m_on_ref, r_up_ref, r_dn_ref,
                     L0=L0, map_mode=args.map, outpng=args.overlay_png)