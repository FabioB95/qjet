# src/benchmark.py
import os, argparse, numpy as np, csv
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_field(folder):
    def pick(fldr, cand):
        for name in cand:
            p = os.path.join(fldr, name)
            if os.path.exists(p):
                return np.load(p)
        raise FileNotFoundError(f"Nessun file tra {cand} in {fldr}")
    Sx = pick(folder, ["Sx_opt.npy", "Sx.npy"])
    Sy = pick(folder, ["Sy_opt.npy", "Sy.npy"])
    # accept more Sz names so we don't have to copy
    Sz = pick(folder, ["Sz_opt_recon.npy", "Sz_opt.npy", "Sz.npy"])
    if Sx.ndim == 1:
        Sx = Sx.reshape(100,100,136); Sy = Sy.reshape(100,100,136); Sz = Sz.reshape(100,100,136)
    return Sx, Sy, Sz

def grid_like(A):
    nx, ny, nz = A.shape
    x = np.linspace(-2,2,nx); y = np.linspace(-2,2,ny); z = np.linspace(-4,4,nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return x, y, z, X, Y, Z

def normalize(Sx, Sy, Sz):
    B = np.sqrt(Sx**2 + Sy**2 + Sz**2) + 1e-30
    return Sx/B, Sy/B, Sz/B

def interp_trilinear(A, px, py, pz, x, y, z):
    nx, ny, nz = A.shape
    tx = (px-x[0])/(x[-1]-x[0])*(nx-1)
    ty = (py-y[0])/(y[-1]-y[0])*(ny-1)
    tz = (pz-z[0])/(z[-1]-z[0])*(nz-1)
    tx = np.clip(tx, 0, nx-1-1e-6)
    ty = np.clip(ty, 0, ny-1-1e-6)
    tz = np.clip(tz, 0, nz-1-1e-6)
    i0 = np.floor(tx).astype(int); j0 = np.floor(ty).astype(int); k0 = np.floor(tz).astype(int)
    i1 = i0 + 1; j1 = j0 + 1; k1 = k0 + 1
    fx = tx - i0; fy = ty - j0; fz = tz - k0
    c000 = A[i0,j0,k0]; c100 = A[i1,j0,k0]; c010 = A[i0,j1,k0]; c110 = A[i1,j1,k0]
    c001 = A[i0,j0,k1]; c101 = A[i1,j0,k1]; c011 = A[i0,j1,k1]; c111 = A[i1,j1,k1]
    c00 = c000*(1-fx) + c100*fx; c01 = c001*(1-fx) + c101*fx
    c10 = c010*(1-fx) + c110*fx; c11 = c011*(1-fx) + c111*fx
    c0 = c00*(1-fy) + c10*fy;   c1 = c01*(1-fy) + c11*fy
    return c0*(1-fz) + c1*fz

def B_interp(px, py, pz, bx, by, bz, x, y, z):
    return np.array([
        interp_trilinear(bx, px, py, pz, x, y, z),
        interp_trilinear(by, px, py, pz, x, y, z),
        interp_trilinear(bz, px, py, pz, x, y, z)
    ])

def trace_line(p0, bx, by, bz, x, y, z, h=0.04, nsteps=600, direction=+1, r_clip=0.38, vmin=1e-8):
    pts=[np.array(p0,float)]; p=np.array(p0,float)
    for _ in range(nsteps):
        if not (x[0]<=p[0]<=x[-1] and y[0]<=p[1]<=y[-1] and z[0]<=p[2]<=z[-1]): break
        if np.hypot(p[0],p[1]) > r_clip: break
        v1 = B_interp(p[0],p[1],p[2],bx,by,bz,x,y,z)*direction
        n1 = np.linalg.norm(v1); 
        if n1 < vmin: break
        pm = p + 0.5*h*(v1/n1)
        v2 = B_interp(pm[0],pm[1],pm[2],bx,by,bz,x,y,z)*direction
        n2 = np.linalg.norm(v2)
        if n2 < vmin: break
        p = p + h*(v2/n2); pts.append(p.copy())
    return np.array(pts)

def core_profile(
    bx, by, bz, x, y, z,
    seeds=64,
    R_MIN=0.08, R_MAX=0.26,
    Z0=0.22,                 # fallback if no z0_list is given
    z0_list=None,            # NEW: multiple launch planes (absolute, nonnegative)
    R_CLIP=0.38,
    h=0.04, nsteps=600, vmin=1e-8,
    agg="percentile", q=35.0 # NEW: width reducer (percentile is robust core proxy)
):
    rng = np.random.default_rng(7)
    radii_up, radii_dn = {}, {}

    if z0_list is None:
        z0_abs_list = [abs(float(Z0))]
    else:
        z0_abs_list = [abs(float(v)) for v in z0_list if np.isfinite(float(v)) and float(v) >= 0.0] or [abs(float(Z0))]

    def _reduce(vals):
        arr = np.array([v for v in vals if np.isfinite(v)], float)
        if arr.size == 0: return np.nan
        if agg == "median":     return float(np.median(arr))
        if agg == "min":        return float(np.min(arr))
        # default: percentile
        return float(np.percentile(arr, q))

    for _ in range(seeds):
        r = rng.uniform(R_MIN, R_MAX)
        a = rng.uniform(0, 2*np.pi)
        x0, y0 = r*np.cos(a), r*np.sin(a)

        for z0_abs in z0_abs_list:
            for (sgn, bag) in [(+1, radii_up), (-1, radii_dn)]:
                zstart = sgn * z0_abs
                pts = trace_line((x0, y0, zstart), bx, by, bz, x, y, z,
                                 h=h, nsteps=nsteps, direction=sgn, r_clip=R_CLIP, vmin=vmin)
                if len(pts) < 3: 
                    continue
                zi = ((pts[:,2]-z[0])/(z[-1]-z[0])*(len(z)-1)).astype(int)
                zi = np.clip(zi, 0, len(z)-1)
                for px, py, k in zip(pts[:,0], pts[:,1], zi):
                    rr = np.hypot(px, py)
                    bag.setdefault(k, []).append(rr)

    out_up = np.full(len(z), np.nan, float)
    out_dn = np.full(len(z), np.nan, float)
    for k, vals in radii_up.items(): out_up[k] = _reduce(vals)
    for k, vals in radii_dn.items(): out_dn[k] = _reduce(vals)
    return out_up, out_dn

def axial_metrics(bx,by,bz):
    b2=bx**2+by**2+bz**2
    return float(np.median(np.abs(bz))), float(np.nanmean(bz**2/(b2+1e-30)))

def _smooth(x,w=9):
    if w<3 or len(x)<w: return x
    k=np.ones(w)/w; y=np.copy(x); m=~np.isnan(x)
    y[m]=np.convolve(x[m],k,mode="same"); return y

def opening_angle(r,z,zmin=1.0,window=9):
    r=_smooth(r,window); dz=(z[-1]-z[0])/(len(z)-1)
    ang=[]
    for i in range(1,len(z)-1):
        if np.isnan(r[i-1]) or np.isnan(r[i]) or np.isnan(r[i+1]): continue
        if abs(z[i])<zmin: continue
        dr=(r[i+1]-r[i-1])/(2*dz); ang.append(np.degrees(np.arctan2(dr,1.0)))
    return float(np.nanmedian(ang)) if len(ang) else np.nan

def save_csv(outcsv, rows, fieldnames):
    os.makedirs(os.path.dirname(outcsv) or ".", exist_ok=True)
    with open(outcsv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[OK] Saved {outcsv}")

def plot_profiles(z, r_upA, r_dnA, r_upB, r_dnB, outpng):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,5))
        plt.plot(z,r_upA,label="A up");  plt.plot(z,r_dnA,label="A down")
        plt.plot(z,r_upB,"--",label="B up"); plt.plot(z,r_dnB,"--",label="B down")
        plt.xlabel("z"); plt.ylabel("median r(z)"); plt.legend(); plt.tight_layout()
        os.makedirs(os.path.dirname(outpng) or ".", exist_ok=True)
        plt.savefig(outpng,dpi=200); plt.close()
        print(f"[OK] Saved {outpng}")
    except Exception as e:
        print("[WARN] plot skipped:", e)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--A", required=True)
    ap.add_argument("--B", required=True)
    ap.add_argument("--out", default="figures/benchmark_core.csv")
    args=ap.parse_args()

    SxA,SyA,SzA=load_field(args.A); SxB,SyB,SzB=load_field(args.B)
    x,y,z,_,_,_=grid_like(SxA)
    bxA,byA,bzA=normalize(SxA,SyA,SzA); bxB,byB,bzB=normalize(SxB,SyB,SzB)
    r_upA, r_dnA = core_profile(bxA,byA,bzA,x,y,z)
    r_upB, r_dnB = core_profile(bxB,byB,bzB,x,y,z)
    rows=[{"model":"A"},{"model":"B"}]  # minimal CLI; detailed metrics available elsewhere
    save_csv(args.out, rows, list(rows[0].keys()))