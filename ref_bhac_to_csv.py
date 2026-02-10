# ref_bhac_to_csv.py  — robusto: cell->point, point-only, sample+fallback interpolate, asse auto
import os, glob, numpy as np, csv
import pyvista as pv

# -------- CONFIG --------
OUTDIR = r"..\bhac\runs\komissarov3d\output"

NX, NY, NZ = 100, 100, 136
XHALF, YHALF, ZHALF = 2.0, 2.0, 4.0   # box locale: [-2,2]x[-2,2]x[-4,4]

# core/streamlines (come benchmark interno)
R_MIN, R_MAX = 0.08, 0.26
Z0 = 0.22
R_CLIP = 0.38
N_SEEDS = 64
SEED_RNG = 7

def stats(arr, name):
    try:
        return f"{name}: min={np.nanmin(arr):.3e} max={np.nanmax(arr):.3e}"
    except ValueError:
        return f"{name}: (empty)"

# ---------- 1) Leggi e unisci i pezzi VTU ----------
files = sorted(glob.glob(os.path.join(OUTDIR, "data3D_d*_x+*_n*.vtu")))
if not files:
    raise SystemExit(f"Nessun VTU trovato in {OUTDIR}. Controlla il percorso: {OUTDIR}")

print(f"[BHAC] Trovati {len(files)} pezzi VTU. Carico e unisco ...")
blocks = [pv.read(f) for f in files]
mesh = blocks[0]
for m in blocks[1:]:
    mesh = mesh.merge(m, merge_points=True)

print("[BHAC] Mesh unita:")
print(mesh)

# ---------- 2) Assicurati che i campi siano POINT-DATA ----------
if len(mesh.point_data) == 0 and len(mesh.cell_data) > 0:
    mesh = mesh.cell_data_to_point_data(pass_cell_data=False)
    print("[BHAC] Convertito cell_data -> point_data")

pnames = set(mesh.point_data.keys())
print("[BHAC] Point arrays disponibili:", sorted(pnames))

def pick_name_point(mesh, candidates):
    pnames = set(mesh.point_data.keys())
    for c in candidates:
        if c in pnames:
            return c
    return None

Bx_name = pick_name_point(mesh, ["b1","Bx","B_x","B1","B^1","B_1","magnetic_field_x"])
By_name = pick_name_point(mesh, ["b2","By","B_y","B2","B^2","B_2","magnetic_field_y"])
Bz_name = pick_name_point(mesh, ["b3","Bz","B_z","B3","B^3","B_3","magnetic_field_z"])
if any(n is None for n in (Bx_name, By_name, Bz_name)):
    raise RuntimeError(f"Non trovo 3 componenti B nei point-data. Point arrays: {list(mesh.point_data.keys())}")

print(f"[BHAC] Campi magnetici (point-data): {Bx_name}, {By_name}, {Bz_name}")
print("[BHAC] Stats point-data (pre):",
      stats(mesh.point_data[Bx_name], Bx_name), "|",
      stats(mesh.point_data[By_name], By_name), "|",
      stats(mesh.point_data[Bz_name], Bz_name))

# ---------- 3) Griglia WORLD centrata e arrays locali ----------
xmin,xmax,ymin,ymax,zmin,zmax = mesh.bounds
cx, cy, cz = 0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax)
print(f"[BHAC] Centro dominio ~ ({cx:.3f}, {cy:.3f}, {cz:.3f})")

x_local = np.linspace(-XHALF, XHALF, NX)
y_local = np.linspace(-YHALF, YHALF, NY)
z_local = np.linspace(-ZHALF, ZHALF, NZ)

x_world = x_local + cx
y_world = y_local + cy
z_world = z_local + cz

grid_world = pv.RectilinearGrid(x_world, y_world, z_world)

# ---------- 4) Ricampionamento: sample() -> fallback interpolate() ----------

# ---------- 4) Ricampionamento con KDTree (nearest neighbor) ----------
print("[BHAC] Ricampionamento con KDTree (nearest) ...")
from scipy.spatial import cKDTree

# punti sorgente (mesh) in WORLD coords
P = mesh.points  # (Npts, 3)
Vx = np.asarray(mesh.point_data[Bx_name])
Vy = np.asarray(mesh.point_data[By_name])
Vz = np.asarray(mesh.point_data[Bz_name])

# query points = griglia WORLD (x_world, y_world, z_world)
Xw, Yw, Zw = np.meshgrid(x_world, y_world, z_world, indexing="ij")
Q = np.column_stack([Xw.ravel(), Yw.ravel(), Zw.ravel()])  # (NX*NY*NZ, 3)

# KDTree + query
tree = cKDTree(P)
dist, idx = tree.query(Q, k=1)  # indici dei punti più vicini

Sx = Vx[idx].reshape(NX, NY, NZ, order="C")
Sy = Vy[idx].reshape(NX, NY, NZ, order="C")
Sz = Vz[idx].reshape(NX, NY, NZ, order="C")

print("[BHAC] Stats KDTree:",
      stats(Sx, "Sx"), "|", stats(Sy, "Sy"), "|", stats(Sz, "Sz"))


# ---------- 5) Normalizza e auto-detect asse ----------
Bnorm = np.sqrt(Sx**2 + Sy**2 + Sz**2) + 1e-30
bx0, by0, bz0 = Sx/Bnorm, Sy/Bnorm, Sz/Bnorm

# stima asse nel core (r<0.30)
X,Y,Z = np.meshgrid(x_local, y_local, z_local, indexing="ij")
R = np.sqrt(X**2 + Y**2)
mask = (R < 0.30)

comp_meds = np.array([
    np.median(np.abs(bx0[mask])),
    np.median(np.abs(by0[mask])),
    np.median(np.abs(bz0[mask]))
])
axis_idx = int(np.argmax(comp_meds))  # 0->x, 1->y, 2->z
axes_names = ["x","y","z"]
print(f"[BHAC] Asse più assiale (med|b*| core): {axes_names[axis_idx]}")

# rimappa per avere la componente assiale come 'bz'
if axis_idx == 0:
    bx, by, bz = by0, bz0, bx0
elif axis_idx == 1:
    bx, by, bz = bx0, bz0, by0
else:
    bx, by, bz = bx0, by0, bz0

print(f"[BHAC] |bz| median={np.median(np.abs(bz)):.3f}, max={np.max(np.abs(bz)):.3f}")

# ---------- 6) Trilineare locali ----------
def interp_trilinear(A, px, py, pz):
    tx = (px - x_local[0])/(x_local[-1]-x_local[0])*(NX-1)
    ty = (py - y_local[0])/(y_local[-1]-y_local[0])*(NY-1)
    tz = (pz - z_local[0])/(z_local[-1]-z_local[0])*(NZ-1)
    tx = np.clip(tx, 0, NX-1-1e-6)
    ty = np.clip(ty, 0, NY-1-1e-6)
    tz = np.clip(tz, 0, NZ-1-1e-6)
    i0 = np.floor(tx).astype(int); j0 = np.floor(ty).astype(int); k0 = np.floor(tz).astype(int)
    i1 = i0+1; j1 = j0+1; k1 = k0+1
    fx = tx-i0; fy = ty-j0; fz = tz-k0
    c000=A[i0,j0,k0]; c100=A[i1,j0,k0]; c010=A[i0,j1,k0]; c110=A[i1,j1,k0]
    c001=A[i0,j0,k1]; c101=A[i1,j0,k1]; c011=A[i0,j1,k1]; c111=A[i1,j1,k1]
    c00=c000*(1-fx)+c100*fx; c01=c001*(1-fx)+c101*fx
    c10=c010*(1-fx)+c110*fx; c11=c011*(1-fx)+c111*fx
    c0=c00*(1-fy)+c10*fy; c1=c01*(1-fy)+c11*fy
    return c0*(1-fz)+c1*fz

def B_interp(px,py,pz):
    return np.array([interp_trilinear(bx,px,py,pz),
                     interp_trilinear(by,px,py,pz),
                     interp_trilinear(bz,px,py,pz)], float)

# ---------- 7) Streamlines e profili r(z) ----------
def trace_line(p0, h=0.04, nsteps=600, direction=+1, r_clip=0.38, vmin=1e-6):
    pts=[np.array(p0,float)]; p=np.array(p0,float)
    for _ in range(nsteps):
        if not (x_local[0]<=p[0]<=x_local[-1] and y_local[0]<=p[1]<=y_local[-1] and z_local[0]<=p[2]<=z_local[-1]): break
        if np.hypot(p[0],p[1])>r_clip: break
        v1=B_interp(p[0],p[1],p[2])*direction; n1=np.linalg.norm(v1)
        if n1<vmin: break
        pm=p+0.5*h*(v1/n1)
        v2=B_interp(pm[0],pm[1],pm[2])*direction; n2=np.linalg.norm(v2)
        if n2<vmin: break
        p=p+h*(v2/n2); pts.append(p.copy())
    return np.array(pts)

rng = np.random.default_rng(SEED_RNG)
r_up, r_dn = {}, {}
for _ in range(N_SEEDS):
    r = rng.uniform(R_MIN, R_MAX); a = rng.uniform(0, 2*np.pi)
    x0, y0 = r*np.cos(a), r*np.sin(a)
    for (z0,dirn,bag) in [(Z0,+1,r_up), (-Z0,-1,r_dn)]:
        pts = trace_line((x0,y0,z0), direction=dirn, r_clip=R_CLIP)
        if len(pts) < 3: continue
        zi = ((pts[:,2]-z_local[0])/(z_local[-1]-z_local[0])*(NZ-1)).astype(int)
        zi = np.clip(zi, 0, NZ-1)
        for px,py,k in zip(pts[:,0], pts[:,1], zi):
            rr = float(np.hypot(px,py))
            (bag.setdefault(k,[])).append(rr)

def median_profile(bag):
    arr = np.full(NZ, np.nan)
    for k,vals in bag.items(): arr[k] = np.median(vals)
    return arr

r_up_med   = median_profile(r_up)
r_down_med = median_profile(r_dn)

# ---------- 8) Salva CSV ----------
os.makedirs("data", exist_ok=True)
with open("data/ref_profile.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["z","r_up_ref","r_down_ref"])
    w.writeheader()
    for zi, ru, rd in zip(z_local, r_up_med, r_down_med):
        w.writerow({"z": float(zi),
                    "r_up_ref": float(np.nan if np.isnan(ru) else ru),
                    "r_down_ref": float(np.nan if np.isnan(rd) else rd)})
print("[OK] Saved data/ref_profile.csv")
