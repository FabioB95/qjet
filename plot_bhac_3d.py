# plot_all_slices_rho.py
import os, re, glob, numpy as np, pyvista as pv

OUTDIR = r"..\bhac\runs\komissarov3d\output"
os.makedirs("figs", exist_ok=True)

# prendi solo slice: d2_ e d3_
files = sorted(glob.glob(os.path.join(OUTDIR, "data3D_d*_x+*_n*.vtu")))
if not files:
    raise SystemExit("Nessuna slice trovata in output/")

# raggruppa i pezzi MPI per base (togli _n000X.vtu)
groups = {}
for f in files:
    base = re.sub(r"_n\d{4}\.vtu$", "", f)
    groups.setdefault(base, []).append(f)

pl = pv.Plotter(off_screen=True)

# per avere una mappa cromatica stabile, calcolo p5-p95 su TUTTE le slice
sample_rho = []
for base, pieces in groups.items():
    mb = pv.MultiBlock([pv.read(p) for p in sorted(pieces)])
    mesh = mb.combine()
    if "rho" in mesh.cell_data and "rho" not in mesh.point_data:
        mesh = mesh.cell_data_to_point_data()
    sample_rho.append(mesh["rho"])
p5, p95 = np.nanpercentile(np.concatenate(sample_rho), [5, 95])

# disegna ogni slice e accumula
for base, pieces in groups.items():
    mb   = pv.MultiBlock([pv.read(p) for p in sorted(pieces)])
    mesh = mb.combine()
    if "rho" in mesh.cell_data and "rho" not in mesh.point_data:
        mesh = mesh.cell_data_to_point_data()
    pl.add_mesh(mesh, scalars="rho", clim=(p5, p95), cmap="viridis",
                show_scalar_bar=False, opacity=0.6)

pl.add_axes()
pl.set_background("white")
pl.show(screenshot="figs/all_slices_rho.png")
print("Salvato: figs/all_slices_rho.png")
