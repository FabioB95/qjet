# bhac_viz_all.py
# - Slices d2/d3 sovrapposte (rho) con clim coerenti
# - Slice MERIDIANA dal 3D (ultimo timestep trovato)
# Dipendenze: pyvista, numpy, pillow (solo se vuoi fare mosaici extra)

import os, re, glob
from pathlib import Path
import numpy as np
import pyvista as pv

# === CONFIG ================================================================
OUTDIR = Path(r"..\bhac\runs\komissarov3d\output")  # <<-- modifica se serve
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGS = OUTDIR / "figs"
FIGS.mkdir(exist_ok=True)

pv.global_theme.allow_empty_mesh = True  # evita crash se una slice è vuota
# ===========================================================================


def combine_pieces(base_pattern: str):
    """Unisce pezzi MPI che matchano base_pattern (es: 'data3D0005*.vtu')."""
    pieces = sorted(glob.glob(str(OUTDIR / base_pattern)))
    if not pieces:
        return None
    mb = pv.MultiBlock([pv.read(p) for p in pieces])
    return mb.combine()


def ensure_point_scalar(mesh: pv.DataSet, name: str) -> pv.DataSet:
    """Se 'name' è solo in cell_data, converti a point_data."""
    if name in mesh.point_data:
        return mesh
    if name in mesh.cell_data:
        return mesh.cell_data_to_point_data()
    raise KeyError(f"Variabile '{name}' non trovata nel mesh.")


# ---------------------------------------------------------------------------
# 1) SLICES d2/d3 SOVRAPPOSTE (rho)
# ---------------------------------------------------------------------------
slice_files = sorted(glob.glob(str(OUTDIR / "data3D_d*_x+*_n*.vtu")))
if slice_files:
    # raggruppa per base (togli _n000X.vtu)
    groups = {}
    for f in slice_files:
        base = re.sub(r"_n\d{4}\.vtu$", "", f)
        groups.setdefault(base, []).append(f)

    # prendi tutte le rho per coerentizzare la colorbar (percentili globali)
    all_rho = []
    meshes = []
    for base, pieces in groups.items():
        mb = pv.MultiBlock([pv.read(p) for p in sorted(pieces)])
        m = mb.combine()
        try:
            m = ensure_point_scalar(m, "rho")
            meshes.append(m)
            all_rho.append(m["rho"])
        except KeyError:
            pass

    if meshes and all_rho:
        all_rho = np.concatenate(all_rho)
        p5, p95 = np.nanpercentile(all_rho, [5, 95])

        pl = pv.Plotter(off_screen=True, window_size=(1600, 1100))
        pl.set_background("white")
        for m in meshes:
            pl.add_mesh(
                m, scalars="rho", cmap="viridis",
                clim=(p5, p95), show_scalar_bar=False, opacity=0.65
            )
        pl.add_axes()
        out_png = str(FIGS / "all_slices_rho.png")
        pl.show(screenshot=out_png)
        pl.close()
        print(f"[OK] Salvato: {out_png}")
    else:
        print("[WARN] Nessuna 'rho' nelle slice d2/d3.")
else:
    print("[INFO] Nessuna slice d2/d3 trovata.")


# ---------------------------------------------------------------------------
# 2) SLICE MERIDIANA DAL 3D (ultimo timestep)
#    (normale=(0,1,0) -> piano XZ; questo interseca sempre il tuo 'spicchio')
# ---------------------------------------------------------------------------
# cerco sia i file “single” che i pezzi MPI
step_nums = set()
for f in glob.glob(str(OUTDIR / "data3D????.vtu")):
    step_nums.add(re.findall(r"data3D(\d{4})\.vtu$", os.path.basename(f))[0])
for f in glob.glob(str(OUTDIR / "data3D????_n*.vtu")):
    step_nums.add(re.findall(r"data3D(\d{4})_n\d{4}\.vtu$", os.path.basename(f))[0])

if step_nums:
    last = sorted(step_nums)[-1]
    # preferisco unire i pezzi, altrimenti uso il singolo file
    ds = combine_pieces(f"data3D{last}_n*.vtu") or pv.read(OUTDIR / f"data3D{last}.vtu")

    try:
        ds = ensure_point_scalar(ds, "rho")
        # faccio una slice MERIDIANA (piano XZ, normale lungo y)
        sl = ds.slice(normal=(0, 1, 0), origin=ds.center)
        if sl.n_points == 0:
            # fallback: altra meridiana (piano YZ)
            sl = ds.slice(normal=(1, 0, 0), origin=ds.center)

        if sl.n_points == 0:
            print("[WARN] La slice meridiana non interseca il dominio.")
        else:
            # log10 per aumentare il contrasto
            logrho = np.log10(np.clip(np.asarray(sl["rho"]), 1e-30, None))
            sl["logrho"] = logrho
            clim = (np.nanpercentile(logrho, 5), np.nanpercentile(logrho, 95))

            pl = pv.Plotter(off_screen=True, window_size=(1400, 1000))
            pl.set_background("white")
            pl.add_mesh(sl, scalars="logrho", cmap="viridis",
                        clim=clim, show_scalar_bar=True)
            pl.add_axes()
            # vista “laterale” per una meridiana
            pl.camera_position = "xz"
            out_png = str(FIGS / f"slice_meridiana_3D_step{last}.png")
            pl.show(screenshot=out_png)
            pl.close()
            print(f"[OK] Salvato: {out_png}")

    except KeyError:
        print("[WARN] Variabile 'rho' non trovata nel 3D.")
else:
    print("[INFO] Nessun file 3D data3D????*.vtu trovato.")
