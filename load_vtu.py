# src/load_vtu.py
import pyvista as pv
import glob
import numpy as np

def load_bhac_vtu(pattern="../bhac/runs/komissarov3d/output/data3D*.vtu"):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nessun file trovato con pattern {pattern}")
    
    # prendiamo un timestep
    fname = files[0]
    print(f"Carico: {fname}")
    
    ds = pv.read(fname)

    # lista variabili disponibili
    print("Variabili disponibili:", ds.array_names)

    # cerchiamo i campi magnetici
    for key in ["b1","b2","b3","B1","B2","B3"]:
        if key in ds.array_names:
            print(f"Trovato campo: {key}, shape={ds[key].shape}")

    return ds

if __name__ == "__main__":
    ds = load_bhac_vtu()
    print("OK: lettura VTU riuscita")
