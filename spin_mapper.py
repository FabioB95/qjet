# src/spin_mapper.py
import pyvista as pv
import numpy as np
import glob
import os

def normalize_vectors(b1, b2, b3):
    """Normalizza i vettori per ottenere spin unitari (Sx,Sy,Sz)."""
    mag = np.sqrt(b1**2 + b2**2 + b3**2)
    mag[mag == 0] = 1e-30
    Sx = b1 / mag
    Sy = b2 / mag
    Sz = b3 / mag
    return Sx, Sy, Sz

def process_vtu(vtu_file, out_dir="../data/spins"):
    """Carica un file VTU BHAC, estrae b1,b2,b3 e salva gli spin unitari."""
    print(f"Carico file: {vtu_file}")
    ds = pv.read(vtu_file)

    if not all(k in ds.array_names for k in ["b1", "b2", "b3"]):
        raise KeyError("Il file VTU non contiene b1,b2,b3")

    b1, b2, b3 = ds["b1"], ds["b2"], ds["b3"]
    Sx, Sy, Sz = normalize_vectors(b1, b2, b3)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(vtu_file))[0]
    np.save(os.path.join(out_dir, f"{base}_Sx.npy"), Sx)
    np.save(os.path.join(out_dir, f"{base}_Sy.npy"), Sy)
    np.save(os.path.join(out_dir, f"{base}_Sz.npy"), Sz)

    print(f"Salvati spin unitari per {base} in {out_dir}/")

if __name__ == "__main__":
    # prendi il primo VTU disponibile
    files = sorted(glob.glob("../bhac/runs/komissarov3d/output/data3D*.vtu"))
    if not files:
        raise FileNotFoundError("Nessun file VTU trovato!")
    process_vtu(files[0])
