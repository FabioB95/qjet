import os
import numpy as np
import h5py

# Cartelle
dumps_dir = "/mnt/c/Users/f.buffoli008/Desktop/ASTRO/QUANTUM_BH/data/grmhd"
out_dir = dumps_dir

def read_dump_ascii(filename):
    """
    Legge un dump HARMPI come lista di righe numeriche.
    Ritorna un array 2D con padding (per uniformità).
    """
    with open(filename, "r", errors="ignore") as f:
        lines = f.readlines()

    rows = []
    max_len = 0

    for line in lines:
        parts = line.strip().split()
        # salta righe vuote o non numeriche
        try:
            nums = [float(x) for x in parts]
        except ValueError:
            continue
        if nums:
            rows.append(nums)
            if len(nums) > max_len:
                max_len = len(nums)

    # padding per avere array rettangolare
    data = np.full((len(rows), max_len), np.nan)
    for i, row in enumerate(rows):
        data[i, :len(row)] = row

    return data


# Converti tutti i dumpXXX
for fname in sorted(os.listdir(dumps_dir)):
    if fname.startswith("dump"):
        in_path = os.path.join(dumps_dir, fname)
        print(f"Converting {in_path} ...")

        arr = read_dump_ascii(in_path)

        out_name = fname + ".h5"
        out_path = os.path.join(out_dir, out_name)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("data", data=arr)

        print(f"  → Saved {out_path}")

# Converti gdump separatamente
for gfile in ["gdump", "gdump2"]:
    gpath = os.path.join(dumps_dir, gfile)
    if os.path.exists(gpath):
        print(f"Converting {gfile} ...")
        garr = read_dump_ascii(gpath)
        out_path = os.path.join(out_dir, f"{gfile}.h5")
        with h5py.File(out_path, "w") as f:
            f.create_dataset("metric", data=garr)
        print(f"  → Saved {out_path}")

for prefix in [ "fdump", "rdump"]:
    for fname in sorted(os.listdir(dumps_dir)):
        if fname.startswith(prefix):
            in_path = os.path.join(dumps_dir, fname)
            print(f"Converting {in_path} ...")

            arr = read_dump_ascii(in_path)

            out_name = fname + ".h5"
            out_path = os.path.join(out_dir, out_name)

            with h5py.File(out_path, "w") as f:
                f.create_dataset("data", data=arr)

            print(f"  → Saved {out_path}")