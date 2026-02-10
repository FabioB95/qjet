import os

# cartella dove hai i file .h5
base_dir = "/mnt/c/Users/f.buffoli008/Desktop/ASTRO/QUANTUM_BH/data/grmhd"

count = 0
for fname in os.listdir(base_dir):
    if fname.endswith(".h5.h5"):
        path = os.path.join(base_dir, fname)
        print(f"Deleting {path}")
        os.remove(path)
        count += 1

print(f"✅ Deleted {count} files with .h5.h5")