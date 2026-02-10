import json, os
import numpy as np
from sklearn.cluster import KMeans

# --- I/O ---
K = 8                         # cambia in 8 o 12 se vuoi
counts_json = f"figures/K{K}_counts.json"
spins_Sx = "../data/spins/data3D0000_Sx.npy"
spins_Sy = "../data/spins/data3D0000_Sy.npy"
spins_Sz = "../data/spins/data3D0000_Sz.npy"
out_dir = f"figures/K{K}_optimized"
os.makedirs(out_dir, exist_ok=True)

# --- carica counts e prendi il bitstring più probabile ---
with open(counts_json) as f:
    counts = json.load(f)
total = sum(counts.values())
probs = {k: v/total for k, v in counts.items()}
best_bit = max(probs.items(), key=lambda kv: kv[1])[0].replace(" ", "")

# --- util: bitstring -> spins ±1 (ordine qubit0 = LSB a dx) ---
def bitstring_to_spins(bitstring, K):
    b = bitstring[-K:]
    return np.array([+1 if x=='0' else -1 for x in b[::-1]], dtype=int)

spins_qubits = bitstring_to_spins(best_bit, K)
print(f"[K={K}] best bitstring={best_bit}  spins={spins_qubits.tolist()}  p={probs[max(probs, key=probs.get)]:.4g}")

# --- carica campo originale (unitario) ---
Sx = np.load(spins_Sx); Sy = np.load(spins_Sy); Sz = np.load(spins_Sz)
mask = np.isfinite(Sx) & np.isfinite(Sy) & np.isfinite(Sz)

# --- ricostruisci gli stessi cluster con KMeans(random_state=0) sull'indice lineare ---
idx_lin = np.arange(mask.sum(), dtype=float)[:, None]
km = KMeans(n_clusters=K, n_init=10, random_state=0)
labels = km.fit_predict(idx_lin)

# --- per ogni cluster, applica il segno su Sz e rinormalizza ---
Sx_v = Sx[mask].copy(); Sy_v = Sy[mask].copy(); Sz_v = Sz[mask].copy()
for c in range(K):
    sel = labels == c
    if not np.any(sel): continue
    zsign = spins_qubits[c]
    Sz_v[sel] = zsign * np.abs(Sz_v[sel])
    # rinormalizza (evita divisioni per zero)
    mag = np.sqrt(Sx_v[sel]**2 + Sy_v[sel]**2 + Sz_v[sel]**2) + 1e-15
    Sx_v[sel] /= mag; Sy_v[sel] /= mag; Sz_v[sel] /= mag

# --- rimetti in griglia piena ---
Sx_opt = np.full_like(Sx, np.nan); Sy_opt = np.full_like(Sy, np.nan); Sz_opt = np.full_like(Sz, np.nan)
Sx_opt[mask], Sy_opt[mask], Sz_opt[mask] = Sx_v, Sy_v, Sz_v

# --- salvataggi ---
np.save(os.path.join(out_dir, "Sx_opt.npy"), Sx_opt)
np.save(os.path.join(out_dir, "Sy_opt.npy"), Sy_opt)
np.save(os.path.join(out_dir, "Sz_opt.npy"), Sz_opt)
np.save(os.path.join(out_dir, "Sz_delta.npy"), (Sz_opt - Sz))

print(f"Salvato campo ottimizzato in {out_dir}")
