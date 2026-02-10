import numpy as np, os

OPT_DIR = "figures/K16_optimized"
Sz_opt_f = os.path.join(OPT_DIR, "Sz_opt.npy")
Sz_del_f = os.path.join(OPT_DIR, "Sz_delta.npy")

def stats(name, A):
    A = np.asarray(A)
    finite = np.isfinite(A)
    q = np.quantile(A[finite], [0, .01, .1, .5, .9, .99, 1]) if finite.any() else None
    return {
        "name": name, "shape": A.shape, "dtype": str(A.dtype),
        "finite_%": 100*finite.mean(),
        "min": float(np.nanmin(A)), "max": float(np.nanmax(A)),
        "quant": q.tolist() if q is not None else None,
        "zeros_%": 100*np.mean(np.isclose(A, 0.0, atol=1e-14))
    }

assert os.path.exists(Sz_opt_f), "Manca Sz_opt.npy"
Sz_opt = np.load(Sz_opt_f)

print("== STATS Sz_opt ==")
print(stats("Sz_opt", Sz_opt))

use_delta = False
if os.path.exists(Sz_del_f):
    Sz_delta = np.load(Sz_del_f)
    print("\n== STATS Sz_delta ==")
    print(stats("Sz_delta", Sz_delta))
    # criterio semplice: se delta non è tutto zero, usiamolo come candidato
    if np.any(np.abs(Sz_delta) > 1e-12):
        use_delta = True

if use_delta:
    Sz_fixed = Sz_delta.astype(np.float32, copy=False)
    np.save(os.path.join(OPT_DIR, "Sz_opt_fixed.npy"), Sz_fixed)
    print("\n[OK] Creato Sz_opt_fixed.npy da Sz_delta (non nullo).")
else:
    print("\n[ATTENZIONE] Sz_delta mancante o nullo. Non posso ricostruire Sz.")
