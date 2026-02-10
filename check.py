import numpy as np, os

PATH = "figures/K16_optimized"
Sx = np.load(os.path.join(PATH, "Sx_opt.npy"))
Sy = np.load(os.path.join(PATH, "Sy_opt.npy"))
Sz = np.load(os.path.join(PATH, "Sz_opt.npy"))

def stats(name, A):
    A = np.asarray(A)
    print(f"\n{name}: shape={A.shape}, dtype={A.dtype}, size={A.size}")
    finite = np.isfinite(A)
    print(f"  finite: {finite.mean()*100:.2f}%")
    if finite.any():
        B = A[finite]
        print(f"  min..max: {B.min():.6g} .. {B.max():.6g}")
        q = np.quantile(B, [0, .01, .1, .5, .9, .99, 1])
        print(f"  quantiles: {q}")
        zeros = (np.abs(B) < 1e-14).mean()*100
        print(f"  |val|<1e-14: {zeros:.2f}%")
    else:
        print("  (no finite values)")

stats("Sx", Sx); stats("Sy", Sy); stats("Sz", Sz)

# prova a dedurre shape 3D se array è 1D
for nx, ny, nz in [(100,100,136), (128,128,128), (64,64,128)]:
    if Sx.size == nx*ny*nz:
        print(f"\nPossibile shape coerente: ({nx},{ny},{nz})")
        break
