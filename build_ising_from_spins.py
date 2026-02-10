# build_ising_from_spins.py
import argparse, numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

def build_weights_from_spins(Sx, Sy, Sz, K=8, k_nn=3, cJ=1.0, ch=0.6):
    mask = np.isfinite(Sx) & np.isfinite(Sy) & np.isfinite(Sz)
    V = np.stack([Sx[mask], Sy[mask], Sz[mask]], axis=1)
    idx = np.arange(V.shape[0], dtype=float)[:, None]  # 1D embedding
    km = KMeans(n_clusters=K, n_init=10, random_state=0)
    labels = km.fit_predict(idx)

    mean_S = np.zeros((K, 3))
    centers = np.zeros((K, 1))
    for i in range(K):
        sel = labels == i
        mean_S[i] = V[sel].mean(axis=0)
        centers[i] = idx[sel].mean(axis=0)

    u = -ch * mean_S[:, 2]  # favore verso +z
    tree = cKDTree(centers)
    w = np.zeros((K, K))
    for i in range(K):
        d, nn = tree.query(centers[i], k=min(4, K))  # k_nn=3 -> 1 self + 3 vicini
        for j in np.atleast_1d(nn)[1:]:
            Jij = cJ * float(np.dot(mean_S[i], mean_S[j])) / (float(abs(centers[i]-centers[j])) + 1e-6)
            w[i, j] += -Jij
            w[j, i] += -Jij
    return w, u

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Sx", default="../data/spins/data3D0000_Sx.npy")
    ap.add_argument("--Sy", default="../data/spins/data3D0000_Sy.npy")
    ap.add_argument("--Sz", default="../data/spins/data3D0000_Sz.npy")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--k_nn", type=int, default=3)
    ap.add_argument("--cJ", type=float, default=0.8)
    ap.add_argument("--ch", type=float, default=0.6)
    ap.add_argument("--out", default="ising")
    args = ap.parse_args()

    Sx = np.load(args.Sx); Sy = np.load(args.Sy); Sz = np.load(args.Sz)
    w, u = build_weights_from_spins(Sx, Sy, Sz, K=args.K, k_nn=args.k_nn, cJ=args.cJ, ch=args.ch)
    np.save(f"{args.out}_w_K{args.K}.npy", w)
    np.save(f"{args.out}_u_K{args.K}.npy", u)
    print(f"Saved: {args.out}_w_K{args.K}.npy {args.out}_u_K{args.K}.npy  (K={args.K}, edges~{(np.abs(w)>0).sum()//2})")

if __name__ == "__main__":
    main()
