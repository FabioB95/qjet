# qaoa_bhac_hardware.py
import argparse, json, os
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.visualization import circuit_drawer

def qaoa_ansatz(gamma, beta, w, u):
    K = u.shape[0]
    qc = QuantumCircuit(K, K)
    for q in range(K): qc.h(q)
    for i in range(K):
        for j in range(i+1, K):
            wij = float(w[i, j])
            if abs(wij) > 0: qc.rzz(2*gamma*wij, i, j)
    for i in range(K):
        ui = float(u[i])
        if abs(ui) > 0: qc.rz(2*gamma*ui, i)
    for q in range(K): qc.rx(2*beta, q)
    qc.measure(range(K), range(K))
    return qc

def energy_from_counts(counts, w, u):
    K = u.shape[0]
    tot = sum(counts.values())
    E = 0.0
    for s, c in counts.items():
        bit = s.replace(" ", "")[-K:]
        z = np.array([+1 if b=='0' else -1 for b in bit[::-1]], float)
        p = c / tot
        E += p * (np.dot(u, z) + 0.5 * np.sum((w + w.T) * np.outer(z, z)))
    return E

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", required=True, help="file .npy dei pesi ZZ")
    ap.add_argument("--u", required=True, help="file .npy dei pesi Z")
    ap.add_argument("--out", default="figures/runK")
    ap.add_argument("--grid", type=int, default=5)  # risoluzione griglia beta,gamma
    return ap.parse_args()

def main():
    args = parse_args()
    w = np.load(args.w); u = np.load(args.u); K = u.shape[0]
    print(f"[INFO] K={K}, edges~{(np.abs(w)>0).sum()//2}")

    with open("ibm_secrets.json") as f:
        secrets = json.load(f)
    service = QiskitRuntimeService(
        channel=secrets["channel"], token=secrets["token"], instance=secrets["instance"]
    )
    backend = service.backend("ibm_torino")
    print("Backend:", backend.name, "| qubits:", backend.num_qubits)

    sampler = Sampler(mode=backend)
    G = np.linspace(0.0, np.pi, args.grid)
    B = np.linspace(0.0, np.pi, args.grid)

    best = {"E": 1e9, "gamma": None, "beta": None, "counts": None, "tqc": None}
    for g in G:
        for b in B:
            qc = qaoa_ansatz(g, b, w, u)
            tqc = transpile(qc, backend=backend, optimization_level=1)
            res0 = sampler.run([tqc]).result()[0]

            # SamplerV2: BitArray in res0.data.c
            if hasattr(res0.data, "c"):
                counts = res0.data.c.get_counts()
            elif hasattr(res0.data, "quasi_dists"):
                counts = res0.data.quasi_dists[0].binary_probabilities()
            else:
                raise RuntimeError(f"Formato risultato non riconosciuto: {res0}")

            E = energy_from_counts(counts, w, u)
            print(f"gamma={g:.3f}, beta={b:.3f} -> E={E:.6e}")
            if E < best["E"]:
                best.update({"E": E, "gamma": float(g), "beta": float(b), "counts": counts, "tqc": tqc})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(f"{args.out}_best.json", "w") as f:
        json.dump({"K": K, "gamma": best["gamma"], "beta": best["beta"], "E": best["E"]}, f, indent=2)
    with open(f"{args.out}_counts.json", "w") as f:
        json.dump(best["counts"], f, indent=2)

    try:
        circuit_drawer(best["tqc"], output="mpl", filename=f"{args.out}_circuit.png")
        print("Saved circuit:", f"{args.out}_circuit.png")
    except Exception as e:
        print("Circuit draw failed:", e)

    # qualche metrica del circuito
    try:
        from collections import Counter
        ops = best["tqc"].count_ops()
        depth = best["tqc"].depth()
        with open(f"{args.out}_metrics.json", "w") as f:
            json.dump({"depth": depth, "ops": {k: int(v) for k, v in ops.items()}}, f, indent=2)
        print("Saved metrics:", f"{args.out}_metrics.json")
    except Exception:
        pass

    print(f"\n=== BEST (K={K}) ===\nE={best['E']:.6e}  gamma={best['gamma']:.4f}  beta={best['beta']:.4f}")
    print(f"Counts saved → {args.out}_counts.json")

if __name__ == "__main__":
    main()
