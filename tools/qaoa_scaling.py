#!/usr/bin/env python3
"""
QAOA scaling helper (use locally with Qiskit).

Example:
  python tools/qaoa_scaling.py --Ks 16,32 --ps 1,3,4 --out src/figures/QAOA_SCALING
"""
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
except Exception:
    QuantumCircuit = None  # Qiskit not available here

def qaoa_circuit(K, p):
    qc = QuantumCircuit(K)
    gamma = [Parameter(f"γ_{i+1}") for i in range(p)]
    beta  = [Parameter(f"β_{i+1}") for i in range(p)]
    # initial layer
    for q in range(K):
        qc.h(q)
    # cost + mixer layers (linear chain entanglers for simplicity)
    for layer in range(p):
        for q in range(K - 1):
            qc.cx(q, q + 1)
            qc.rz(gamma[layer], q + 1)
            qc.cx(q, q + 1)
        for q in range(K):
            qc.rx(2 * beta[layer], q)
    return qc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks", default="16,32", help="comma-separated list, e.g., 16,32")
    ap.add_argument("--ps", default="1,3,4", help="comma-separated list, e.g., 1,3,4")
    ap.add_argument("--out", default="src/figures/QAOA_SCALING")
    ap.add_argument("--backend", default=None, help="Optional: transpile target backend name (basis gates)")
    args = ap.parse_args()

    Ks = [int(x) for x in args.Ks.split(",")]
    ps = [int(x) for x in args.ps.split(",")]
    os.makedirs(args.out, exist_ok=True)

    rows = []
    if QuantumCircuit is None:
        print("[WARN] Qiskit not available in this environment; producing empty summary.")
        for K in Ks:
            for p in ps:
                rows.append({"K": K, "p": p, "depth": None, "two_qubit": None, "rz": None, "sx": None})
    else:
        for K in Ks:
            for p in ps:
                qc = qaoa_circuit(K, p)
                # If no backend provided, use a generic basis similar to ibm backends
                basis_gates = ['rz', 'sx', 'x', 'ecr'] if args.backend is None else None
                tqc = transpile(qc, optimization_level=2, basis_gates=basis_gates)
                ops = tqc.count_ops()
                depth = tqc.depth()
                rows.append({
                    "K": K, "p": p,
                    "depth": int(depth),
                    "two_qubit": int(ops.get('ecr', 0) + ops.get('cx', 0)),
                    "rz": int(ops.get('rz', 0)),
                    "sx": int(ops.get('sx', 0) + ops.get('x', 0)),
                })
                with open(os.path.join(args.out, f"K{K}_p{p}_metrics.json"), "w") as f:
                    json.dump(rows[-1], f, indent=2)
                print(f"[OK] K={K} p={p} depth={rows[-1]['depth']} 2Q={rows[-1]['two_qubit']}")

    # Save summary CSV
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.out, "summary.csv")
        df.to_csv(csv_path, index=False)
        print("[OK] Wrote", csv_path)

        # Quick plots
        for metric in ["depth", "two_qubit"]:
            plt.figure()
            for K in Ks:
                sub = df[df["K"] == K]
                plt.plot(sub["p"], sub[metric], marker="o", label=f"K={K}")
            plt.xlabel("p (layers)")
            plt.ylabel(metric)
            plt.title(f"QAOA scaling: {metric}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, f"{metric}_vs_p.png"), dpi=140)
            plt.close()
    except Exception as e:
        print("[WARN] Could not write summary plots/CSV:", e)

if __name__ == "__main__":
    main()
