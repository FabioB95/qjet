# qaoa_make_circuit.py
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RZGate, SXGate, CZGate
from qiskit.transpiler import CouplingMap
import json, pathlib
import matplotlib.pyplot as plt

FIGDIR = pathlib.Path("figures"); FIGDIR.mkdir(exist_ok=True)

def k16_abstract_pdf():
    qc = QuantumCircuit(16, name="QAOA_p1_K16")
    # toy p=1 layer: cost (ZZ via CZ+RZ), mixer (RX~SX)
    for i in range(0,16,2):
        qc.cz(i, (i+1)%16)
        qc.append(RZGate(0.8), [i]); qc.append(RZGate(0.8), [(i+1)%16])
    for q in range(16):
        qc.sx(q); qc.x(q); qc.sx(q)  # simple mixer proxy
    qc.draw(output="mpl", fold=40, filename=str(FIGDIR/"K16_circuit.pdf"))

def torino_layout_pdf():
    # example 8-qubit slice with a synthetic coupling map; replace with real backend when available
    cmap = CouplingMap(couplinglist=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)])
    qc = QuantumCircuit(8, name="QAOA_p1_layout")
    for i in range(0,8,2):
        qc.cz(i, i+1); qc.rz(0.8, i); qc.rz(0.8, i+1)
    for q in range(8):
        qc.sx(q); qc.x(q); qc.sx(q)
    tqc = transpile(qc, basis_gates=["rz","sx","x","cz"], coupling_map=cmap,
                    optimization_level=3, seed_transpiler=42)
    tqc.draw(output="mpl", fold=40, filename=str(FIGDIR/"qaoa_circuit_torino.pdf"))
    meta = {"basis": ["rz","sx","x","cz"], "seed_transpiler": 42,
            "depth": tqc.depth(), "size": tqc.size()}
    (FIGDIR/"qaoa_circuit_torino.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    k16_abstract_pdf()
    torino_layout_pdf()
    print("[OK] Wrote figures/K16_circuit.pdf and figures/qaoa_circuit_torino.pdf")
