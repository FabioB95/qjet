import json
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit, transpile
import numpy as np

# ==========================
# 1. Connessione a IBM
# ==========================
with open("ibm_secrets.json") as f:
    secrets = json.load(f)

service = QiskitRuntimeService(
    channel=secrets["channel"],
    token=secrets["token"],
    instance=secrets["instance"],
)

backend = service.backend("ibm_torino")  # hardware reale
print(f"Backend: {backend.name} | qubits: {backend.num_qubits}")

# ==========================
# 2. Definizione Hamiltoniano (2-qubit Ising)
# ==========================
op = SparsePauliOp.from_list([
    ("ZZ", -1.0),
    ("ZI", -0.5),
    ("IZ", -0.8),
])
print("Hamiltoniano:", op)

# ==========================
# 3. Costruzione circuito QAOA a mano (p=1)
# ==========================
def qaoa_ansatz(gamma, beta):
    qc = QuantumCircuit(2)
    # inizializza in |+>
    qc.h([0, 1])
    # termine ZZ
    qc.cx(0, 1)
    qc.rz(2 * gamma, 1)
    qc.cx(0, 1)
    # termini singoli
    qc.rz(2 * 0.5 * gamma, 0)  # ZI coeff -0.5
    qc.rz(2 * 0.8 * gamma, 1)  # IZ coeff -0.8
    # mixer
    qc.rx(2 * beta, [0, 1])
    qc.measure_all()
    return qc

# ==========================
# 4. Funzione per stimare energia
# ==========================
def energy_from_counts(counts):
    E = 0
    for bitstring, prob in counts.items():
        z = [1 - 2*int(b) for b in bitstring[::-1]]  # mappa 0->+1, 1->-1
        term1 = -1.0 * z[0]*z[1]
        term2 = -0.5 * z[0]
        term3 = -0.8 * z[1]
        E += (term1 + term2 + term3) * prob
    return E

# ==========================
# 5. Lancio su backend reale
# ==========================
sampler = Sampler(mode=backend)

gammas = np.linspace(0, np.pi, 3)
betas  = np.linspace(0, np.pi, 3)

best_E = 999
best_params = None

for g in gammas:
    for b in betas:
        qc = qaoa_ansatz(g, b)
        tqc = transpile(qc, backend)

        job = sampler.run([tqc])
        res = job.result()[0]

        # Recupera conteggi o quasi-distribuzione
        try:
            counts = res.data.meas.get_counts()
        except Exception:
            quasi = res.data.meas.get("quasi_dist")
            counts = quasi.binary_probabilities()

        E = energy_from_counts(counts)
        print(f"gamma={g:.2f}, beta={b:.2f}, E={E:.3f}")

        if E < best_E:
            best_E = E
            best_params = (g, b)

print("\n=== RISULTATO ===")
print(f"Best energy: {best_E:.3f} con parametri gamma={best_params[0]:.3f}, beta={best_params[1]:.3f}")

# Salva i parametri ottimi per il disegno del circuito
import os, json
os.makedirs("figures", exist_ok=True)
with open("figures/qaoa_best_params.json", "w") as f:
    json.dump({"gamma": float(best_params[0]), "beta": float(best_params[1])}, f, indent=2)
print("Parametri salvati in figures/qaoa_best_params.json")
