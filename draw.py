# file: draw_qaoa_circuit.py
import os, json
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import circuit_drawer, plot_circuit_layout

# (opzionale, se vuoi mostrare anche la versione transpiled su Torino)
from qiskit_ibm_runtime import QiskitRuntimeService

# ---- 0) Carica i parametri (se hai già lanciato l'hardware script) ----
os.makedirs("figures", exist_ok=True)
params_path = "figures/qaoa_best_params.json"
if os.path.exists(params_path):
    with open(params_path) as f:
        P = json.load(f)
    gamma, beta = float(P["gamma"]), float(P["beta"])
else:
    # fallback: valori sensati per il toy Ising
    gamma, beta = 3.1416, 0.0
    print("ATTENZIONE: non ho trovato figures/qaoa_best_params.json, uso gamma=π, beta=0.")

# ---- 1) Definisci l'ansatz QAOA p=1 per H = -ZZ -0.5 ZI -0.8 IZ ----
def qaoa_ansatz(gamma, beta):
    qc = QuantumCircuit(2, 2)
    # Stato iniziale |+>^2
    qc.h([0, 1])
    # ZZ: implementiamo e^{-i gamma * ( - Z0 Z1 ) } = RZZ(2*gamma) (con segno gestito nel coeff)
    qc.cx(0, 1)
    qc.rz(2 * gamma, 1)
    qc.cx(0, 1)
    # Campi locali: -0.5 Z0 -0.8 Z1 -> RZ su ciascun qubit
    qc.rz(2 * 0.5 * gamma, 0)   # coeff 0.5
    qc.rz(2 * 0.8 * gamma, 1)   # coeff 0.8
    # Mixer: RX su entrambi
    qc.rx(2 * beta, [0, 1])
    # Misura per completezza
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc

qc = qaoa_ansatz(gamma, beta)

# ---- 2) Disegna e salva il circuito "logico" (non transpiled) ----
circuit_drawer(qc, output="mpl", filename="figures/qaoa_circuit.png")
print("Salvato: figures/qaoa_circuit.png")

# ---- 3) (Opzionale ma consigliato) Versione transpiled su ibm_torino + layout ----
try:
    with open("ibm_secrets.json") as f:
        secrets = json.load(f)
    service = QiskitRuntimeService(
        channel=secrets["channel"], token=secrets["token"], instance=secrets["instance"]
    )
    backend = service.backend("ibm_torino")

    tqc = transpile(qc.remove_final_measurements(inplace=False), backend=backend, optimization_level=1)
    # Ri-aggiungo le misure dopo la transpile (per disegno), se vuoi:
    tqc.measure_all()

    # Disegno del circuito transpiled
    circuit_drawer(tqc, output="mpl", filename="figures/qaoa_circuit_torino.png")
    print("Salvato: figures/qaoa_circuit_torino.png")

    # Layout fisico (mappa dei qubit) — richiede matplotlib
    fig = plot_circuit_layout(tqc, backend)
    fig.savefig("figures/qaoa_layout_torino.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Salvato: figures/qaoa_layout_torino.png")
except Exception as e:
    print("Nota: non sono riuscito a fare il transpile/mappa su ibm_torino (ok se offline). Dettagli:", e)
