import json
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit import transpile

# ---- CARICA SEGRETI ----
with open("ibm_secrets.json") as f:
    secrets = json.load(f)

token = secrets["token"]
channel = secrets.get("channel", "ibm_cloud")
instance = secrets.get("instance", None)

service = QiskitRuntimeService(channel=channel, token=token, instance=instance)

# ---- SCEGLI BACKEND ----
backend = service.backend("ibm_torino")
print("Uso backend:", backend.name, "| Qubit:", backend.num_qubits)

# ---- HAMILTONIANO ISING 2-QUBIT ----
op = SparsePauliOp.from_list([
    ("ZZ", -1.0),
    ("ZI", -0.5),
    ("IZ", -0.8),
])

# ---- Sampler + QAOA ----
sampler = Sampler(mode=backend)
qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=20), reps=1)

# NB: compute_minimum_eigenvalue costruisce circuiti interni -> dobbiamo forzare transpilation
# Forse Qiskit 1.x non lo fa in automatico, quindi: transpiler_levels per sicurezza
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
qaoa._pass_manager = pm  # override interno

result = qaoa.compute_minimum_eigenvalue(operator=op)

print("Energia stimata:", result.eigenvalue)
print("Parametri ottimali:", result.optimal_point)
