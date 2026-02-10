import warnings
# Suppress scipy sparse matrix warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scipy.sparse.linalg')

from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer import AerSimulator
from qiskit import transpile

# For Qiskit 2.2.1, use StatevectorSampler
from qiskit.primitives import StatevectorSampler

# ---- HAMILTONIANO TOY MODEL (2 qubit Ising) ----
op = SparsePauliOp.from_list([
    ("ZZ", -1.0),
    ("ZI", -0.5),
    ("IZ", -0.8),
])

print("Hamiltonian operator:", op)

# ==============================
# TEST 1: Energia con QAOA + StatevectorSampler
# ==============================
print("\n=== TEST 1: Energia stimata con QAOA + StatevectorSampler ===")

sampler = StatevectorSampler()
qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=20), reps=1)

result = qaoa.compute_minimum_eigenvalue(operator=op)

print(f"Energia stimata: {result.eigenvalue}")
print(f"Parametri ottimali: {result.optimal_point}")

# ========================================
# TEST 2: Bitstring con AerSimulator locale
# ========================================
print("\n=== TEST 2: Distribuzione bitstring con AerSimulator ===")

backend = AerSimulator()

# Get optimal parameters using the statevector sampler
qaoa_opt = QAOA(sampler=StatevectorSampler(), optimizer=COBYLA(maxiter=10), reps=1)
result_opt = qaoa_opt.compute_minimum_eigenvalue(operator=op)

print(f"Optimal parameters: {result_opt.optimal_point}")

# Create the circuit with optimal parameters and run on AerSimulator
circuit = qaoa_opt.ansatz
circuit.assign_parameters(result_opt.optimal_point, inplace=True)
circuit.measure_all()

# Transpile and run with shots
circuit_transpiled = transpile(circuit, backend=backend)
job = backend.run(circuit_transpiled, shots=1024)
result_counts = job.result().get_counts()

print(f"Distribuzione bitstring: {result_counts}")

# Analysis
sorted_counts = dict(sorted(result_counts.items(), key=lambda item: item[1], reverse=True))
print(f"Most probable states: {list(sorted_counts.items())[:5]}")

total_shots = sum(result_counts.values())
print(f"\nTotal shots: {total_shots}")
for state, count in list(sorted_counts.items())[:3]:
    prob = count / total_shots
    print(f"State {state}: {count} shots ({prob:.3f} probability)")

# Interpret the results
print(f"\n=== INTERPRETATION ===")
print("The QAOA found the ground state energy of:", result.eigenvalue)
print("The most probable bitstring configurations are:", list(sorted_counts.keys())[:3])
print("These correspond to the quantum states that minimize the Ising Hamiltonian.")