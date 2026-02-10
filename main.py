# main.py
import numpy as np
from spin_mapper import load_magnetic_field, normalize_vectors
from hamiltonian import total_hamiltonian, compute_gradients
from optimizer import riemannian_projected_descent

# Load data
Br, Bt, Bp = load_magnetic_field('../data/grmhd/dump000.h5')
Sx, Sy, Sz = normalize_vectors(Br, Bt, Bp)

# Set parameters
h_i = np.zeros_like(Sz)
h_i[0] = 0.5  # frame-dragging strength at horizon

print("Initial state:")
H, comps = total_hamiltonian(Sx, Sy, Sz, Br, h_i)
print(f"  H = {H:.6f}")

# Optimization loop
eta = 0.1
n_iter = 20

for it in range(n_iter):
    # Compute gradients
    grad_Sx, grad_Sy, grad_Sz = compute_gradients(Sx, Sy, Sz, Br, h_i)
    
    # Update spins
    Sx, Sy, Sz = riemannian_projected_descent(Sx, Sy, Sz, grad_Sx, grad_Sy, grad_Sz, eta)
    
    # Recompute energy
    H, comps = total_hamiltonian(Sx, Sy, Sz, Br, h_i)
    
    if it % 5 == 0:
        print(f"Iter {it}: H = {H:.6f}, Sz[0] = {Sz[0]:.4f}")

print("\nFinal state:")
print(f"  Sx[0], Sy[0], Sz[0] = {Sx[0]:.4f}, {Sy[0]:.4f}, {Sz[0]:.4f}")
print(f"  Final H = {H:.6f}")