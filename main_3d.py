# src/main_3d.py
import numpy as np
from QUANTUM_BH.src.load_vtu import load_harm_data
from hamiltonian import total_hamiltonian, compute_gradients
from optimizer import riemannian_projected_descent

# Load data using new loader
data = load_harm_data()

# Extract magnetic field components
Br = data['Br']
Bt = data['Bt']
Bp = data['Bp']

# Normalize spins (only at valid points)
mag = np.sqrt(Br**2 + Bt**2 + Bp**2)
mag[mag == 0] = 1e-15
Sx = Br / mag
Sy = Bt / mag
Sz = Bp / mag

print(f"Initial spins: Sx[0]={Sx[0]:.3f}, Sy[0]={Sy[0]:.3f}, Sz[0]={Sz[0]:.3f}")

# Set frame-dragging parameters
h_i = np.zeros_like(Sz)
h_i[0] = 0.5  # Only at first valid point

# Compute initial energy
H, comps = total_hamiltonian(Sx, Sy, Sz, Br, h_i)
print(f"Initial H = {H:.6f}")

# Optimization loop
eta = 0.1
n_iter = 20

for it in range(n_iter):
    grad_Sx, grad_Sy, grad_Sz = compute_gradients(Sx, Sy, Sz, Br, h_i)
    Sx, Sy, Sz = riemannian_projected_descent(Sx, Sy, Sz, grad_Sx, grad_Sy, grad_Sz, eta)
    
    H, comps = total_hamiltonian(Sx, Sy, Sz, Br, h_i)
    
    if it % 5 == 0:
        print(f"Iter {it}: H = {H:.6f}, Sz[0] = {Sz[0]:.4f}")

print(f"\nFinal spins: Sx[0]={Sx[0]:.3f}, Sy[0]={Sy[0]:.3f}, Sz[0]={Sz[0]:.3f}")
print(f"Final H = {H:.6f}")