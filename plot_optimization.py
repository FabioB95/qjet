# src/plot_optimization.py
import numpy as np
import matplotlib.pyplot as plt

# Simulate the optimization history (since you don't save it in main_3d.py)
n_iter = 20
Sz_history = np.zeros(n_iter)
H_history = np.zeros(n_iter)

# Recreate the optimization loop to save history
from QUANTUM_BH.src.load_vtu import load_harm_data
from hamiltonian import total_hamiltonian, compute_gradients
from optimizer import riemannian_projected_descent

data = load_harm_data()
Br, Bt, Bp = data['Br'], data['Bt'], data['Bp']

# Initialize spins
mag = np.sqrt(Br**2 + Bt**2 + Bp**2)
mag[mag == 0] = 1e-15
Sx, Sy, Sz = Br/mag, Bt/mag, Bp/mag

h_i = np.zeros_like(Sz)
h_i[0] = 0.5

for it in range(n_iter):
    grad_Sx, grad_Sy, grad_Sz = compute_gradients(Sx, Sy, Sz, Br, h_i)
    Sx, Sy, Sz = riemannian_projected_descent(Sx, Sy, Sz, grad_Sx, grad_Sy, grad_Sz, 0.1)
    
    H, _ = total_hamiltonian(Sx, Sy, Sz, Br, h_i)
    Sz_history[it] = Sz[0]
    H_history[it] = H

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(Sz_history, 'o-', label='$S_z$ (vertical alignment)')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('$S_z$')
ax1.set_title('Spin Alignment with BH Axis')
ax1.grid(True)

ax2.plot(H_history, 'o-', color='red', label='Total Hamiltonian')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('H')
ax2.set_title('Energy Minimization')
ax2.grid(True)

plt.tight_layout()
plt.savefig('../figures/optimization_convergence.png', dpi=150, bbox_inches='tight')
plt.show()s