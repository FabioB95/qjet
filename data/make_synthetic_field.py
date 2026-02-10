import numpy as np, os
os.makedirs("data/M87", exist_ok=True)
Nx, Ny, Nz = 128, 128, 128
x = np.linspace(-1,1,Nx)[:,None,None]
y = np.linspace(-1,1,Ny)[None,:,None]
z = np.linspace(0,2,Nz)[None,None,:]
# Axial-dominated field with gentle opening (parabolic)
Bz = 1.0 - 0.3*(x**2 + y**2) + 0.2*z
Bx = 0.2*x
By = 0.2*y
np.savez("data/M87/optimized_field.npz", Bx=Bx, By=By, Bz=Bz)
print("Saved data/M87/optimized_field.npz", Bx.shape)
