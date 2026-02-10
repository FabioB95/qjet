import numpy as np
Sx = np.load("../data/spins/data3D0000_Sx.npy")
print("Shape Sx:", Sx.shape)

for n in range(20,200):  # test divisori possibili
    if Sx.shape[0] % n == 0:
        print(f"{n} divides -> {Sx.shape[0]//n}")
