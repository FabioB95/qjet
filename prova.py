# check_data.py
import h5py

with h5py.File('data/grmhd/dump000.h5', 'r') as f:
    print("Keys:", list(f.keys()))
    
    data = f['data'][:]  # shape (44, 57)
    print("Data shape:", data.shape)