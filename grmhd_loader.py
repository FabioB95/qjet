# src/grmhd_loader.py
import h5py
import numpy as np

def _first_key(f, candidates):
    for k in candidates:
        if k in f:
            return k
    return None

def _find_field_array(f):
    """
    Try to locate magnetic field components in common BHAC/HARM layouts.
    Returns arrays (B1, B2, B3) with shape [Nr, Nθ, Nφ].
    """
    # Common direct keys:
    direct = [('B1', 'B2', 'B3'), ('bb1', 'bb2', 'bb3')]
    for triplet in direct:
        if all(k in f for k in triplet):
            B1 = f[triplet[0]][:]
            B2 = f[triplet[1]][:]
            B3 = f[triplet[2]][:]
            return B1, B2, B3

    # Some files store in a "prims" cube: [nprim, Nr, Nθ, Nφ]
    if 'prims' in f:
        prims = f['prims']
        # If names present, use them; else assume first 3 are B^i (adjust if needed)
        names_key = _first_key(f, ['prim_names', 'prims_names', 'names'])
        if names_key is not None:
            names = [n.decode() if hasattr(n, 'decode') else str(n) for n in f[names_key][:]]
            # Find indices for magnetic fields by name
            def idx_of(prefixes):
                for p in prefixes:
                    if p in names:
                        return names.index(p)
                return None
            iB1 = idx_of(['B1','bb1','B^1'])
            iB2 = idx_of(['B2','bb2','B^2'])
            iB3 = idx_of(['B3','bb3','B^3'])
            if None not in (iB1, iB2, iB3):
                return prims[iB1], prims[iB2], prims[iB3]
        # Fallback: assume first 3 are B-components
        return prims[0], prims[1], prims[2]

    raise KeyError("Could not find magnetic field arrays (B1,B2,B3 or prims). Inspect file keys.")

def _find_coords(f):
    # Try flat keys
    x1 = f.get('x1', None)
    x2 = f.get('x2', None)
    x3 = f.get('x3', None)
    if all(v is not None for v in (x1, x2, x3)):
        return x1[:], x2[:], x3[:]

    # Try nested groups (common in some outputs)
    for gname in ['grid', 'coords', 'geometry']:
        if gname in f:
            g = f[gname]
            x1 = g.get('x1', None)
            x2 = g.get('x2', None)
            x3 = g.get('x3', None)
            if all(v is not None for v in (x1, x2, x3)):
                return x1[:], x2[:], x3[:]

    raise KeyError("Could not find coordinate arrays x1,x2,x3.")

def _find_sqrt_g(f):
    # Try typical names
    for key in ['gdet', 'sqrt_g', 'gdetu', 'g']:
        if key in f:
            arr = f[key][:]
            # If 'g' is full metric determinant, take abs root; others are already gdet
            if key == 'g':
                return np.sqrt(np.abs(arr))
            return arr
    # Fallback: return ones; we will refine later
    return None

def load_grmhd_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        B1, B2, B3 = _find_field_array(f)
        x1, x2, x3 = _find_coords(f)
        sqrt_g = _find_sqrt_g(f)

    R, Theta, Phi = np.meshgrid(x1, x2, x3, indexing='ij')

    info = {
        'B1': np.array(B1), 'B2': np.array(B2), 'B3': np.array(B3),
        'R': R, 'Theta': Theta, 'Phi': Phi,
        'x1': np.array(x1), 'x2': np.array(x2), 'x3': np.array(x3),
        'sqrt_g': sqrt_g if sqrt_g is not None else np.ones_like(B1, dtype=float),
    }
    print("[grmhd_loader] shapes:",
          "B:", info['B1'].shape, 
          "R:", info['R'].shape,
          "sqrt_g:", info['sqrt_g'].shape if info['sqrt_g'] is not None else None)
    return info

if __name__ == "__main__":
    data = load_grmhd_data("data/grmhd/torus.out0000.0.h5")
    print("Loaded. Radial points:", len(data['x1']))
