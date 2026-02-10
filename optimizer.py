import numpy as np

def riemannian_projected_descent(Sx, Sy, Sz, grad_Sx, grad_Sy, grad_Sz, eta=0.1):
    """
    Perform one step of Riemannian gradient descent on S^2
    """
    # Project gradient onto tangent space: g_perp = g - (g·σ) σ
    dot = Sx * grad_Sx + Sy * grad_Sy + Sz * grad_Sz
    proj_grad_Sx = grad_Sx - dot * Sx
    proj_grad_Sy = grad_Sy - dot * Sy
    proj_grad_Sz = grad_Sz - dot * Sz

    # Update
    new_Sx = Sx - eta * proj_grad_Sx
    new_Sy = Sy - eta * proj_grad_Sy
    new_Sz = Sz - eta * proj_grad_Sz

    # Re-normalize to unit length
    mag = np.sqrt(new_Sx**2 + new_Sy**2 + new_Sz**2)
    # Avoid division by zero
    mag = np.where(mag == 0, 1e-15, mag)
    new_Sx /= mag
    new_Sy /= mag
    new_Sz /= mag

    return new_Sx, new_Sy, new_Sz