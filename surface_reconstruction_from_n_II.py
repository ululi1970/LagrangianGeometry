
"""
Global spectral least-squares reconstruction of a surface immersion x(u,v)
from a given normal field n(u,v) and second fundamental form II.

Input:
  n  : array shape (3, Ny, Nx)  -- unit normal field
  II : array shape (3, 3, Ny, Nx) with II[a,b,:,:] = II_ab

Output:
  x_rec : array shape (3, Ny, Nx) -- reconstructed immersion (mean mode fixed)
  t1,t2 : reconstructed tangent vectors

Domain:
  Periodic grid on [0, 2π)^2 (Fourier pseudospectral)

Author: ChatGPT
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

def spectral_derivative(field, axis):
    """
    Spectral derivative on a periodic [0,2π)^2 grid.
    axis=0 -> derivative in v (rows)
    axis=1 -> derivative in u (cols)
    """
    field = np.asarray(field)
    scalar = False
    if field.ndim == 2:
        scalar = True
        field = field[None, ...]

    comps, Ny, Nx = field.shape
    F = fft2(field, axes=(1,2))

    ky = fftfreq(Ny) * Ny
    kx = fftfreq(Nx) * Nx

    if axis == 0:
        mult = (1j * ky)[:, None]
    else:
        mult = (1j * kx)[None, :]

    dF = F * mult[None, ...]
    df = ifft2(dF, axes=(1,2)).real

    if scalar:
        return df[0]
    return df

def pointwise_solve_tangents(n, II):
    """
    Solve for tangent vectors t1,t2 from n and II using Weingarten relations.
    """
    _, Ny, Nx = n.shape
    t1 = np.zeros_like(n)
    t2 = np.zeros_like(n)
    print(type(np.array([0,0,0])))
    
    dn_dv = spectral_derivative(n, axis=0)
    dn_du = spectral_derivative(n, axis=1)

    for j in range(Ny):
        for i in range(Nx):
            A = np.vstack([
                n[:,j,i],
                dn_du[:,j,i],
                dn_dv[:,j,i]
            ])

            b1 = np.array([0.0, -II[0,0,j,i], -II[1,0,j,i]])
            b2 = np.array([0.0, -II[0,1,j,i], -II[1,1,j,i]])

            try:
                t1[:,j,i] = np.linalg.solve(A, b1)
                t2[:,j,i] = np.linalg.solve(A, b2)
            except np.linalg.LinAlgError:
                t1[:,j,i], *_ = np.linalg.lstsq(A, b1, rcond=None)
                t2[:,j,i], *_ = np.linalg.lstsq(A, b2, rcond=None)

    return t1, t2

def reconstruct_x_from_t_spectral_ls(t1, t2, alpha=1e-6,
                                    reg_type='biharmonic',
                                    mean_pos=None):
    """
    Global spectral least-squares solve of
        ∂_u x = t1 ,  ∂_v x = t2
    with regularization.
    """
    comps, Ny, Nx = t1.shape

    T1h = fft2(t1, axes=(1,2))
    T2h = fft2(t2, axes=(1,2))

    ky = fftfreq(Ny) * Ny
    kx = fftfreq(Nx) * Nx
    KX, KY = np.meshgrid(kx, ky)

    K2 = KX**2 + KY**2

    if reg_type == 'tikhonov':
        S = 1.0
    elif reg_type == 'laplace':
        S = K2
    else:
        S = K2**2

    denom = K2 + alpha * S

    T1h_t = np.moveaxis(T1h, 0, -1)
    T2h_t = np.moveaxis(T2h, 0, -1)

    num = -1j * (KX[...,None] * T1h_t + KY[...,None] * T2h_t)

    denom_safe = denom.copy()
    denom_safe[0,0] = 1.0

    Xh = num / denom_safe[...,None]

    if mean_pos is None:
        Xh[0,0,:] = 0.0
    else:
        Xh[0,0,:] = mean_pos

    Xh_ifft = np.moveaxis(Xh, -1, 0)
    x_rec = ifft2(Xh_ifft, axes=(1,2)).real

    return x_rec

def reconstruct_from_n_and_II(n, II, alpha=1e-6,
                              reg_type='biharmonic',
                              mean_pos=None):
    """
    Full reconstruction pipeline.
    """
    n = np.asarray(n, dtype=float)
    II = np.asarray(II, dtype=float)

    t1, t2 = pointwise_solve_tangents(n, II)
    x_rec = reconstruct_x_from_t_spectral_ls(
        t1, t2, alpha=alpha, reg_type=reg_type, mean_pos=mean_pos
    )

    return x_rec, t1, t2
