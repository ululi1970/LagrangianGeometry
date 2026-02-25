
#!/usr/bin/env python3
"""spectral_g_solver.py

Fourier–Galerkin (pseudospectral) RK4 solver for ∂_t g = L_v g on T^3 that accepts
precomputed spectral coefficients v_hat(t) for the velocity field.

Usage:
    - Import the module and call `run_solver(...)` with a numpy array `v_hat_time` of shape
      (num_time_levels, 3, N, N, N) containing spectral coefficients for v at each integer
      time index (assumed separated by dt).
    - Or run the script directly to execute a self-test that generates a sample time-dependent
      v_hat_time and runs the solver, saving diagnostics and final result to disk.

Notes on v_hat_time interpolation:
    The RK4 integrator evaluates the RHS at times t, t+dt/2, t+dt. If your precomputed
    v_hat_time is provided at integer time levels 0,1,2,..., the solver linearly
    interpolates between v_hat_time[n] and v_hat_time[n+1] for the mid-stage evaluations.
    If you can provide v_hat_time at a finer temporal resolution matching your RK
    stage times, you can disable interpolation by setting interp=False when calling
    run_solver. By default interp=True uses linear interpolation within each step.

Outputs:
    - Saves final g (physical) and diagnostic error (if analytic solution known) to an .npz file.

Author: ChatGPT (converted to runnable script)
"""
import sys
#import numpy as np

import numpy as np

import os
import argparse
import random
import tracemalloc
import matplotlib.pyplot as plt

def build_wavenumbers(N, L=2*np.pi):
    dx = L / N
    k1d = np.fft.fftfreq(N, d=dx) * 2*np.pi  # angular wavenumbers
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    return KX, KY, KZ

def gphys_to_ghat(g_phys):
    return np.fft.fftn(g_phys, axes=(2,3,4))
def ghat_to_gphys(g_hat):
    return np.fft.ifftn(g_hat, axes=(2,3,4)).real

def vhat_to_vphys(v_hat):
    return np.fft.ifftn(v_hat, axes=(1,2,3)).real

def vphys_to_vhat(v_phys):
    return np.fft.fftn(v_phys, axes=(1,2,3))

def phys_to_hat(phys):
    return np.fft.fftn(phys)

def hat_to_phys(hat):
    return np.fft.ifftn(hat).real

def min_eig_field(gphys):
    # gphys shape (3,3,N,N,N)
    N = gphys.shape[2]
    mats = gphys.reshape(9, N, N, N).transpose(1,2,3,0).reshape(-1,9)
    mats = mats.reshape(-1,3,3)
    eigs = np.linalg.eigvalsh(mats)
    return eigs[:,0].min()

def determinant_field(gphys):
    # gphys shape (3,3,N,N,N)
    N = gphys.shape[2]
    mats = gphys.reshape(9, N, N, N).transpose(1,2,3,0).reshape(-1,9)
    mats = mats.reshape(-1,3,3)
    dets = np.linalg.det(mats).reshape(N,N,N)
    return dets.min(), dets.max(), dets.mean()

def plot_eigenvalue_slice_2d(k, z_index=0, L=2*np.pi, cmap="viridis"):
    """
    Plot a 2D slice k(x,y,z_index) as a heatmap.

    Args:
        k        : numpy array shape (N,N,N) (e.g. k1 or k2)
        z_index  : fixed index in z-direction (default 0)
        L        : domain size (default 2*pi)
        cmap     : matplotlib colormap
    """
    N = k.shape[0]

    # physical coordinates (optional but recommended)
    try:
        x = np.linspace(0, L, N, endpoint=False).get()
    except:
        x = np.linspace(0, L, N, endpoint=False)
    try:
        y = np.linspace(0, L, N, endpoint=False).get()
    except:
        y = np.linspace(0, L, N, endpoint=False)

    slice2d = k[:, :, z_index]
    # print(type(slice2d ))
    # sys.exit()
    plt.figure()
    plt.imshow(
        slice2d.T,                 # transpose so x,y axes look natural
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect="equal",
        cmap=cmap
    )
    plt.colorbar(label="eigenvalue")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Eigenvalue slice k(x,y,{z_index})")
    plt.show()


def plot_eigenvalue_slice(k, i=None, j=None):
    """
    Plot eigenvalue k along a slice (i, j, 0).

    Args:
        k : numpy array of shape (N,N,N)  (e.g. k1 or k2)
        i : fixed x-index (int), plots k[i, :, 0]
        j : fixed y-index (int), plots k[:, j, 0]

    Exactly one of i or j must be specified.
    """
    N = k.shape[0]

    if (i is None and j is None) or (i is not None and j is not None):
        raise ValueError("Specify exactly one of i or j.")

    if i is not None:
        data = k[i, :, 0]
        
        try:
            x = np.arange(N).get()
        except:
            x = np.arange(N)
        xlabel = "y-index"
        title = f"Eigenvalue slice k[{i}, y, 0]"
    else:
        data = k[:, j, 0]
        try:
            x = np.arange(N).get()
        except:
            x = np.arange(N)
        xlabel = "x-index"
        title = f"Eigenvalue slice k[x, {j}, 0]"

    plt.figure()
    plt.plot(x, data)
    plt.xlabel(xlabel)
    plt.ylabel("eigenvalue")
    plt.title(title)
    plt.grid(True)
    plt.show()

def eigenvalues_second_fundamental_form(S):
    """
    Compute eigenvalues of a symmetric 2x2 tensor field pointwise.

    Args:
        S: numpy array shape (2,2,N,N,N), symmetric tensor field

    Returns:
        k1, k2: numpy arrays shape (N,N,N), eigenvalues
                ordered so that k1 >= k2 pointwise
    """
    # extract components
    S11 = S[0,0]
    S22 = S[1,1]
    S12 = 0.5 * (S[0,1] + S[1,0])  # enforce symmetry numerically

    # trace and discriminant
    tr = S11 + S22
    diff = S11 - S22
    disc = np.sqrt(0.25 * diff**2 + S12**2)

    # eigenvalues
    k1 = 0.5 * tr + disc
    k2 = 0.5 * tr - disc

    return k1, k2


def covariant_laplacian(f_phys, g_phys, L=2*np.pi):
    """
    Compute Laplace-Beltrami (rough Laplacian) Δ f = 1/sqrt(g) ∂_i( sqrt(g) g^{ij} ∂_j f )
    for a scalar f on a periodic cubic grid using spectral derivatives.

    Args:
        f_phys: numpy array shape (N,N,N), real scalar field
        g_phys: numpy array shape (3,3,N,N,N), real metric components g_{ij}
        L:      domain length (default 2*pi)

    Returns:
        lap_phys: numpy array shape (N,N,N), real Laplace-Beltrami Δ f
    """
    # sizes and wavenumbers
    _, _, N, _, _ = g_phys.shape
    dx = L / N
    k1d = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    ik = [1j * KX, 1j * KY, 1j * KZ]

    # --- 1) first derivatives of f (spectral)
    f_hat = np.fft.fftn(f_phys)                     # (N,N,N)
    df_hat = [ik[m] * f_hat for m in range(3)]      # list length 3
    df_phys = np.array([np.fft.ifftn(df_hat[m]).real for m in range(3)])  # shape (3,N,N,N)

    # --- 2) inverse metric g^{ij} and sqrt(det g)
    # reshape to invert pointwise: mats shape (N^3,3,3)
    mats = g_phys.reshape(9, N, N, N).transpose(1,2,3,0).reshape(-1, 3, 3)
    # compute inverse and determinant
    invs = np.linalg.inv(mats)                       # shape (N^3,3,3)
    dets = np.linalg.det(mats)                       # shape (N^3,)
    # reshape back to (3,3,N,N,N) and (N,N,N)
    ginv = invs.reshape(N, N, N, 3, 3).transpose(3,4,0,1,2)  # (3,3,N,N,N)
    sqrtg = np.sqrt(np.maximum(dets, 0.0)).reshape(N, N, N)  # ensure non-negative (numerical safety)

    # guard against zero determinant
    eps = 1e-14
    sqrtg = np.where(sqrtg < eps, eps, sqrtg)

    # --- 3) compute contravariant vector v^i = g^{i j} ∂_j f
    # use einsum: 'ijxyz, jxyz -> ixyz'
    # ginv is (i,j,x,y,z), df_phys is (j,x,y,z) -> result (i,x,y,z)
    v_phys = np.einsum('ijxyz,jxyz->ixyz', ginv, df_phys)  # shape (3,N,N,N)

    # --- 4) multiply by sqrtg: w^i = sqrtg * v^i
    w_phys = v_phys * sqrtg[None, ...]   # shape (3,N,N,N)

    # --- 5) divergence ∂_i w^i using spectral derivatives
    # transform w to spectral (over spatial axes)
    w_hat = np.fft.fftn(w_phys, axes=(1,2,3))   # shape (3,N,N,N) complex
    # compute ∂_i w^i in spectral: sum_i ik[i] * w_hat[i]
    div_hat = ik[0] * w_hat[0] + ik[1] * w_hat[1] + ik[2] * w_hat[2]  # shape (N,N,N) complex
    div_phys = np.fft.ifftn(div_hat).real

    # --- 6) Laplace-Beltrami: Δ f = (1 / sqrtg) * div_phys
    lap_phys = div_phys / sqrtg

    return lap_phys


def covariant_hessian(f_phys, g_phys, L=2*np.pi):
    """
    Compute covariant Hessian ∇_i ∇_j f on a periodic cubic grid via spectral derivatives.

    Args:
        f_phys:   scalar field, shape (N,N,N), real
        g_phys:   metric field, shape (3,3,N,N,N), real
        L:        domain length (default 2*pi)

    Returns:
        Hess_phys: array shape (3,3,N,N,N) with ∇_i ∇_j f (real)
    """
    # --- setup
    _, _, N, _, _ = g_phys.shape
    dx = L / N
    # angular wavenumbers matching np.fft.fftn with spacing dx
    k1d = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    ik = [1j * KX, 1j * KY, 1j * KZ]

    # --- spectral of f and spectral derivatives
    f_hat = np.fft.fftn(f_phys)                    # shape (N,N,N), complex
    # first derivatives in spectral space: ik[m] * f_hat
    df_hat = [ik[m] * f_hat for m in range(3)]
    # second derivatives spectral: ik[i]*ik[j]*f_hat
    d2f_hat = [[ik[i] * ik[j] * f_hat for j in range(3)] for i in range(3)]

    # inverse transform to physical space
    df_phys = np.array([np.fft.ifftn(df_hat[m]).real for m in range(3)])      # shape (3,N,N,N)
    d2f_phys = np.array([[np.fft.ifftn(d2f_hat[i][j]).real for j in range(3)] for i in range(3)])
    # d2f_phys shape (3,3,N,N,N) where d2f_phys[i,j] = ∂_i∂_j f

    # --- compute Christoffel symbols Gamma^k_{ij}
    # spectral g
    g_hat = np.fft.fftn(g_phys, axes=(2,3,4))     # shape (3,3,N,N,N), complex

    # derivatives of g: dg[m,i,j] = ∂_m g_{ij}
    dg = np.empty((3,3,3,N,N,N), dtype=float)     # (m,i,j, x,y,z)
    for m in range(3):
        tmp_hat = ik[m] * g_hat                   # multiply all (i,j) by ik[m]
        dg[m] = np.fft.ifftn(tmp_hat, axes=(2,3,4)).real

    # invert metric pointwise: ginv[k,l,x,y,z]
    mats = g_phys.reshape(9, N, N, N).transpose(1,2,3,0).reshape(-1, 3, 3)  # (N^3,3,3)
    invs = np.linalg.inv(mats)                   # (N^3,3,3)
    ginv = invs.reshape(N, N, N, 3, 3).transpose(3,4,0,1,2)   # (3,3,N,N,N)

    # compute Gamma^k_{ij} = 1/2 * g^{k l} * (∂_i g_{j l} + ∂_j g_{i l} - ∂_l g_{ij})
    Gamma = np.zeros((3,3,3,N,N,N), dtype=float)  # (k,i,j,...)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                # sum over l:
                tmp = np.zeros((N,N,N), dtype=float)
                for l in range(3):
                    tmp += ginv[k,l] * (dg[i,j,l] + dg[j,i,l] - dg[l,i,j])
                Gamma[k,i,j] = 0.5 * tmp

    # --- assemble covariant Hessian: ∇_i ∇_j f = ∂_i∂_j f - Γ^k_{ij} ∂_k f
    # term contraction over k: sum_k Gamma[k,i,j] * df_phys[k]
    # Use einsum for compactness: 'kij...,k...->ij...'
    term = np.einsum('kij...,k...->ij...', Gamma, df_phys)   # shape (3,3,N,N,N)
    Hess_phys = d2f_phys - term

    # symmetrize numerically (Hessian is symmetric)
    for i in range(3):
        for j in range(i+1, 3):
            avg = 0.5 * (Hess_phys[i,j] + Hess_phys[j,i])
            Hess_phys[i,j] = avg
            Hess_phys[j,i] = avg

    return Hess_phys

# def compute_rhs_hat(g_hat_local, n_local , sff_local , v_hat_local, e1, e2, ikx, iky, ikz, mask3d):
#     # g_hat_local: (3,3,N,N,N)
#     # v_hat_local: (3,N,N,N)
#     # spectral derivatives
#     # this version uses einsum for clarity. Not sure if faster than loops.
#     _,_,N,_,_ = g_hat_local.shape
#     dgx_hat = ikx * g_hat_local
#     dgy_hat = iky * g_hat_local
#     dgz_hat = ikz * g_hat_local
#     dgx = np.fft.ifftn(dgx_hat, axes=(2,3,4)).real
#     dgy = np.fft.ifftn(dgy_hat, axes=(2,3,4)).real
#     dgz = np.fft.ifftn(dgz_hat, axes=(2,3,4)).real
#     dg = np.stack([dgx, dgy, dgz], axis=0)  # shape (3,3,3,N,N,N)
#     del dgx, dgy, dgz, dgx_hat, dgy_hat, dgz_hat

#     g_phys = np.fft.ifftn(g_hat_local, axes=(2,3,4)).real   
#     # v derivatives
#     dvx_hat = ikx * v_hat_local
#     dvy_hat = iky * v_hat_local
#     dvz_hat = ikz * v_hat_local
#     dvx = np.fft.ifftn(dvx_hat, axes=(1,2,3)).real
#     dvy = np.fft.ifftn(dvy_hat, axes=(1,2,3)).real
#     dvz = np.fft.ifftn(dvz_hat, axes=(1,2,3)).real
#     dv = np.stack([dvx, dvy, dvz], axis=0)  # shape (3,3,N,N,N)
#     del dvx, dvy, dvz, dvx_hat, dvy_hat, dvz_hat
#     v_phys= np.fft.ifftn(v_hat_local, axes=(1,2,3)).real
#     # term1: v^k ∂_k g_{ij}
#     rhs_gphys = np.einsum('kxyz,kijxyz->ijxyz', v_phys, dg)
#     # g_{k j} * dv_i^k
#     del dg
#     rhs_gphys += np.einsum('ikxyz,kjxyz->ijxyz', g_phys, dv)
#     # g_{i k} * dv_j^k
#     rhs_gphys += np.einsum('kjxyz,kixyz->ijxyz', g_phys, dv)
#     del dv
#     rhs_ghat_local = gphys_to_ghat(rhs_gphys)
#     rhs_ghat_local[:,:,~mask3d] = 0.0

    
def compute_rhs_hat(g_hat_local, n_local , sff_local , v_hat_local, e1, e2, ikx, iky, ikz, mask3d):
    # g_hat_local: (3,3,N,N,N)
    # v_hat_local: (3,N,N,N)
    # spectral derivatives
    # _,_,N,_,_ = g_hat_local.shape
    # dgx_hat = ikx * g_hat_local
    # dgy_hat = iky * g_hat_local
    # dgz_hat = ikz * g_hat_local
    # dgx = np.fft.ifftn(dgx_hat, axes=(2,3,4)).real
    # dgy = np.fft.ifftn(dgy_hat, axes=(2,3,4)).real
    # dgz = np.fft.ifftn(dgz_hat, axes=(2,3,4)).real
    # g_phys = np.fft.ifftn(g_hat_local, axes=(2,3,4)).real

    # # v derivatives
    # dvx_hat = ikx * v_hat_local
    # dvy_hat = iky * v_hat_local
    # dvz_hat = ikz * v_hat_local
    # dvx = np.fft.ifftn(dvx_hat, axes=(1,2,3)).real
    # dvy = np.fft.ifftn(dvy_hat, axes=(1,2,3)).real
    # dvz = np.fft.ifftn(dvz_hat, axes=(1,2,3)).real
    # v_phys = np.fft.ifftn(v_hat_local, axes=(1,2,3)).real

    # # term1: v^k ∂_k g_{ij}
   
    # term1 = v_phys[0][None,None] * dgx + v_phys[1][None,None] * dgy + v_phys[2][None,None] * dgz
     
   
    # # term2 and term3: use small loops over k (k = 0,1,2) - cheap
    # term2 = np.zeros_like(g_phys)
    # term3 = np.zeros_like(g_phys)
    # for k in range(3):
    #     dv_k = np.stack([dvx[k], dvy[k], dvz[k]], axis=0)    # shape (i,N,N,N)
    #     # term2_{ij} += g_{k j} * dv_i^k  (broadcasting to (i,j,N,N,N))
    #     term2 += (dv_k[:, None, ...] * g_phys[k][None, ...])
    #     # term3_{ij} += g_{i k} * dv_j^k
    #     term3 += (dv_k[None, :, ...] * g_phys[:, k, ...])

   
    # rhs_gphys = term1 + term2 + term3
    
    # rhs_ghat_local = gphys_to_ghat(rhs_gphys)
    # rhs_ghat_local[:,:,~mask3d] = 0.0

    _,_,N,_,_ = g_hat_local.shape
    dgx_hat = ikx * g_hat_local
    dgy_hat = iky * g_hat_local
    dgz_hat = ikz * g_hat_local
    dgx = np.fft.ifftn(dgx_hat, axes=(2,3,4)).real
    dgy = np.fft.ifftn(dgy_hat, axes=(2,3,4)).real
    dgz = np.fft.ifftn(dgz_hat, axes=(2,3,4)).real
    dg = np.stack([dgx, dgy, dgz], axis=0)  # shape (3,3,3,N,N,N)
    del dgx, dgy, dgz, dgx_hat, dgy_hat, dgz_hat

    g_phys = np.fft.ifftn(g_hat_local, axes=(2,3,4)).real   
    # v derivatives
    dvx_hat = ikx * v_hat_local
    dvy_hat = iky * v_hat_local
    dvz_hat = ikz * v_hat_local
    dvx = np.fft.ifftn(dvx_hat, axes=(1,2,3)).real
    dvy = np.fft.ifftn(dvy_hat, axes=(1,2,3)).real
    dvz = np.fft.ifftn(dvz_hat, axes=(1,2,3)).real
    dv = np.stack([dvx, dvy, dvz], axis=0)  # shape (3,3,N,N,N)
    del dvx, dvy, dvz, dvx_hat, dvy_hat, dvz_hat
    v_phys= np.fft.ifftn(v_hat_local, axes=(1,2,3)).real
    # term1: v^k ∂_k g_{ij}
    rhs_gphys = np.einsum('kxyz,kijxyz->ijxyz', v_phys, dg)
    # g_{k j} * dv_i^k
    del dg
    rhs_gphys += np.einsum('ikxyz,kjxyz->ijxyz', g_phys, dv)
    # g_{i k} * dv_j^k
    rhs_gphys += np.einsum('kjxyz,kixyz->ijxyz', g_phys, dv)
    del dv
    rhs_ghat_local = gphys_to_ghat(rhs_gphys)
    rhs_ghat_local[:,:,~mask3d] = 0.0

    # evolution of n is ∂_t n_i = - Lie_v g(n,e1) e1 - Lie_v g(n,e2) e2 - 1/2 Lie_v g(n,n) n
    strain_tensor = 0.5 * np.einsum('ixyz,ijxyz,jxyz->xyz', n_local, rhs_gphys, n_local) # we use this later for sff evolution
    term1 = - np.einsum('ixyz,ijxyz,jxyz->xyz', n_local, rhs_gphys, e1) 
    term2 = - np.einsum('ixyz,ijxyz,jxyz->xyz', n_local, rhs_gphys, e2) 
    
    B = np.stack([term1, term2], axis=0) # shape (2,N,N,N)
    del term1, term2
    G11 = np.einsum('ixyz,ijxyz,jxyz->xyz', e1, g_phys, e1)
    G12 = np.einsum('ixyz,ijxyz,jxyz->xyz', e1, g_phys, e2)
    G22 = np.einsum('ixyz,ijxyz,jxyz->xyz', e2, g_phys, e2)
    #print(G11.min(), G11.max(), G12.min(), G12.max())
    Det = G11 * G22 - G12**2 + 1e-16  # avoid div by zero
    invG11 = G22 / Det
    invG22 = G11 / Det
    invG12 = - G12 / Det
    del Det, G11, G12, G22
    R = np.stack([invG11, invG12], axis=0)  # shape (3,N,N,N)
    
    rhs_n = np.einsum('ixyz,ixyz->xyz',R,B) * e1
    R = np.stack([invG12, invG22], axis=0)  # shape (3,N,N,N)
    rhs_n += np.einsum('ixyz,ixyz->xyz',R,B) * e2
    rhs_n += - strain_tensor * n_local
    # print("\n")
    # print(rhs_n[0,0,0,0], - rhs_gphys[0,2,0,0,0])
    # print("\n")
    # print(rhs_n[1,0,0,0], - rhs_gphys[1,2,0,0,0])
    # print("\n")
    # print(rhs_n[2,0,0,0], - 0.5 *rhs_gphys[2,2,0,0,0])
    # sys.exit()
    # evolution of sff is ∂_t sff = 1/2 Lie_v g(n,n) sff + \nabla_i\nabla_j (g(n,v)) 
    vn = np.einsum('ixyz,ijxyz,jxyz->xyz', n_local, g_phys, v_phys)  # g(n,v)
    # compute \nabla_i \nabla_j (g(n,v)) in spectral space
    Hess_vn = covariant_hessian(vn, g_phys)  # shape (3,3,N,N,N)
    # what we need is Hess_vn projected onto the 2D surface orthogonal to n
    Hess = np.zeros((2,2,N,N,N))
    for a in range(2):
        for b in range(2):
            ea = e1 if a == 0 else e2
            eb = e1 if b == 0 else e2
            # project: Hess_vn_{ij} ea^i eb^j
            Hess[a,b] = np.einsum('ixyz,ijxyz,jxyz->xyz', ea, Hess_vn, eb)
    rhs_sff = Hess + strain_tensor[None,None] * sff_local

    return rhs_ghat_local, rhs_n, rhs_sff


def rk4_step_with_timevarying_v(g_hat_local, n_local, sff_local, v0_hat, vhalf_hat, v1_hat, dt, e1,e2, ikx, iky, ikz, mask3d):
    _,_,N,_,_ = g_hat_local.shape
    k1g, k1n, k1sff = compute_rhs_hat(g_hat_local,              n_local,            sff_local,              v0_hat,    e1,e2,ikx, iky, ikz, mask3d)
    
    k2g, k2n, k2sff = compute_rhs_hat(g_hat_local + 0.5*dt*k1g, n_local+0.5*dt*k1n, sff_local+0.5*dt*k1sff, vhalf_hat, e1,e2,ikx, iky, ikz, mask3d)
    
    k3g, k3n, k3sff = compute_rhs_hat(g_hat_local + 0.5*dt*k2g, n_local+0.5*dt*k2n, sff_local+0.5*dt*k2sff, vhalf_hat, e1,e2,ikx, iky, ikz, mask3d)
    
    k4g, k4n, k4sff = compute_rhs_hat(g_hat_local + dt*k3g,     n_local+dt*k3n ,    sff_local+dt*k3sff,     v1_hat,    e1,e2,ikx, iky, ikz, mask3d)
   
    # kTotal = (k1 + 2*k2 + 2*k3 + k4)
    # print(np.max(np.abs(kTotal)))
    g_new = g_hat_local + (dt/6.0)*(k1g + 2*k2g + 2*k3g + k4g)
    
    n_new = n_local + (dt/6.0)*(k1n + 2*k2n + 2*k3n + k4n)
    sff_new = sff_local + (dt/6.0)*(k1sff + 2*k2sff + 2*k3sff + k4sff)
    return g_new, n_new, sff_new

def run_solver(g_phys0,  n, sff, dt, num_steps, e1,e2, L=2*np.pi, dealias=True, interp=True, verbose=True):
    """Run the spectral RK4 solver.

    Args:
        g_phys0: initial metric in physical space, shape (3,3,N,N,N)
        n: normal vector in physical space, shape (3,N,N,N)
        v_hat_time: precomputed spectral coefficients shape (T,3,N,N,N) where T >= num_steps+1
        dt: time step
        num_steps: number of steps to advance
        L: domain size (default 2π)
        dealias: whether to apply 2/3 rule
        interp: whether to linearly interpolate v within each step (if True)
        verbose: print diagnostics
    Returns:
        g_hat_final (spectral), g_phys_final (physical)
    """
    N = g_phys0.shape[2]
    KX, KY, KZ = build_wavenumbers(N, L=L)
    ikx = 1j * KX
    iky = 1j * KY
    ikz = 1j * KZ
    if dealias:
        Kcut = N // 3
        mask3d = np.logical_and(np.logical_and(np.abs(KX) <= Kcut, np.abs(KY) <= Kcut), np.abs(KZ) <= Kcut)
    else:
        mask3d = np.ones_like(KX, dtype=bool)
    
    # spectral initial metric
    g_hat = gphys_to_ghat(g_phys0)
    
    
    
    times = []
    diagnostics = {'min_eig':[], 'L2_err':[], 'max_err':[]}

    C = list(random.random()    for _ in range(3))
    for s in range(num_steps):
        v0 = generate_TG_v_hat_time(s*dt,N, coeff=C)
        vhalf = generate_TG_v_hat_time((s+1/2)*dt,N, coeff=C)
        v1 = generate_TG_v_hat_time((s+1)*dt,N, coeff=C)
        if not interp:
            # if not interpolating, simply use v0 for all stages (explicit assumption)
            v1 = v0
        
        g_hat, n, sff = rk4_step_with_timevarying_v(g_hat, n, sff, v0, vhalf, v1, dt, e1,e2, ikx, iky, ikz, mask3d)

        # symmetrize in physical space to remove roundoff asymmetry
        
        g_phys_tmp = ghat_to_gphys(g_hat)
        
       
        for i in range(3):
            for j in range(i+1, 3):
                avg = 0.5*(g_phys_tmp[i,j] + g_phys_tmp[j,i])
                g_phys_tmp[i,j] = avg
                g_phys_tmp[j,i] = avg
        # update spectral representation
        g_hat = gphys_to_ghat(g_phys_tmp)
       

        if verbose and (s % max(1, num_steps//10) == 0):
            # me = min_eig_field(g_phys_tmp)
            # diagnostics['min_eig'].append(me)
            times.append((s+1)*dt)
            me=0
            if verbose:
                print(f"Step {s+1}/{num_steps}: min eig(g) = {me:.6g}")
                print(f"    det(g): min={determinant_field(g_phys_tmp)[0]:.6g}, max={determinant_field(g_phys_tmp)[1]:.6g}, mean={determinant_field(g_phys_tmp)[2]:.6g}")
                _,K = compute_Ricci_from_g(g_phys_tmp) 
                print(f"    min Gauss curvature R = {K.min():.6g}, max R = {K.max():.6g}")
                nNorm33 = np.einsum('ixyz,ijxyz,jxyz->xyz', n, g_phys_tmp, n)
                print(f"    Normal vector norm min={nNorm33.min():.6g}, max={nNorm33.max():.6g}, mean={nNorm33.mean():.6g}")
                nNorm31 = np.einsum('ixyz,ijxyz,jxyz->xyz', n, g_phys_tmp, e1)
                print(f"    Normal-E1 inner product min={nNorm31.min():.6g}, max={nNorm31.max():.6g}, mean={nNorm31.mean():.6g}")
                nNorm32 = np.einsum('ixyz,ijxyz,jxyz->xyz', n, g_phys_tmp, e2)
                print(f"    Normal-E2 inner product min={nNorm32.min():.6g}, max={nNorm32.max():.6g}, mean={nNorm32.mean():.6g}")
                nNorm12 = np.einsum('ixyz,ijxyz,jxyz->xyz', e1, g_phys_tmp, e2)
                print(f"    E1-E2 inner product min={nNorm12.min():.6g}, max={nNorm12.max():.6g}, mean={nNorm12.mean():.6g}")
                k1,k2 = eigenvalues_second_fundamental_form(sff)
                print(f"    Max abs principal curvatures k1,k2: max|k1|={np.abs(k1).max():.6g}, max|k2|={np.abs(k2).max():.6g}")
                print(f"    Gauss curvature K=k1*k2: min K={ (k1*k2).min():.6g}, max K={(k1*k2).max():.6g}")
    g_phys_final = ghat_to_gphys(g_hat)
    k1,k2 = eigenvalues_second_fundamental_form(sff)
    print(n[:,0,0,0])
    try:
        plot_eigenvalue_slice_2d(np.asnumpy(k1), z_index=N//2)
        plot_eigenvalue_slice_2d(np.asnumpy(k2), z_index=N//2)
        plot_eigenvalue_slice_2d(nNorm31.get(), z_index=N//2)
    except:
        plot_eigenvalue_slice_2d(k1, z_index=N//2)
        plot_eigenvalue_slice_2d(k2, z_index=N//2)
        plot_eigenvalue_slice_2d(nNorm31, z_index=N//2)
    return g_hat, g_phys_final, diagnostics

def generate_TG_v_hat_time(time, N, visc = 0.0, k = 1, coeff = tuple(random.random() for _ in range(3))):

    """Generate Taylor-Green vortex spectral v_hat_time for testing."""
    # produce v_hat_time shape (num_steps+1, 3, N, N, N)
    v_phys_time = np.zeros((3, N, N, N))
    A,B,C = coeff
    v_phys_time[0] =  A * np.sin(k * np.linspace(0, 2*np.pi, N, endpoint=False)[None,None,:]) + C * np.cos(k * np.linspace(0, 2*np.pi, N, endpoint=False)[None,:,None]) 
    v_phys_time[1] =  B * np.sin(k * np.linspace(0, 2*np.pi, N, endpoint=False)[:,None,None]) + A * np.cos(k * np.linspace(0, 2*np.pi, N, endpoint=False)[None,None,:])
    v_phys_time[2] =  C * np.sin(k * np.linspace(0, 2*np.pi, N, endpoint=False)[None,:,None]) + B * np.cos(k * np.linspace(0, 2*np.pi, N, endpoint=False)[:,None,None])
    v_phys_time *= np.exp(-visc * k**2 * time)
    v_hat_time = np.fft.fftn(v_phys_time, axes=(1,2,3))  # add time axis

    return v_hat_time


def generate_sample_v_hat_time(num_steps, N, variation='linear'):
    """Generate a simple time-varying spectral v_hat_time for testing."""
    # produce v_hat_time shape (num_steps+1, 3, N, N, N)
    v_hat_time = np.zeros((num_steps+1, 3, N, N, N), dtype=complex)
    # choose base constant Fourier mode (zero mode) and modulate it
    base = np.array([0.3, -0.5, 0.7])
    for n in range(num_steps+1):
        t = n / float(num_steps)
        if variation == 'linear':
            a = 1.0 + 0.5 * t
            vec = base * a
        elif variation == 'osc':
            vec = base * (1.0 + 0.3*np.sin(2*np.pi*t))
        else:
            vec = base
        # set zero mode spectral coefficient (note FFT normalization: fft(const) = const * N^3)
        v_hat_time[n,0,0,0,0] = vec[0] * (N**3)
        v_hat_time[n,1,0,0,0] = vec[1] * (N**3)
        v_hat_time[n,2,0,0,0] = vec[2] * (N**3)
    return v_hat_time

def build_initial_condition(N):
    L = 2*np.pi
    dx = L / N
    x = np.arange(N) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    g_phys0 = np.zeros((3,3,N,N,N), dtype=float)
    g_phys0[0,0] = 1.0#+ 0.15 * np.sin(X) * np.cos(Y)
    g_phys0[1,1] = 1.0# + 0.10 * np.cos(Y) * np.sin(Z)
    g_phys0[2,2] = 1.0# + 0.08 * np.cos(Z) * np.sin(X)
    # g_phys0[0,1] = 0.06 * np.sin(Z)
    # g_phys0[0,2] = 0.04 * np.cos(Y)
    # g_phys0[1,2] = 0.05 * np.sin(X)
    # g_phys0[1,0] = g_phys0[0,1]
    # g_phys0[2,0] = g_phys0[0,2]
    # g_phys0[2,1] = g_phys0[1,2]
    # initial orthonormal frame fields
    e1 = np.zeros((3,N,N,N))
    e2 = np.zeros((3,N,N,N))
    n = np.zeros((3,N,N,N))
    e1[0] = 1.0
    e2[1] = 1.0
    n[2] = 1.0
    # initialize the second fundamental form (this is a rank two tensor field on the 2D surfaces orthogonal to n)
    sff = np.zeros((2,2,N,N,N))

    return g_phys0,e1,e2,n, sff

def main_cli():
    parser = argparse.ArgumentParser(description='Spectral RK4 solver for ∂_t g = L_v g on T^3 with time-varying v_hat.')
    parser.add_argument('--N', type=int, default=16, help='grid size per direction (power of two recommended).')
    parser.add_argument('--dt', type=float, default=0.01, help='time step.')
    parser.add_argument('--steps', type=int, default=40, help='number of RK4 steps.')
    
    parser.add_argument('--out', type=str, default='g_solver_output.npz', help='output filename (.npz) to save final g and diagnostics.')
    args = parser.parse_args()

    N = args.N
    dt = args.dt
    steps = args.steps

    # build initial metric
    g_phys0,e1,e2,n , sff = build_initial_condition(N)

    

    # run solver
    
    #tracemalloc.start()
    g_hat_final, g_phys_final, diagnostics = run_solver(g_phys0, n, sff, dt, steps, e1,e2, L=2*np.pi, dealias=True, interp=True, verbose=True)
    #current, peak = tracemalloc.get_traced_memory()
    #print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    #tracemalloc.stop()
    # save outputs
    np.savez(args.out, g_phys_final=g_phys_final, diagnostics=diagnostics)
    print(f"Saved final metric and diagnostics to {args.out}")


def compute_Ricci_from_g(g_phys, L=2*np.pi):
    """
    Compute the Ricci tensor from a metric field on a periodic cubic grid using spectral derivatives.

    Args:
        g_phys: numpy array shape (3,3,N,N,N) with g_phys[i,j,x,y,z] (real)
        L: domain size in each direction (default 2*pi)

    Returns:
        Ric_phys: numpy array shape (3,3,N,N,N) (real) with Ricci tensor components Ric_{ij}(x)
    Notes:
        - Uses FFT-based spectral derivatives: ideal for smooth periodic data on T^3.
        - Complexity dominated by a small number of 3D FFTs of 3x3 (and 3x3x3) component arrays.
    """
    # dims
    _, _, N, _, _ = g_phys.shape
    dx = L / N

    # build angular wavenumbers (compatible with np.fft.fftn using spacing dx)
    k1d = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    ik = [1j * KX, 1j * KY, 1j * KZ]

    # 1) compute spectral representations of metric components
    # fft over the last three axes
    g_hat = gphys_to_ghat(g_phys)  # shape (3,3,N,N,N)

    # 2) compute partial_m g_{ij} (for m = 0,1,2)
    # dg[m, i, j, x, y, z] = ∂_m g_{ij}
    dg = np.empty((3, 3, 3, N, N, N), dtype=float)
    for m in range(3):
        tmp_hat = ik[m] * g_hat            # multiply each (i,j) component in spectral space
        dg[m] = np.fft.ifftn(tmp_hat, axes=(2,3,4)).real

    # 3) compute inverse metric g^{ij} at each grid point (vectorized via reshape)
    # reshape to (num_points, 3,3), invert, reshape back
    mats = g_phys.reshape(9, -1).T.reshape(-1, 3, 3)   # shape (N^3, 3,3)
    invs = np.linalg.inv(mats)                         # shape (N^3,3,3)
    ginv = invs.reshape(N, N, N, 3, 3).transpose(3,4,0,1,2)  # shape (3,3,N,N,N)
    # reorder to (3,3,N,N,N) as expected (k,l, x,y,z)
    # currently ginv[k,l,x,y,z]

    # 4) compute Christoffel symbols Gamma^k_{ij} = 1/2 g^{k l} (∂_i g_{j l} + ∂_j g_{i l} - ∂_l g_{ij})
    # We'll build S_{i j l} = ∂_i g_{j l} + ∂_j g_{i l} - ∂_l g_{i j}
    # dg has index order dg[m,i,j,...] = ∂_m g_{ij}
    S = np.empty((3,3,3,N,N,N), dtype=float)  # S[i,j,l,...]
    for i in range(3):
        for j in range(3):
            for l in range(3):
                S[i, j, l] = dg[i, j, l] + dg[j, i, l] - dg[l, i, j]

    # contract ginv[k,l,...] with S[i,j,l,...] to get Gamma[k,i,j,...]
    # gamma[k,i,j] = 0.5 * sum_l ginv[k,l] * S[i,j,l]
    Gamma = 0.5 * np.einsum('klxyz,ijlxyz->kijxyz', ginv, S)  # careful indexing in einsum below

    # NOTE: np.einsum call above uses a compact notation; to avoid subtle mistakes with einsum labels across
    # many axes we expand with explicit loops (cheap since loops are over 3 indices only)
    # Gamma = np.zeros((3,3,3,N,N,N), dtype=float)  # Gamma[k,i,j]
    # for k in range(3):
    #     for i in range(3):
    #         for j in range(3):
    #             # sum over l
    #             tmp = np.zeros((N,N,N), dtype=float)
    #             for l in range(3):
    #                 tmp += ginv[k,l] * S[i,j,l]
    #             Gamma[k,i,j] = 0.5 * tmp

    # 5) compute spectral derivatives of Gamma: ∂_p Gamma^k_{ij}
    Gamma_hat = np.fft.fftn(Gamma, axes=(3,4,5))  # fft on last 3 axes (x,y,z)
    dGamma = [None, None, None]   # list of length 3, dGamma[p] has shape (3,3,3,N,N,N)
    for p in range(3):
        dGamma[p] = np.fft.ifftn(ik[p] * Gamma_hat, axes=(3,4,5)).real

    # 6) assemble Ricci using formula:
    # Ric_{ij} = ∂_k Γ^k_{ij} - ∂_i Γ^k_{k j} + Γ^k_{k m} Γ^m_{i j} - Γ^k_{i m} Γ^m_{k j}
    # note that the second and third terms should be zero if |g| is constant
    # We compute each term with small-index loops (k and m in 0..2)
    Ric = np.zeros((3,3,N,N,N), dtype=float)
    # First term: sum_k ∂_k Γ^k_{ij}  -> for each k use dGamma[k][k,i,j]
    term1 = np.zeros_like(Ric)
    for k in range(3):
        term1 += dGamma[k][k]   # dGamma[k] is array (k,i,j,...); dGamma[k][k] picks (i,j,...)

    # Second term: ∂_i Γ^k_{k j}: for each i we need sum_k dGamma[i][k,k,j]
    # term2 = np.zeros_like(Ric)
    # for i in range(3):
    #     tmp = np.zeros((N,N,N), dtype=float)
    #     for k in range(3):
    #         tmp += dGamma[i][k,k]   # dGamma[i] has shape (k,i,j,...) so index [k,k,j] gives (j,...) aligned
    #     term2[i] = tmp  # term2[i,j,...] but care: assign across j index; we'll build explicitly below

    # The previous loop made a shape mismatch; do explicit double loops (safe and clear)
    term2 = np.zeros_like(Ric)
    for i in range(3):
        for j in range(3):
            tmp = np.zeros((N,N,N), dtype=float)
            for k in range(3):
                tmp += dGamma[i][k,k,j]  # dGamma[p][k,i,j]
            term2[i,j] = tmp

    # Third term: A_m = Γ^k_{k m}  (sum over k)
    A = np.zeros((3,N,N,N), dtype=float)   # A[m, ...]
    for m in range(3):
        for k in range(3):
            A[m] += Gamma[k,k,m]

    term3 = np.zeros_like(Ric)
    for i in range(3):
        for j in range(3):
            tmp = np.zeros((N,N,N), dtype=float)
            for m in range(3):
                tmp += A[m] * Gamma[m,i,j]
            term3[i,j] = tmp

    # Fourth term: sum_{k,m} Γ^k_{i m} Γ^m_{k j}
    term4 = np.zeros_like(Ric)
    for i in range(3):
        for j in range(3):
            tmp = np.zeros((N,N,N), dtype=float)
            for k in range(3):
                for m in range(3):
                    tmp += Gamma[k,i,m] * Gamma[m,k,j]
            term4[i,j] = tmp

    Ric = term1 - term2 + term3 - term4

    # ensure symmetry of Ric (numerical roundoff)
    for i in range(3):
        for j in range(i+1,3):
            avg = 0.5 * (Ric[i,j] + Ric[j,i])
            Ric[i,j] = avg
            Ric[j,i] = avg

    GaussCurv = np.einsum('ikxyz,kixyz->xyz', ginv, Ric)/3  # R = g^{ij} Ric_{ij}
    return Ric, GaussCurv


if __name__ == '__main__':
    main_cli()
