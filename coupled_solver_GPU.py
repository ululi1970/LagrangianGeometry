#!/usr/bin/env python3
"""
coupled_solver_GPU.py

A GPU-accelerated pseudospectral solver for the coupled evolution of a metric g 
and a one-form lambda on a triply periodic domain T^3, using CuPy.
"""

import sys
import cupy as cp
from cupy.fft import fftn, ifftn, fftfreq
from cupyx.scipy.sparse.linalg import LinearOperator, cg
import numpy as np
import time
import os
import argparse

# ==============================================================================
#           BUILDING BLOCK 1: Poisson-Beltrami Solver (GPU)
# ==============================================================================

class BeltramiPoissonSolver:
    """
    Solves the Poisson-Beltrami equation Δ_g u = f for a scalar u on T^3
    using a preconditioned conjugate gradient method on GPU.
    """
    def __init__(self, N, L=2*cp.pi):
        self.N = N
        self.shape = (N, N, N)
        self.L = L
        
        k1d = fftfreq(N, d=L/(2*cp.pi*N))
        KX, KY, KZ = cp.meshgrid(k1d, k1d, k1d, indexing='ij')
        self.K = cp.array([KX, KY, KZ])
        self.g_phys = None
        self.sqrt_g = None
        self.g_inv = None

    def set_metric(self, g_phys):
        """Sets the metric for the Poisson problem."""
        self.g_phys = cp.asarray(g_phys)
        
        # Calculate determinant and inverse metric
        # Reshape to (N^3, 3, 3) for batched linalg
        mats = self.g_phys.reshape(9, -1).T.reshape(-1, 3, 3)
        det_g = cp.linalg.det(mats).reshape(self.shape)
        invs = cp.linalg.inv(mats)
        self.g_inv = invs.reshape(self.N, self.N, self.N, 3, 3).transpose(3,4,0,1,2)
        
        self.sqrt_g = cp.sqrt(cp.maximum(det_g, 1e-16))

    def _apply_op_A(self, u_flat):
        """Applies the operator A(u) = ∂_i(sqrt(g)g^ij ∂_j u)."""
        u = u_flat.reshape(self.shape)
        
        u_hat = fftn(u)
        grad_u_hat = 1j * self.K * u_hat[None, ...]
        grad_u_phys = ifftn(grad_u_hat, axes=(1,2,3)).real

        V_phys = cp.einsum('ijxyz,jxyz->ixyz', self.sqrt_g[None,None,...] * self.g_inv, grad_u_phys)

        V_hat = fftn(V_phys, axes=(1,2,3))
        div_V_hat = cp.sum(1j * self.K * V_hat, axis=0)
        div_V_phys = ifftn(div_V_hat).real
        
        return div_V_phys.flatten()

    def get_preconditioner(self):
        g_inv_avg = cp.mean(self.g_inv, axis=(2,3,4))
        
        symbol = cp.zeros(self.shape)
        for i in range(3):
            for j in range(3):
                symbol -= self.K[i] * g_inv_avg[i,j] * self.K[j]
        
        symbol[0,0,0] = 1.0
        inv_symbol = 1.0 / symbol
        inv_symbol[0,0,0] = 0.0
        
        def prec_func(r_flat):
            r = r_flat.reshape(self.shape)
            r_hat = fftn(r)
            u_hat = r_hat * inv_symbol
            u = ifftn(u_hat).real
            return u.flatten()
        
        return LinearOperator((self.N**3, self.N**3), matvec=prec_func)

    def solve(self, f, rtol=1e-8, atol=1e-10, maxiter=200):
        if self.g_phys is None:
            raise RuntimeError("Metric not set. Call set_metric() before solving.")
            
        rhs_integ = f * self.sqrt_g
        rhs_integ_mean = cp.mean(rhs_integ)
        rhs_corrected = f - rhs_integ_mean / self.sqrt_g

        rhs_for_solver = (self.sqrt_g * rhs_corrected).flatten()
        
        A = LinearOperator((self.N**3, self.N**3), matvec=lambda v: -self._apply_op_A(v))
        M = self.get_preconditioner()
        
        u_flat, info = cg(A, -rhs_for_solver, tol=rtol, maxiter=maxiter, M=M)
        
        if info > 0:
            print(f"Warning: CG for pressure did not converge after {info} iterations")
        
        u = u_flat.reshape(self.shape)
        u -= cp.mean(u)
        return u

# ==============================================================================
#           GEOMETRIC HELPER FUNCTIONS (GPU)
# ==============================================================================

def build_wavenumbers(N, L=2*cp.pi):
    k1d = fftfreq(N, d=L/(2*cp.pi*N))
    KX, KY, KZ = cp.meshgrid(k1d, k1d, k1d, indexing='ij')
    return cp.array([KX, KY, KZ])

def compute_christoffel(g_phys, g_inv, K):
    """Computes Christoffel symbols Γ^k_{ij}."""
    ik = 1j * K
    g_hat = fftn(g_phys, axes=(2,3,4))
    
    dg_hat = ik[None, None, :, :, :, :] * g_hat[:, :, None, :, :, :]
    dg_phys = ifftn(dg_hat, axes=(3,4,5)).real

    term = dg_phys.transpose(1,2,0,3,4,5) + dg_phys.transpose(2,1,0,3,4,5) - dg_phys
    
    Gamma = 0.5 * cp.einsum('kl...,lij...->kij...', g_inv, term)
    return Gamma

def compute_connection_laplacian_1form(lambda_phys, g_phys, g_inv, K, Gamma):
    """Computes the connection Laplacian on a 1-form: (∇_k∇^k λ)_i."""
    ik = 1j * K
    
    lambda_hat = fftn(lambda_phys, axes=(1,2,3))
    d_lambda_hat = ik[None, :, :, :, :] * lambda_hat[:, None, :, :, :]
    d_lambda_phys = ifftn(d_lambda_hat, axes=(2,3,4)).real

    gamma_lambda = cp.einsum('mki...,m...->ki...', Gamma, lambda_phys)
    T_phys = d_lambda_phys.transpose(1,0,2,3,4) - gamma_lambda

    T_hat = fftn(T_phys, axes=(2,3,4))
    d_T_hat = ik[None, None, :, :, :, :] * T_hat[:, :, None, :, :, :]
    d_T_phys = ifftn(d_T_hat, axes=(3,4,5)).real
    
    gamma_T_1 = cp.einsum('mji...,mk...->kij...', Gamma, T_phys)
    gamma_T_2 = cp.einsum('mjk...,im...->kij...', Gamma, T_phys.transpose(1,0,2,3,4))

    nabla_T_phys = d_T_phys - gamma_T_1 - gamma_T_2

    result = cp.einsum('jk...,kij...->i...', g_inv, nabla_T_phys)

    return result

def compute_lie_derivative_g(g_phys, v_phys, K):
    """Computes L_v g = v^k ∂_k g_ij + g_kj ∂_i v^k + g_ik ∂_j v^k"""
    ik = 1j * K
    
    g_hat = fftn(g_phys, axes=(2,3,4))
    dg_hat = ik[None, None, :, :, :, :] * g_hat[:, :, None, :, :, :]
    dg_phys = ifftn(dg_hat, axes=(3,4,5)).real
    
    v_hat = fftn(v_phys, axes=(1,2,3))
    dv_hat = ik[None, :, :, :, :] * v_hat[:, None, :, :, :]
    dv_phys = ifftn(dv_hat, axes=(2,3,4)).real

    term1 = cp.einsum('kxyz,ijkxyz->ijxyz', v_phys, dg_phys)
    term2 = cp.einsum('kjxyz,ikxyz->ijxyz', g_phys, dv_phys)
    term3 = cp.einsum('ikxyz,jkxyz->ijxyz', g_phys, dv_phys)
    
    return term1 + term2 + term3

def compute_divergence(W_phys, sqrt_g, K):
    """Computes divergence of a contravariant vector field W^i: div(W) = 1/sqrt(g) ∂_i(sqrt(g) W^i)"""
    ik = 1j * K
    
    term_to_div = sqrt_g[None, ...] * W_phys
    term_to_div_hat = fftn(term_to_div, axes=(1,2,3))
    
    div_hat = cp.sum(ik * term_to_div_hat, axis=0)
    div_phys = ifftn(div_hat).real
    
    eps = 1e-14
    safe_sqrt_g = cp.where(sqrt_g < eps, eps, sqrt_g)
    
    return div_phys / safe_sqrt_g

# ==============================================================================
#           MAIN COUPLED SOLVER (GPU)
# ==============================================================================

class CoupledSolver:
    def __init__(self, N, L=2*cp.pi, nu=0.01):
        self.N = N
        self.shape = (N, N, N)
        self.L = L
        self.nu = nu
        self.K = build_wavenumbers(N, L)
        self.poisson_solver = BeltramiPoissonSolver(N, L)

    def get_rhs(self, g_phys, lambda_phys):
        """Computes the RHS for the coupled system (∂_t g, ∂_t λ)."""
        # --- 1. Geometric preliminaries
        mats = g_phys.reshape(9, -1).T.reshape(-1, 3, 3)
        det_g = cp.linalg.det(mats).reshape(self.shape)
        sqrt_g = cp.sqrt(cp.maximum(det_g, 1e-16))
        invs = cp.linalg.inv(mats)
        g_inv = invs.reshape(self.N, self.N, self.N, 3, 3).transpose(3,4,0,1,2)
        Gamma = compute_christoffel(g_phys, g_inv, self.K)

        # --- 2. Compute v from λ (v = λ^♯)
        v_phys = cp.einsum('ijxyz,jxyz->ixyz', g_inv, lambda_phys)

        # --- 3. Compute RHS for g: L_v g
        g_rhs = compute_lie_derivative_g(g_phys, v_phys, self.K)
        
        # --- 4. Compute RHS for λ
        
        # 4a. Viscous term: -ν * Δ_H λ = ν * (∇_k∇^k λ)
        conn_lap_lambda = compute_connection_laplacian_1form(lambda_phys, g_phys, g_inv, self.K, Gamma)
        viscous_term = self.nu * conn_lap_lambda

        # 4b. Solve for pressure p
        # Δ_g p = div(viscous_term^♯) - div((L_v g)^♯ . v)
        viscous_term_sharp = cp.einsum('ij...,j...->i...', g_inv, viscous_term)
        div_viscous_sharp = compute_divergence(viscous_term_sharp, sqrt_g, self.K)
        
        # New source term, Sp = -div(W) where W^i = g^ij (L_v g)_jk v^k
        W_vec = cp.einsum('ij...,jk...,k...->i...', g_inv, g_rhs, v_phys)
        pressure_source_term = -compute_divergence(W_vec, sqrt_g, self.K)
        
        total_pressure_rhs = div_viscous_sharp + pressure_source_term

        self.poisson_solver.set_metric(g_phys)
        p = self.poisson_solver.solve(total_pressure_rhs)
        
        # 4c. Compute pressure gradient -dp
        p_hat = fftn(p)
        grad_p_hat = 1j * self.K * p_hat[None, ...]
        grad_p_phys = ifftn(grad_p_hat, axes=(1,2,3)).real
        
        # 4d. Assemble λ_rhs
        lambda_rhs = -grad_p_phys + viscous_term
        
        return g_rhs, lambda_rhs

    def solve(self, g0_phys, lambda0_phys, dt, num_steps, verbose=True):
        """Main time-stepping loop."""
        g_phys = cp.asarray(g0_phys)
        lambda_phys = cp.asarray(lambda0_phys)

        for s in range(num_steps):
            g1, l1 = self.get_rhs(g_phys, lambda_phys)
            g2, l2 = self.get_rhs(g_phys + 0.5*dt*g1, lambda_phys + 0.5*dt*l1)
            g3, l3 = self.get_rhs(g_phys + 0.5*dt*g2, lambda_phys + 0.5*dt*l2)
            g4, l4 = self.get_rhs(g_phys + dt*g3, lambda_phys + dt*l3)
            
            g_phys += (dt/6.0) * (g1 + 2*g2 + 2*g3 + g4)
            lambda_phys += (dt/6.0) * (l1 + 2*l2 + 2*l3 + l4)

            # Ensure metric symmetry
            g_phys = 0.5 * (g_phys + g_phys.transpose(1,0,2,3,4))

            if verbose and (s % max(1, num_steps//10) == 0):
                mats = g_phys.reshape(9, -1).T.reshape(-1, 3, 3)
                try:
                    det_g = cp.linalg.det(mats)
                    invs = cp.linalg.inv(mats)
                    g_inv = invs.reshape(self.N, self.N, self.N, 3, 3).transpose(3,4,0,1,2)
                    v_phys = cp.einsum('ijxyz,jxyz->ixyz', g_inv, lambda_phys)
                    max_v = float(cp.max(cp.sqrt(cp.sum(v_phys**2, axis=0))))
                    min_det, max_det = float(det_g.min()), float(det_g.max())
                except Exception:
                    min_det, max_det, max_v = float('nan'), float('nan'), float('nan')

                print(f"Step {s+1}/{num_steps}:")
                print(f"  min/max det(g): {min_det:.3g}, {max_det:.3g}")
                print(f"  max|v|: {max_v:.3g}")
                if np.isnan(min_det):
                    print("  !!! Solver has diverged. Halting. !!!")
                    break

        return g_phys, lambda_phys

def main_cli():
    parser = argparse.ArgumentParser(description='Coupled solver for metric and one-form (GPU).')
    parser.add_argument('--N', type=int, default=16, help='Grid size.')
    parser.add_argument('--dt', type=float, default=0.005, help='Time step.')
    parser.add_argument('--steps', type=int, default=20, help='Number of steps.')
    parser.add_argument('--nu', type=float, default=0.05, help='Viscosity.')
    parser.add_argument('--out', type=str, default='coupled_solver_gpu_output.npz', help='Output filename.')
    args = parser.parse_args()

    N = args.N
    
    # Initialize on CPU and transfer to GPU
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    g0_phys = np.zeros((3,3,N,N,N), dtype=float)
    g0_phys[0,0] = 1.0
    g0_phys[1,1] = 1.0
    g0_phys[2,2] = 1.0
    
    lambda0_phys = np.zeros((3,N,N,N), dtype=float)
    lambda0_phys[0] = np.sin(X) * np.cos(Y) * np.cos(Z)
    lambda0_phys[1] = -np.cos(X) * np.sin(Y) * np.cos(Z)
    lambda0_phys[2] = 0.0

    print(f"Starting GPU-accelerated solver on N={N} grid...")
    start_time = time.time()
    
    solver = CoupledSolver(N, nu=args.nu)
    g_final, lambda_final = solver.solve(g0_phys, lambda0_phys, args.dt, args.steps)
    
    cp.cuda.Stream.null.synchronize()
    print(f"Solver finished in {time.time() - start_time:.2f}s")
    
    # Move back to CPU for saving
    np.savez(args.out, 
             g_final=g_final.get(), 
             lambda_final=lambda_final.get())
    print(f"Saved final state to {args.out}")


if __name__ == '__main__':
    main_cli()
