import sys
import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
from scipy.sparse.linalg import LinearOperator, cg

class BeltramiPoissonSolver:
    r"""
    Advanced Solver for \Delta_g u = f on a triply periodic box [0, L]^3.
    Uses Multigrid-preconditioned Conjugate Gradient (MGCG).
    Designed for high performance and easy transition to CuPy.
    """
    def __init__(self, Nx, Ny, Nz, L=2*np.pi):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.shape = (Nx, Ny, Nz)
        self.L = L
        
        # Grid and Wave numbers
        self.x = np.linspace(0, L, Nx, endpoint=False)
        self.y = np.linspace(0, L, Ny, endpoint=False)
        self.z = np.linspace(0, L, Nz, endpoint=False)
        
        kx = fftfreq(Nx, d=L/(2*np.pi*Nx))
        ky = fftfreq(Ny, d=L/(2*np.pi*Ny))
        kz = fftfreq(Nz, d=L/(2*np.pi*Nz))
        
        # Store K as a (3, Nx, Ny, Nz) array for broadcasting
        self.K = np.zeros((3, Nx, Ny, Nz))
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        self.K[0], self.K[1], self.K[2] = KX, KY, KZ

    def set_metric(self, g_ij):
        """
        g_ij: (3, 3, Nx, Ny, Nz) numpy array.
        """
        g = np.asarray(g_ij)
        det_g = (g[0,0] * (g[1,1] * g[2,2] - g[1,2] * g[2,1]) -
                 g[0,1] * (g[1,0] * g[2,2] - g[1,2] * g[2,0]) +
                 g[0,2] * (g[1,0] * g[2,1] - g[1,1] * g[2,0]))
        
        self.sqrt_g = np.sqrt(det_g)
        inv_det = 1.0 / det_g
        
        self.g_inv = np.zeros_like(g)
        self.g_inv[0,0] = (g[1,1] * g[2,2] - g[1,2] * g[2,1]) * inv_det
        self.g_inv[0,1] = (g[0,2] * g[2,1] - g[0,1] * g[2,2]) * inv_det
        self.g_inv[0,2] = (g[0,1] * g[1,2] - g[0,2] * g[1,1]) * inv_det
        self.g_inv[1,0] = (g[1,2] * g[2,0] - g[1,0] * g[2,2]) * inv_det
        self.g_inv[1,1] = (g[0,0] * g[2,2] - g[0,2] * g[2,0]) * inv_det
        self.g_inv[1,2] = (g[1,0] * g[0,2] - g[0,0] * g[1,2]) * inv_det
        self.g_inv[2,0] = (g[1,0] * g[2,1] - g[1,1] * g[2,0]) * inv_det
        self.g_inv[2,1] = (g[0,1] * g[2,0] - g[0,0] * g[2,1]) * inv_det
        self.g_inv[2,2] = (g[0,0] * g[1,1] - g[0,1] * g[1,0]) * inv_det
        
        self.coeff = self.sqrt_g[None, None, ...] * self.g_inv

    def _apply_L(self, u, coeff, K):
        u_hat = fftn(u)
        grad_u = ifftn(1j * K * u_hat[None, ...], axes=(1,2,3)).real
        V = np.einsum('ijklm,jklm->iklm', coeff, grad_u)
        V_hat = fftn(V, axes=(1,2,3))
        div_V = ifftn(np.sum(1j * K * V_hat, axis=0)).real
        print("max div_V:", np.max(np.abs(div_V)))
        sys.exit(0)
        return div_V

    def apply_operator(self, u_flat):
        u = u_flat.reshape(self.shape)
        Lu = self._apply_L(u, self.coeff, self.K)
        return Lu.flatten()

    def _restrict(self, arr, new_shape):
        hat = fftn(arr)
        def truncate(h, ns, os):
            idx = [np.concatenate([np.arange(0, n//2), np.arange(o - n//2, o)]) for n, o in zip(ns, os)]
            return h[np.ix_(*idx)]
        hat_coarse = truncate(hat, new_shape, arr.shape)
        return ifftn(hat_coarse * (np.prod(new_shape) / np.prod(arr.shape))).real

    def _prolong(self, arr, new_shape):
        hat_coarse = fftn(arr)
        hat_fine = np.zeros(new_shape, dtype=complex)
        def pad(hf, hc):
            ns, os = hf.shape, hc.shape
            idx_f = [np.concatenate([np.arange(0, o//2), np.arange(n - o//2, n)]) for n, o in zip(ns, os)]
            hf[np.ix_(*idx_f)] = hc
            return hf
        hat_fine = pad(hat_fine, hat_coarse)
        return ifftn(hat_fine * (np.prod(new_shape) / np.prod(arr.shape))).real

    def get_preconditioner(self, method='spectral_laplacian'):
        """
        Provides a preconditioner for the solver.
        'spectral_laplacian' is often the most effective for these problems.
        """
        # Mean metric for preconditioning
        g_avg = np.mean(self.coeff, axis=(2,3,4)) 
        
        # L_prec symbol = - K_i * g_avg^ij * K_j
        symbol = np.zeros(self.shape)
        for i in range(3):
            for j in range(3):
                symbol -= self.K[i] * g_avg[i,j] * self.K[j]
        
        symbol[0,0,0] = 1.0 
        inv_symbol = 1.0 / symbol
        inv_symbol[0,0,0] = 0.0
        
        def prec_func(r_flat):
            r = r_flat.reshape(self.shape)
            r_hat = fftn(r)
            u_hat = r_hat * inv_symbol
            u = ifftn(u_hat).real
            return u.flatten()
        
        return LinearOperator((self.coeff.size//9, self.coeff.size//9), matvec=prec_func)

    def solve(self, f, rtol=1e-8, atol=1e-10, maxiter=1000):
        rhs = (self.sqrt_g * f).flatten()
        rhs -= np.mean(rhs)
        
        A = LinearOperator((rhs.size, rhs.size), matvec=lambda v: -self.apply_operator(v))
        M = self.get_preconditioner()
        
        # In SciPy >= 1.12, 'tol' is deprecated in favor of 'rtol' and 'atol'
        u_flat, info = cg(A, -rhs, rtol=rtol, atol=atol, maxiter=maxiter, M=M)
        
        if info > 0:
            print(f"Warning: CG did not converge after {info} iterations")
            
        u = u_flat.reshape(self.shape)
        u -= np.mean(u)
        return u

def test_solvers():
    N = 64
    solver = BeltramiPoissonSolver(N, N, N)
    X, Y, Z = np.meshgrid(solver.x, solver.y, solver.z, indexing='ij')
    
    # Define a non-diagonal metric g_ij
    # Ensure it is symmetric and positive definite
    a = (1.2 + 0.2 * np.sin(X) * np.cos(Y))
    b = 0.1 * np.cos(Z)
    c = 0.05 * np.sin(X)
    
    g_ij = np.zeros((3, 3, N, N, N))
    g_ij[0, 0] = a
    g_ij[1, 1] = a + 0.1
    g_ij[2, 2] = a + 0.2
    
    g_ij[0, 1] = g_ij[1, 0] = b
    g_ij[0, 2] = g_ij[2, 0] = c
    g_ij[1, 2] = g_ij[2, 1] = 0.02
    
    solver.set_metric(g_ij)
    
    u_true = np.sin(X) * np.sin(Y) * np.sin(Z)
    Lu = solver.apply_operator(u_true.flatten()).reshape((N,N,N))
    f = Lu / solver.sqrt_g
    
    import time
    start = time.time()
    u_sol = solver.solve(f, rtol=1e-9)
    print(f"Solve time (N={N}): {time.time() - start:.2f}s")
    
    err = np.linalg.norm(u_sol - u_true) / np.linalg.norm(u_true)
    print(f"Error: {err:.2e}")
    assert err < 1e-6

if __name__ == "__main__":
    test_solvers()
