import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
from scipy.sparse.linalg import LinearOperator, cg

class MultigridBeltramiSolver:
    r"""
    Poisson-Beltrami solver with a full Multigrid V-cycle preconditioner.
    Optimized for high resolutions (e.g., 1024^3) and GPU-ready.
    """
    def __init__(self, Nx, Ny, Nz, L=2*np.pi, n_levels=3):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.shape = (Nx, Ny, Nz)
        self.L = L
        self.n_levels = n_levels
        
        # Base grid wave numbers
        self.K = self._get_K(self.shape)

    def _get_K(self, shape):
        """Compute wave numbers for a given grid shape"""
        kx = fftfreq(shape[0], d=self.L/(2*np.pi*shape[0]))
        ky = fftfreq(shape[1], d=self.L/(2*np.pi*shape[1]))
        kz = fftfreq(shape[2], d=self.L/(2*np.pi*shape[2]))
        K = np.zeros((3,) + shape)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K[0], K[1], K[2] = KX, KY, KZ
        return K

    def set_metric(self, g_ij):
        """
        Set the metric and build the multigrid hierarchy.
        g_ij should be (3, 3, Nx, Ny, Nz)
        """
        self.g = np.asarray(g_ij)
        # Compute coefficients for the fine level
        det_g = (self.g[0,0]*(self.g[1,1]*self.g[2,2] - self.g[1,2]*self.g[2,1]) -
                 self.g[0,1]*(self.g[1,0]*self.g[2,2] - self.g[1,2]*self.g[2,0]) +
                 self.g[0,2]*(self.g[1,0]*self.g[2,1] - self.g[1,1]*self.g[2,0]))
        self.sqrt_g = np.sqrt(det_g)
        
        # Inverse metric
        inv_det = 1.0 / det_g
        g_inv = np.zeros_like(self.g)
        g_inv[0,0] = (self.g[1,1]*self.g[2,2] - self.g[1,2]*self.g[2,1]) * inv_det
        g_inv[0,1] = (self.g[0,2]*self.g[2,1] - self.g[0,1]*self.g[2,2]) * inv_det
        g_inv[0,2] = (self.g[0,1]*self.g[1,2] - self.g[0,2]*self.g[1,1]) * inv_det
        g_inv[1,0] = (self.g[1,2]*self.g[2,0] - self.g[1,0]*self.g[2,2]) * inv_det
        g_inv[1,1] = (self.g[0,0]*self.g[2,2] - self.g[0,2]*self.g[2,0]) * inv_det
        g_inv[1,2] = (self.g[1,0]*self.g[0,2] - self.g[0,0]*self.g[1,2]) * inv_det
        g_inv[2,0] = (self.g[1,0]*self.g[2,1] - self.g[1,1]*self.g[2,0]) * inv_det
        g_inv[2,1] = (self.g[0,1]*self.g[2,0] - self.g[0,0]*self.g[2,1]) * inv_det
        g_inv[2,2] = (self.g[0,0]*self.g[1,1] - self.g[0,1]*self.g[1,0]) * inv_det
        
        self.coeff = self.sqrt_g[None, None, ...] * g_inv
        
        # Build Hierarchy
        self.levels = []
        curr_coeff = self.coeff
        curr_shape = self.shape
        
        for i in range(self.n_levels):
            self.levels.append({
                'shape': curr_shape,
                'coeff': curr_coeff,
                'K': self._get_K(curr_shape)
            })
            if i < self.n_levels - 1:
                new_shape = tuple(s // 2 for s in curr_shape)
                # Spectral restriction of coefficients
                new_coeff = np.zeros((3, 3) + new_shape)
                for j in range(3):
                    for k in range(3):
                        new_coeff[j, k] = self._restrict(curr_coeff[j, k], new_shape)
                curr_coeff = new_coeff
                curr_shape = new_shape

    def _restrict(self, arr, new_shape):
        hat = fftn(arr)
        idx = [np.concatenate([np.arange(0, n//2), np.arange(o - n//2, o)]) 
               for n, o in zip(new_shape, arr.shape)]
        hat_coarse = hat[np.ix_(*idx)]
        return ifftn(hat_coarse * (np.prod(new_shape) / np.prod(arr.shape))).real

    def _prolong(self, arr, new_shape):
        hat_coarse = fftn(arr)
        hat_fine = np.zeros(new_shape, dtype=complex)
        idx_f = [np.concatenate([np.arange(0, o//2), np.arange(n - o//2, n)]) 
                 for n, o in zip(new_shape, arr.shape)]
        hat_fine[np.ix_(*idx_f)] = hat_coarse
        return ifftn(hat_fine * (np.prod(new_shape) / np.prod(arr.shape))).real

    def _apply_L(self, u, level):
        u_hat = fftn(u)
        grad_u = ifftn(1j * level['K'] * u_hat[None, ...], axes=(1,2,3)).real
        V = np.einsum('ijklm,jklm->iklm', level['coeff'], grad_u)
        V_hat = fftn(V, axes=(1,2,3))
        div_V = ifftn(np.sum(1j * level['K'] * V_hat, axis=0)).real
        return div_V

    def v_cycle(self, r, level_idx=0):
        level = self.levels[level_idx]
        
        if level_idx == self.n_levels - 1:
            # Coarsest level: Solve or smooth heavily
            return self._smooth(r, level, niters=20)
        
        # Pre-smooth
        e = self._smooth(r, level, niters=1)
        
        # Residual: r - L(e). But CG solves -L u = -rhs.
        # Here we follow standard MG for L e = r
        res = r - self._apply_L(e, level)
        
        # Restrict
        res_coarse = self._restrict(res, self.levels[level_idx+1]['shape'])
        
        # Recursive call
        e_coarse = self.v_cycle(res_coarse, level_idx + 1)
        
        # Prolongate and Correct
        e = e + self._prolong(e_coarse, level['shape'])
        
        # Post-smooth
        e = self._smooth(e, level, niters=1)
        return e

    def _smooth(self, r, level, niters=1):
        """Damped Richardson smoother using the constant-coefficient inverse"""
        g_avg = np.mean(level['coeff'], axis=(2,3,4))
        symbol = np.zeros(level['shape'])
        for i in range(3):
            for j in range(3):
                symbol -= level['K'][i] * g_avg[i,j] * level['K'][j]
        
        # symbol is negative for Poisson. Preconditioner is P = ifft( 1/symbol * fft(...) )
        symbol[0,0,0] = 1.0
        inv_symbol = 1.0 / symbol
        inv_symbol[0,0,0] = 0.0
        
        u = np.zeros_like(r)
        # We want to solve L u = r.
        # Richardson: u_{n+1} = u_n + omega * P(r - L u_n)
        omega = 0.6
        for _ in range(niters):
            res = r - self._apply_L(u, level)
            res_hat = fftn(res)
            u = u + omega * ifftn(res_hat * inv_symbol).real
        return u

    def solve(self, f, rtol=1e-8, atol=1e-10, maxiter=100):
        rhs = (self.sqrt_g * f)
        rhs -= np.mean(rhs)
        
        # A u = b where A = -L
        def mv(v):
            return -self._apply_L(v.reshape(self.shape), self.levels[0]).flatten()
        
        A = LinearOperator((rhs.size, rhs.size), matvec=mv)
        
        # M is preconditioner such that M r approx A^-1 r
        # Since A = -L, and v_cycle solves L e = r, 
        # then e = L^-1 r = -A^-1 r.
        # So A^-1 r = -v_cycle(r).
        def prec(v):
            return -self.v_cycle(v.reshape(self.shape)).flatten()
            
        M = LinearOperator((rhs.size, rhs.size), matvec=prec)
        
        # SciPy >= 1.12 uses rtol and atol
        u_flat, info = cg(A, -rhs.flatten(), rtol=rtol, atol=atol, maxiter=maxiter, M=M)
        if info > 0:
            print(f"Warning: CG did not converge after {info} iterations")
        return u_flat.reshape(self.shape)

if __name__ == "__main__":
    N = 64
    solver = MultigridBeltramiSolver(N, N, N, n_levels=3)
    X, Y, Z = np.meshgrid(np.linspace(0, 2*np.pi, N, False), 
                          np.linspace(0, 2*np.pi, N, False), 
                          np.linspace(0, 2*np.pi, N, False), indexing='ij')
    
    # Non-diagonal metric
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
    
    u_true = np.sin(X)*np.sin(Y)*np.sin(Z)
    f = -solver._apply_L(u_true, solver.levels[0]) / solver.sqrt_g
    
    import time
    t0 = time.time()
    u_sol = solver.solve(f)
    print(f"MG-CG Solve Time: {time.time()-t0:.2f}s")
    print(f"Error: {np.linalg.norm(u_sol-u_true)/np.linalg.norm(u_true):.2e}")
