import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
import sys
class ParallelBeltramiSolver:
    """
    Parallel Poisson-Beltrami solver for T^3.
    Solves A u = b where A = -div(sqrt(g) g^ij grad).
    Includes auto-calibration of FFT scaling and robust wavenumber mapping.
    """
    def __init__(self, N, L=2*np.pi, comm=MPI.COMM_WORLD):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.N = N if isinstance(N, tuple) else (N, N, N)
        self.L = L
        
        # Initialize Parallel FFT (Real-to-Complex)
        self.fft = PFFT(comm, self.N, axes=(0, 1, 2), dtype=np.float64) # ffts aredone in reverse axes order, so 2 is first to be performed and thus is the one that gets the r2c optimization
        
        self.local_phys_shape = self.fft.shape(False) # local shape of physical data
        self.local_fourier_shape = self.fft.shape(True)
        self.phys_slice = self.fft.local_slice(False) # tuple of slices (coordinate ranges in global space for this rank in physical space)
        self.fourier_slice = self.fft.local_slice(True) # tuple of slices (coordinate ranges in global space for this rank in Fourier space)
        
        # 1. CALIBRATE FFT SCALING
        # Find the factor S such that backward(forward(u)) = S * u
        test_u = np.ones(self.local_phys_shape)
        test_h = self.fft.forward(test_u)
        test_back = self.fft.backward(test_h)
        self.fft_norm = self.comm.allreduce(np.max(np.abs(test_back)), op=MPI.MAX)
        if self.rank == 0:
            print(f"FFT Calibration: Scale Factor = {self.fft_norm}")

        # Local physical coordinates
        coords = [np.linspace(0, L, n, endpoint=False) for n in self.N]
        self.X = coords[0][self.phys_slice[0]][:, None, None]
        self.Y = coords[1][self.phys_slice[1]][None, :, None]
        self.Z = coords[2][self.phys_slice[2]][None, None, :]

        # 2. ROBUST WAVENUMBER MAPPING
        from numpy.fft import fftfreq, rfftfreq
        global_w = [
            fftfreq(self.N[0], d=L/(2*np.pi*self.N[0])),
            fftfreq(self.N[1], d=L/(2*np.pi*self.N[1])),
            rfftfreq(self.N[2], d=L/(2*np.pi*self.N[2])) # see comment above about r2c optimization for last axis
        ]
        
        # Identify axes: 
        try:
            output_axes = tuple(a[0] for a in self.fft.axes)
        except:
            output_axes = (0, 1, 2)
            
        self.K = np.zeros((3,) + self.local_fourier_shape)
        for local_dim, phys_axis in enumerate(output_axes):
            w = global_w[phys_axis].copy()
            # Nyquist stabilization
            if self.N[phys_axis] % 2 == 0:
                if phys_axis < 2: w[self.N[phys_axis]//2] = 0.0
                else: w[-1] = 0.0
                
            sl = self.fourier_slice[local_dim]
            sh = [1, 1, 1]
            sh[local_dim] = self.local_fourier_shape[local_dim]
            self.K[phys_axis] = w[sl].reshape(sh) # only the local portion of the wavenumber array for this rank

    def set_metric(self, g_ij_func):
        g = g_ij_func(self.X, self.Y, self.Z)
        det_g = (g[0,0]*(g[1,1]*g[2,2] - g[1,2]*g[2,1]) -
                 g[0,1]*(g[1,0]*g[2,2] - g[1,2]*g[2,0]) +
                 g[0,2]*(g[1,0]*g[2,1] - g[1,1]*g[2,0]))
        self.sqrt_g = np.sqrt(det_g)
        inv_det = 1.0 / det_g
        
        # C^ij = sqrt(g) * g^ij
        self.coeff = np.zeros_like(g)
        self.coeff[0,0] = (g[1,1]*g[2,2] - g[1,2]*g[2,1]) * inv_det
        self.coeff[0,1] = (g[0,2]*g[2,1] - g[0,1]*g[2,2]) * inv_det
        self.coeff[0,2] = (g[0,1]*g[1,2] - g[0,2]*g[1,1]) * inv_det
        self.coeff[1,1] = (g[0,0]*g[2,2] - g[0,2]*g[2,0]) * inv_det
        self.coeff[1,2] = (g[1,0]*g[0,2] - g[0,0]*g[1,2]) * inv_det
        self.coeff[2,2] = (g[0,0]*g[1,1] - g[0,1]*g[1,0]) * inv_det
        
        self.coeff[1,0], self.coeff[2,0], self.coeff[2,1] = self.coeff[0,1], self.coeff[0,2], self.coeff[1,2]
        self.coeff *= self.sqrt_g[None, None, ...]
        
        l_sum = np.sum(self.coeff, axis=(2,3,4))
        g_sum = np.zeros_like(l_sum)
        self.comm.Allreduce(l_sum, g_sum, op=MPI.SUM)
        self.g_avg = g_sum / (float(self.N[0])*float(self.N[1])*float(self.N[2]))

    def apply_A(self, u):
        u_hat = newDistArray(self.fft, forward_output=True)
        u_hat = self.fft.forward(u, u_hat)
        
        # 3. Compute Gradients
        grad_u = np.zeros((3,) + self.local_phys_shape)
        for j in range(3):
            grad_u[j] = self.fft.backward(1j * self.K[j] * u_hat) / self.fft_norm
        
        # 4. Metric product
        V = np.einsum('ijklm,jklm->iklm', self.coeff, grad_u)
        
        # 5. Compute Divergence
        div_hat = newDistArray(self.fft, forward_output=True)
        div_hat.fill(0.0)
        Vi_hat = newDistArray(self.fft, forward_output=True)
        for i in range(3):
            
            Vi_hat = self.fft.forward(V[i], Vi_hat)
            div_hat += 1j * self.K[i] * Vi_hat
            
        return -self.fft.backward(div_hat) / self.fft_norm

    def solve(self, f, rtol=1e-8, maxiter=150):
        b = -(self.sqrt_g * f)
        b -= self.comm.allreduce(np.sum(b), op=MPI.SUM) / (float(self.N[0])*float(self.N[1])*float(self.N[2]))
        
        
        symbol = np.einsum('ixyz,ij,jxyz->xyz', self.K, self.g_avg, self.K)
        mask = (symbol == 0)
        symbol[mask] = 1.0
        inv_symbol = 1.0 / symbol
        inv_symbol[mask] = 0.0
        
        def M_inv(r):
            r_hat = newDistArray(self.fft, forward_output=True)
            r_hat = self.fft.forward(r, r_hat)
            
            return self.fft.backward(r_hat * inv_symbol) / self.fft_norm

        u = np.zeros_like(b)
        r = b - self.apply_A(u)
        z = M_inv(r)
        p = z.copy()
        
        def dot(a, b):
            return self.comm.allreduce(np.sum(a*b), op=MPI.SUM)
            
        rz = dot(r, z)
        b_norm = np.sqrt(dot(b, b))
        if b_norm < 1e-16: return u
        
        for i in range(maxiter):
            Ap = self.apply_A(p)
            pAp = dot(p, Ap)
            
            if pAp <= 0:
                if self.rank == 0: print(f"Warning: pAp={pAp:.2e}")
                break
                
            alpha = rz / pAp
            u += alpha * p
            r -= alpha * Ap
            
            res_norm = np.sqrt(dot(r, r))
            if self.rank == 0 and i % 5 == 0:
                print(f"Iter {i:3d}: Rel. Res {res_norm/b_norm:.2e}")
            
            if res_norm < rtol * b_norm:
                break
                
            z = M_inv(r)
            rz_new = dot(r, z)
            p = z + (rz_new / rz) * p
            rz = rz_new
            
        return u

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    N = 64
    solver = ParallelBeltramiSolver(N, comm=comm)
    
    # 1. Identity metric verification
    def metric_identity(X, Y, Z):
        g = np.zeros((3, 3, X.shape[0], Y.shape[1], Z.shape[2]))
        for i in range(3): g[i,i] = 1.0
        return g

    if rank == 0: print("\n--- RUNNING IDENTITY METRIC TEST ---")
    t0 = MPI.Wtime()
    solver.set_metric(metric_identity)
    t_setup = MPI.Wtime() - t0
    
    u_true = np.sin(solver.X) * np.sin(solver.Y) * np.sin(solver.Z)
    Au = solver.apply_A(u_true)
    g_max = comm.allreduce(np.max(np.abs(Au)), op=MPI.MAX)
    if rank == 0: print(f"IDENTITY TEST: Max(Au) = {g_max:.6f} (Expected: 3.000000)")
    
    f = -Au / solver.sqrt_g
    
    t1 = MPI.Wtime()
    u_sol = solver.solve(f, rtol=1e-10)
    t_solve = MPI.Wtime() - t1
    
    err = np.sqrt(comm.allreduce(np.sum((u_sol - u_true)**2), op=MPI.SUM))
    norm = np.sqrt(comm.allreduce(np.sum(u_true**2), op=MPI.SUM))
    if rank == 0: 
        print(f"IDENTITY METRIC - Final Relative Error: {err/norm:.2e}")
        print(f"IDENTITY METRIC - Setup Time: {t_setup:.4f}s, Solve Time: {t_solve:.4f}s")

    # 2. Variable non-diagonal metric test (Matching poisson_beltrami_3d.py)
    def metric_variable(X, Y, Z):
        a = (1.2 + 0.2 * np.sin(X) * np.cos(Y))
        b = 0.1 * np.cos(Z)
        c = 0.05 * np.sin(X)
        
        g = np.zeros((3, 3, X.shape[0], Y.shape[1], Z.shape[2]))
        g[0, 0] = a
        g[1, 1] = a + 0.1
        g[2, 2] = a + 0.2
        
        g[0, 1] = g[1, 0] = b
        g[0, 2] = g[2, 0] = c
        g[1, 2] = g[2, 1] = 0.02
        return g

    if rank == 0: print("\n--- RUNNING VARIABLE NON-DIAGONAL METRIC TEST ---")
    t0 = MPI.Wtime()
    solver.set_metric(metric_variable)
    t_setup = MPI.Wtime() - t0

    # regenerate u_true for the new metric grid just in case
    u_true = np.sin(solver.X) * np.sin(solver.Y) * np.sin(solver.Z)
    Au = solver.apply_A(u_true)
    f = -Au / solver.sqrt_g
    
    t1 = MPI.Wtime()
    u_sol = solver.solve(f, rtol=1e-10)
    t_solve = MPI.Wtime() - t1
    
    err = np.sqrt(comm.allreduce(np.sum((u_sol - u_true)**2), op=MPI.SUM))
    norm = np.sqrt(comm.allreduce(np.sum(u_true**2), op=MPI.SUM))
    if rank == 0:
        print(f"VARIABLE METRIC - Final Relative Error: {err/norm:.2e}")
        print(f"VARIABLE METRIC - Setup Time: {t_setup:.4f}s, Solve Time: {t_solve:.4f}s")
