# Gemini Workspace Overview

## Project Overview

This project contains a numerical solver for a system of coupled geometric partial differential equations (PDEs) on a triply periodic 3-dimensional manifold ($T^3$). The primary goal is to simulate the evolution of a Riemannian metric ($g$) coupled with a one-form ($\lambda$) representing a momentum field, analogous to the Navier-Stokes equations on a dynamic, curved space.

**Key Technologies:**
*   **Language:** Python
*   **Libraries:** NumPy, SciPy

**Core Mathematical Model:**
The solver evolves a state $(g, \lambda)$ according to the following system:
1.  **Metric Evolution**: $\partial_t g = \mathcal{L}_v g$
2.  **Momentum Evolution**: $\partial_t \lambda = -dp - 
u \Delta_H \lambda$
3.  **Constraint**: $\delta \lambda = 0$ (Divergence-free velocity field)
4.  **Closure**: $v = \lambda^\sharp$ (Musical isomorphism)

**Architecture:**
*   The main solver is implemented in `coupled_solver.py`.
*   It uses a **pseudospectral (Fourier-Galerkin)** method for spatial derivatives, which is highly accurate for smooth fields on periodic domains.
*   Time integration is handled by a classic **4th-order Runge-Kutta (RK4)** scheme.
*   The divergence-free constraint is enforced via a **pressure-projection method**. This involves solving a Poisson-Beltrami equation ($\Delta_g p = f$) for a pressure field `p` at each RK4 stage.
*   The elliptic solve for the pressure is handled by the `BeltramiPoissonSolver` class, which uses a **preconditioned conjugate gradient (CG)** method.

## Building and Running

This is a script-based Python project and does not have a separate build step. The solver can be run directly from the command line.

**Dependencies:**
*   `numpy`
*   `scipy`

**Running the Solver:**
The main solver can be executed with the following command:
```bash
python3 coupled_solver.py [OPTIONS]
```

**Command-Line Options:**
*   `--N`: Grid size per direction (e.g., 16, 32).
*   `--dt`: Time step for the RK4 integrator (e.g., 0.005).
*   `--steps`: Total number of time steps to run.
*   `--nu`: Viscosity parameter.
*   `--out`: Name of the output file (`.npz`) to save the final state.

Example:
```bash
python3 coupled_solver.py --N 16 --dt 0.005 --steps 50 --nu 0.05
```

## Development Conventions

*   **Geometric Rigor:** The solver has been specifically updated to use geometrically correct operators. The viscous term uses the **Laplace-de Rham operator** ($\Delta_H$), not a simpler component-wise Laplacian. The pressure equation includes a source term, $S_p = -	ext{div}\left( (\mathcal{L}_v g)^\sharp \cdot v ight)$, that correctly accounts for the time-dependence of the metric. Adherence to these correct formulations is a key convention of this project.
*   **Code Structure:** The main logic is encapsulated in the `CoupledSolver` class. It relies on a suite of standalone helper functions for specific geometric computations (e.g., `compute_christoffel`, `compute_connection_laplacian_1form`). This modular approach should be maintained.
*   **Numerical Methods:** The core numerical engine is based on FFTs for derivatives and CG for the elliptic solve. Changes should be consistent with this pseudospectral framework.
*   **Documentation:** The mathematical model and implementation details are documented in `solver_documentation.tex`. This file should be updated to reflect any significant changes to the model or code.
