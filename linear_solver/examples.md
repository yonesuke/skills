# Linear Solver Examples

Code patterns for solving linear systems ($Ax=b$) using various backends.

## 1. NumPy (CPU, Dense)
Standard tools for dense systems on CPU.

**Source:** [scripts/numpy_dense.py](scripts/numpy_dense.py)
```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///

import numpy as np

def main():
    print("Running NumPy Linear Solver Demo")

    # System: Ax = b
    N = 100
    np.random.seed(42)
    A = np.random.rand(N, N)
    # Ensure non-singularity
    A = A + np.eye(N) * N
    b = np.random.rand(N)

    # 1. Standard Solve (LU)
    x_lu = np.linalg.solve(A, b)
    print(f"LU Solution Norm: {np.linalg.norm(x_lu)}")

    # 2. Least Squares (if A is not square)
    A_rect = np.random.rand(N + 10, N)
    b_rect = np.random.rand(N + 10)
    x_lstsq, residuals, rank, s = np.linalg.lstsq(A_rect, b_rect, rcond=None)
    print(f"Least Squares Solution Norm: {np.linalg.norm(x_lstsq)}")

    # 3. Cholesky (Manual)
    # Only for SPD matrices: A must be symmetric and positive definite
    A_spd = A @ A.T  # Construct SPD matrix
    L = np.linalg.cholesky(A_spd) # A = L L^T
    # Solve L y = b
    y = np.linalg.solve(L, b) 
    # Solve L^T x = y
    x_chol = np.linalg.solve(L.T, y)
    print(f"Cholesky Solution Norm: {np.linalg.norm(x_chol)}")

if __name__ == "__main__":
    main()
```

## 2. JAX (GPU/TPU)
Dense and Sparse solvers for accelerated hardware.

**Source:** [scripts/jax_dense_sparse.py](scripts/jax_dense_sparse.py)
```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax", "jaxlib"]
# ///

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg

def main():
    print("Running JAX Linear Solver Demo")
    
    key = jax.random.PRNGKey(0)

    # Data Generation
    N = 100
    A = jax.random.normal(key, (N, N))
    A = A + jnp.eye(N) * N # Diagonally dominant -> non-singular
    b = jax.random.normal(key, (N,))

    # 1. Standard Dense Solve (LU on GPU)
    x_dense = jnp.linalg.solve(A, b)
    print(f"Dense Solve Norm: {jnp.linalg.norm(x_dense)}")

    # 2. Sparse / Iterative Solve (CG)
    # JAX requires a linear operator function (matvec)
    def matvec(x):
        return A @ x

    # CG requires Symmetric Positive Definite (SPD) matrix usually
    A_spd = A.T @ A
    b_spd = A.T @ b
    
    # Solving (A^T A) x = A^T b which is the normal equation
    x_cg, info = jax.scipy.sparse.linalg.cg(
        lambda v: A_spd @ v, 
        b_spd, 
        maxiter=1000
    )
    print(f"CG Solve Norm: {jnp.linalg.norm(x_cg)}, Info: {info}")

    # 3. GMRES (General matrices)
    x_gmres, info = jax.scipy.sparse.linalg.gmres(matvec, b)
    print(f"GMRES Solve Norm: {jnp.linalg.norm(x_gmres)}, Info: {info}")

if __name__ == "__main__":
    main()
```

## 3. PyTorch (GPU, Autograd)
Solvers compatible with PyTorch's autograd system.

**Source:** [scripts/pytorch_autograd.py](scripts/pytorch_autograd.py)
```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch"]
# ///

import torch

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU")
    else:
        print("Running on CUDA")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N = 100
    torch.manual_seed(0)
    A = torch.randn(N, N, device=device)
    A = A + torch.eye(N, device=device) * N
    b = torch.randn(N, 1, device=device)

    # 1. Standard Solve (LU)
    x_lu = torch.linalg.solve(A, b)
    print(f"LU Solve Norm: {torch.linalg.norm(x_lu)}")

    # 2. Cholesky Solve
    # Faster for SPD systems
    A_spd = A @ A.T
    # torch.linalg.cholesky_ex is safer for avoiding errors
    L = torch.linalg.cholesky(A_spd)
    x_chol = torch.cholesky_solve(b, L) # Takes L, not A
    print(f"Cholesky Solve Norm: {torch.linalg.norm(x_chol)}")

    # 3. Least Squares
    # driver='gels' (QR) or 'gelsd' (SVD - more stable)
    x_lstsq, residuals, rank, singular_values = torch.linalg.lstsq(A, b, driver='gels')
    print(f"Least Squares Solve Norm: {torch.linalg.norm(x_lstsq)}")

if __name__ == "__main__":
    main()
```

## 4. Lineax (Modern JAX Solvers)
Advanced solvers with better stability and features than native JAX.

**Source:** [scripts/lineax_solvers.py](scripts/lineax_solvers.py)
```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax", "jaxlib", "lineax"]
# ///

import jax
import jax.numpy as jnp
import lineax as lx

def main():
    print("Running Lineax Demo")
    
    key = jax.random.PRNGKey(0)
    N = 100
    A_val = jax.random.normal(key, (N, N))
    b = jax.random.normal(key, (N,))

    # Lineax uses "LinearOperators"
    operator = lx.MatrixLinearOperator(A_val)

    # 1. Auto Solve (Best Practice)
    # Automatically chooses LU, QR, etc.
    solver = lx.AutoLinearSolver(well_posed=True)
    solution = lx.linear_solve(operator, b, solver)
    print(f"Auto Solve Norm: {jnp.linalg.norm(solution.value)}")

    # 2. Explicit Choice (e.g., QR)
    solver_qr = lx.QR()
    solution_qr = lx.linear_solve(operator, b, solver_qr)
    print(f"QR Solve Norm: {jnp.linalg.norm(solution_qr.value)}")

    # 3. Large Scale / Iterative (GMRES)
    # Useful if 'operator' is defined functionally, not as a matrix
    solver_gmres = lx.GMRES(rtol=1e-5, atol=1e-5)
    solution_iter = lx.linear_solve(operator, b, solver_gmres)
    print(f"GMRES Solve Norm: {jnp.linalg.norm(solution_iter.value)}")

if __name__ == "__main__":
    main()
```

## 5. Tridiagonal Systems (Specialized)
Effective solvers for 1D PDE discretizations.

**Source:** [scripts/tridiagonal.py](scripts/tridiagonal.py)
```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax", "jaxlib", "lineax", "scipy", "numpy"]
# ///

import numpy as np
import scipy.linalg
import jax.random as jr
import lineax as lx
import jax
import jax.numpy as jnp

def run_scipy_tdma():
    print("\n--- SciPy TDMA ---")
    N = 100
    np.random.seed(42)
    # Banded format: [Upper diag, Main diag, Lower diag]
    ab = np.zeros((3, N))
    ab[0, 1:] = np.random.rand(N-1) # Upper
    ab[1, :] = np.random.rand(N) + 2 # Main (diagonally dominant)
    ab[2, :-1] = np.random.rand(N-1) # Lower
    b = np.random.rand(N)

    x = scipy.linalg.solve_banded((1, 1), ab, b)
    print(f"SciPy Solution Norm: {np.linalg.norm(x)}")

def run_lineax_tdma():
    print("\n--- Lineax TDMA ---")
    N = 100
    diagonal = jr.normal(jr.PRNGKey(0), (N,))
    upper = jr.normal(jr.PRNGKey(1), (N-1,))
    lower = jr.normal(jr.PRNGKey(2), (N-1,))
    b = jr.normal(jr.PRNGKey(3), (N,))

    operator = lx.TridiagonalLinearOperator(diagonal, lower, upper)
    solution = lx.linear_solve(operator, b, lx.Tridiagonal())
    print(f"Lineax Solution Norm: {jnp.linalg.norm(solution.value)}")

def run_jax_native_tdma():
    print("\n--- JAX Native TDMA ---")
    N = 100
    dl = jax.random.normal(jax.random.key(0), (N,))
    d  = jax.random.normal(jax.random.key(1), (N,))
    du = jax.random.normal(jax.random.key(2), (N,))
    b  = jax.random.normal(jax.random.key(3), (N,))

    # jax.lax.linalg.tridiagonal_solve(dl, d, du, b)
    x = jax.lax.linalg.tridiagonal_solve(dl, d, du, b)
    print(f"JAX Native Solution Norm: {jnp.linalg.norm(x)}")

def main():
    run_scipy_tdma()
    run_lineax_tdma()
    run_jax_native_tdma()

if __name__ == "__main__":
    main()
```
