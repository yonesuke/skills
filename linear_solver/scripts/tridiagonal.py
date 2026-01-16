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
