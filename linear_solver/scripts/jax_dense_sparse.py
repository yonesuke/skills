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
