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
