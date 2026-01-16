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
