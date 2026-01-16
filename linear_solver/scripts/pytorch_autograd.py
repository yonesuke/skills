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
