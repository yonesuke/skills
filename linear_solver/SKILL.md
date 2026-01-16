---
name: linear-solver
description: Solves linear systems (Ax=b) using NumPy, JAX, PyTorch, and Lineax. Covers dense, sparse, and specialized (tridiagonal) solvers.
---
# Linear Solver Skill

Solves $Ax=b$. Choose the backend based on your hardware and scale.

## Contents
- [Examples](examples.md)
    - Code snippets for NumPy, JAX, PyTorch, and Lineax.
- [Reference](reference.md)
    - Detailed breakdown of methods (LU, Cholesky, CG, GMRES).

## Quick Decision Tree

1.  **CPU Only?** -> Use `NumPy` (`scripts/numpy_dense.py`).
2.  **GPU/TPU?** -> Use `JAX` or `PyTorch`.
3.  **Advanced/Stable JAX?** -> Use `Lineax` (`scripts/lineax_solvers.py`).
4.  **Tridiagonal?** -> Use specialized solvers (`scripts/tridiagonal.py`).
