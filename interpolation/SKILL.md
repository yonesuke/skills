---
name: interpolation
description: Techniques for 1D and 2D data interpolation using SciPy (CPU) and JAX/Interpax (GPU/Differentiable).
---
# Interpolation Skill

Connect the dots. Choose SciPy for standard usage, JAX for gradient-based optimization.

## Contents
- [Examples](examples.md)
    - Cubic Splines, PCHIP, Regular Grids.
- [Reference](reference.md)
    - Theoretical background on splines and grid methods.

## Quick Decision Tree
1.  **Standard/CPU?** -> `SciPy` (`CubicSpline`, `RegularGridInterpolator`).
2.  **Differentiable/GPU?** -> `Interpax` (`interpax.interp1d`).
3.  **Monotonicity Important?** -> Use `PCHIP` (SciPy) or `cubic2` (Interpax).
