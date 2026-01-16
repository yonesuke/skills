# Interpolation Examples

1D and 2D interpolation using SciPy and JAX.

## 1. SciPy (Standard)
Includes Splines (Cubic, PCHIP) and 2D Grids.

**Source:** [scripts/scipy_interp.py](scripts/scipy_interp.py)
```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d, RegularGridInterpolator, griddata

def main():
    print("Running SciPy Interpolation Demo")
    
    # --- 1D Interpolation ---
    print("\n1D Interpolation:")
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 0.8, 0.9, 0.1, -0.8, -1.0])
    x_new = np.linspace(0, 5, 5) # Coarse for printing

    # Cubic Spline
    cs = CubicSpline(x, y, bc_type='natural') 
    print(f"Cubic Spline: {cs(x_new)}")

    # PCHIP
    pch = PchipInterpolator(x, y)
    print(f"PCHIP: {pch(x_new)}")
# ...
```

## 2. JAX (Differentiable)
Using `interpax` for GPU-accelerated splines.

**Source:** [scripts/jax_interp.py](scripts/jax_interp.py)
```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax", "jaxlib", "interpax"]
# ///

import jax.numpy as jnp
import interpax
import jax

def main():
    print("Running JAX/Interpax Interpolation Demo")
    
    x = jnp.linspace(0, 5, 6)
    y = jnp.sin(x)
    xq = jnp.linspace(0, 5, 5)

    # 1. Cubic Spline
    # method: 'linear', 'cubic', 'cubic2' (monotonic/pchip), 'cardinal', 'akima'
    yq = interpax.interp1d(xq, x, y, method='cubic')
    print(f"Cubic Spline JAX: {yq}")
# ...
```
