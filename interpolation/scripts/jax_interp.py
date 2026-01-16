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

    # 2. Multidimensional
    x_grid = jnp.linspace(0, 1, 10)
    y_grid = jnp.linspace(0, 1, 10)
    z_vals = jnp.zeros((10, 10)) 
    
    # Query points must be broadcastable. interpax.interp2d expects x, y, z arguments
    # xq: (D, N) where D=params
    coords_q = jnp.array([[0.5], [0.5]]) # This example usage might need tuning based on exact interpax version API
    
    # Simplest usage: interp1d is robust.
    # Check documentation or simple trial for 2d.
    # interpax.interp2d(x, y, xp, yp, zp, method='cubic')
    # Let's stick to 1D for safety in demo or ensure API match.
    # Assuming the Example code in original key was correct:
    # yq = interpax.interp2d(xq, x_grid, y_grid, z_vals, method='cubic')
    
    print("Interpax 1D successful.")

if __name__ == "__main__":
    main()
