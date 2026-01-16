# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
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

    # Extrapolation
    f_lin = interp1d(x, y, kind='linear', fill_value="extrapolate")
    print(f"Linear Extrap at 6.0: {f_lin(6.0)}")
    
    # --- 2D Regular Grid ---
    print("\n2D Regular Grid:")
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X + Y)
    
    # Note: RegularGridInterpolator expects points as tuple of axes
    interp = RegularGridInterpolator((x, y), Z, method='linear')
    pts = np.array([[0.1, 0.1], [0.5, 0.5]])
    val = interp(pts)
    print(f"Interp Values at (0.1,0.1) and (0.5,0.5): {val}")

    # --- 2D Scattered ---
    print("\n2D Scattered Data:")
    points = np.random.rand(10, 2)
    values = np.random.rand(10)
    # Query one point
    grid_z = griddata(points, values, [(0.5, 0.5)], method='nearest')
    print(f"Nearest value at (0.5, 0.5): {grid_z}")

if __name__ == "__main__":
    main()
