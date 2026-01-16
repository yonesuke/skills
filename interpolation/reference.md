# Interpolation Reference

## 1. Taxonomy

| Type | Algorithms | Characteristics | Best For | Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **Global Polynomial** | Lagrange, Newton, Barycentric | Fit single polynomial $P(x)$ to all $N$ data points. | Small $N$ (<20), Analytical derivatives. | $O(N^2)$ setup, $O(N)$ eval |
| **Piecewise Polynomial** | Linear, Cubic Spline, PCHIP, Akima | Fit low-degree polys between adjacent points $x_i, x_{i+1}$. | **General Purpose**. Smooth, oscillation-free (PCHIP). | $O(N)$ |
| **Multidimensional** | Bilinear, Bicubic, Regular Grid | Extension to 2D/3D grids. Tensor product of 1D schemes. | Image processing, Physics grids. | $O(1)$ lookup + compute |
| **Scattered Data** | RBF, IDW, Nearest Neighbor | Data not on grid. Distance-based weights. | GIS, experimental data. | $O(N^2)$ or $O(N)$ (approx) |

---

## 2. Global Polynomial Interpolation
Given $N+1$ points $(x_0, y_0), \dots, (x_N, y_N)$, find unique polynomial $P_N(x)$ of degree $N$.

### Lagrange Form
$$ P(x) = \sum_{j=0}^{N} y_j L_j(x), \quad L_j(x) = \prod_{i \neq j} \frac{x - x_i}{x_j - x_i} $$
*   **Pros**: Explicit formula.
*   **Cons**: Numerically unstable for large $N$ (Runge's Phenomenon). $O(N^2)$ to calculate coeffs. Re-calcs required if new point added.

### Barycentric Lagrange Form (Best Practice)
$$ P(x) = \frac{\sum_{j=0}^N \frac{w_j y_j}{x - x_j}}{\sum_{j=0}^N \frac{w_j}{x - x_j}} $$
Where weights $w_j = \frac{1}{\prod_{i \neq j} (x_j - x_i)}$.
*   **Pros**: Numerical stability, $O(N)$ evaluation. Computing weights is $O(N^2)$ but done once.

### Runge's Phenomenon
Oscillation at edges when interpolating equidistant points with high-degree polynomials.
*   **Solution**: Use **Chebyshev nodes** (clustered at edges) instead of equidistant nodes.

---

## 3. Piecewise Polynomial (Splines)

### Linear
Connect points with lines. Continuous ($C^0$) but not smooth (undef derivative at nodes).
$$ S_i(x) = y_i + \frac{y_{i+1} - y_i}{x_{i+1} - x_i} (x - x_i) $$

### Cubic Spline
Function is piecewise cubic polynomial $S_i(x)$ on $[x_i, x_{i+1}]$.
Conditions:
1.  Interpolation: $S_i(x_i) = y_i, S_i(x_{i+1}) = y_{i+1}$
2.  Smoothness: $S'_i(x_{i+1}) = S'_{i+1}(x_{i+1})$ and $S''_i(x_{i+1}) = S''_{i+1}(x_{i+1})$ ($C^2$ continuity).
3.  Boundary: Natural ($S''=0$) or Clamped ($S'$ fixed).
*   **Algorithm**: Solves tridiagonal linear system ($O(N)$) for derivatives/moments.
*   **Pros**: Minimum curvature (smoothest physical curve).

### PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
Similar to Cubic Spline ($C^1$ continuity), but preserves **monotonicity**.
*   **Logic**: If data is monotonic, PCHIP ensures interpolation is monotonic (no overshoots).
*   **Derivatives**: Computed locally using harmonic mean of slopes, not globally.
*   **Algorithm**: Explicit calculation of derivatives $d_i$.
    $$ \frac{1}{d_i} = \frac{1}{2}\left( \frac{1}{\delta_{i-1}} + \frac{1}{\delta_i} \right) \quad (\text{if signs match}) $$
    where $\delta_i$ is slope of secant.
*   **Pseudo-code**:
    ```python
    if signs of secants match:
        d[i] = harmonic_mean(secant[i-1], secant[i])
    else:
        d[i] = 0 // Extrema
    ```

### Akima Spline
Special derivative calculation designed to reduce "wiggles" and handle outliers better than Spline.
*   $C^1$ continuous.
*   Derivative at $i$ depends on slopes of *two* adjacent segments on each side (5 points).
    $$ d_i = \frac{|m_{i+1} - m_i| m_{i-1} + |m_{i-1} - m_{i-2}| m_i}{|m_{i+1} - m_i| + |m_{i-1} - m_{i-2}|} $$

---

## 4. Multidimensional (2D)

### Bilinear Interpolation (Square Grid)
Interpolate on $x$ then on $y$.
$$ f(x, y) \approx \frac{1}{(x_2-x_1)(y_2-y_1)} [x_2-x, x-x_1] \begin{pmatrix} f(Q_{11}) & f(Q_{12}) \\ f(Q_{21}) & f(Q_{22}) \end{pmatrix} \begin{pmatrix} y_2-y \\ y-y_1 \end{pmatrix} $$
*   Weighted average of 4 neighbors.

### Bicubic Interpolation
Uses 16 neighbors ($4 \times 4$ window). Interpolates value and derivatives.
*   Common in image resizing. Smoother than bilinear.

### Scatter Interpolation
1.  **Nearest Neighbor**: Voronoi cells.
2.  **Linear Barycentric**: Delaunay triangulation -> Plane in triangle.
3.  **Clough-Tocher**: Delaunay -> Cubic polynomial in triangle ($C^1$).
