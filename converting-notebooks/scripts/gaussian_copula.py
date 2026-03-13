# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "scipy",
# ]
# ///
"""Gaussian copula demonstration.

Generates samples from a bivariate normal distribution, transforms each
marginal through the standard normal CDF to map values into [0, 1], and
visualises the resulting Gaussian copula alongside the true density.

Typical usage:
    uv run 20260223_gaussian_copula.py
"""

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# --- Constants ---
N_SAMPLES = 2_000
CORRELATION = 0.8
RANDOM_SEED = 42
DATE_PREFIX = "20260223"
OUTPUT_DIR = "."


def _out(name: str) -> str:
    """Return an output file path prefixed with DATE_PREFIX.

    Args:
        name: Base filename including extension (e.g. ``"copula.png"``).

    Returns:
        Path string of the form ``"<OUTPUT_DIR>/<DATE_PREFIX>_<name>"``.
    """
    return f"{OUTPUT_DIR}/{DATE_PREFIX}_{name}"


def generate_bivariate_normal(
    n_samples: int,
    correlation: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate samples from a correlated bivariate normal distribution.

    Args:
        n_samples: Number of samples to generate.
        correlation: Pearson correlation coefficient in ``[-1, 1]``.
        seed: Random seed for reproducibility.

    Returns:
        Tuple ``(X, Y)`` where each element is a 1-D array of length
        ``n_samples``.
    """
    rng = np.random.default_rng(seed)
    cov = [[1.0, correlation], [correlation, 1.0]]
    samples = rng.multivariate_normal([0.0, 0.0], cov, n_samples)
    return samples[:, 0], samples[:, 1]


def to_copula_space(
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform bivariate normal samples into Gaussian copula space.

    Passes each variable through the standard normal CDF Phi so that
    marginal distributions become uniform on ``[0, 1]``.

    Args:
        X: 1-D array of normally distributed samples.
        Y: 1-D array of normally distributed samples.

    Returns:
        Tuple ``(U, V)`` where ``U = Phi(X)`` and ``V = Phi(Y)``,
        with values in ``[0, 1]``.
    """
    return norm.cdf(X), norm.cdf(Y)


def compute_copula_density(
    rho: float,
    grid_size: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the true Gaussian copula density on a [0, 1]^2 grid.

    The density formula is:

        c(u, v; rho) = phi_2(xi, zeta; rho) / (phi(xi) * phi(zeta))

    where xi = Phi^{-1}(u) and zeta = Phi^{-1}(v).

    Args:
        rho: Pearson correlation coefficient, ``|rho| < 1``.
        grid_size: Number of evaluation points along each axis.

    Returns:
        Tuple ``(ug, vg, density)`` where ``ug`` and ``vg`` are 2-D
        coordinate arrays of shape ``(grid_size, grid_size)`` and
        ``density`` holds the copula density at each grid point.
    """
    eps = 1e-6
    u = np.linspace(eps, 1 - eps, grid_size)
    ug, vg = np.meshgrid(u, u)
    xi, zeta = norm.ppf(ug), norm.ppf(vg)
    exponent = -(rho**2 * (xi**2 + zeta**2) - 2 * rho * xi * zeta) / (
        2 * (1 - rho**2)
    )
    density = np.exp(exponent) / np.sqrt(1 - rho**2)
    return ug, vg, density


def plot_copula(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    correlation: float,
    output_path: str,
) -> None:
    """Plot bivariate normal samples and the Gaussian copula side by side.

    The copula panel overlays the true density as filled contours under
    the empirical scatter.

    Args:
        X: Original normal samples for the horizontal axis.
        Y: Original normal samples for the vertical axis.
        U: Copula-transformed samples for the horizontal axis.
        V: Copula-transformed samples for the vertical axis.
        correlation: Correlation coefficient used in title labels and
            density overlay.
        output_path: Destination path for the saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].scatter(X, Y, alpha=0.4, s=5)
    axes[0].set_title(f"Bivariate Normal (rho={correlation})")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].grid(True)

    ug, vg, density = compute_copula_density(correlation)
    axes[1].contourf(ug, vg, density, levels=20, cmap="Blues")
    axes[1].contour(ug, vg, density, levels=20, colors="steelblue", linewidths=0.5)
    axes[1].scatter(U, V, alpha=0.4, s=5, color="orange", label="samples")
    axes[1].set_title("Gaussian Copula (uniform marginals)")
    axes[1].set_xlabel("u = Phi(X)")
    axes[1].set_ylabel("v = Phi(Y)")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)

    plt.suptitle("Gaussian Copula Example", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logging.info("Figure saved to %s", output_path)


def main() -> None:
    """Run the Gaussian copula demonstration."""
    X, Y = generate_bivariate_normal(N_SAMPLES, CORRELATION, seed=RANDOM_SEED)
    U, V = to_copula_space(X, Y)
    plot_copula(X, Y, U, V, CORRELATION, output_path=_out("gaussian_copula.png"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
