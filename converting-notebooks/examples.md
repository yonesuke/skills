# Converting Notebooks Examples

Target output structure for a converted Jupyter notebook.

## Canonical Output: Gaussian Copula

A notebook with exploratory cells (bivariate normal sampling, CDF transform,
contour plot) converted to a clean script.

**Source:** [scripts/gaussian_copula.py](scripts/gaussian_copula.py)

```python
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
    """Return an output file path prefixed with DATE_PREFIX."""
    return f"{OUTPUT_DIR}/{DATE_PREFIX}_{name}"


def generate_bivariate_normal(
    n_samples: int, correlation: float, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate samples from a correlated bivariate normal distribution.

    Args:
        n_samples: Number of samples to generate.
        correlation: Pearson correlation coefficient in [-1, 1].
        seed: Random seed for reproducibility.

    Returns:
        Tuple (X, Y) where each element is a 1-D array of length n_samples.
    """
    rng = np.random.default_rng(seed)
    cov = [[1.0, correlation], [correlation, 1.0]]
    samples = rng.multivariate_normal([0.0, 0.0], cov, n_samples)
    return samples[:, 0], samples[:, 1]


def to_copula_space(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform bivariate normal samples into Gaussian copula space (U, V in [0,1])."""
    return norm.cdf(X), norm.cdf(Y)


def plot_copula(X, Y, U, V, correlation, output_path) -> None:
    """Plot bivariate normal samples and Gaussian copula side by side and save."""
    # ... see scripts/gaussian_copula.py for full implementation


def main() -> None:
    """Run the Gaussian copula demonstration."""
    X, Y = generate_bivariate_normal(N_SAMPLES, CORRELATION, seed=RANDOM_SEED)
    U, V = to_copula_space(X, Y)
    plot_copula(X, Y, U, V, CORRELATION, output_path=_out("gaussian_copula.png"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
```

## Key Structural Rules

| Element | Rule |
|---|---|
| File name | `YYYYMMDD_<stem>.py` — today's date unless overridden |
| Output files | `YYYYMMDD_<name>.png` / `.csv` via `_out()` helper |
| `# /// script` | **Must be first** — before the module docstring |
| Dependency versions | Pin when known; omit when uncertain (never fabricate) |
| `matplotlib` backend | `matplotlib.use("Agg")` on the line right after `import matplotlib` |
| Constants | All hardcoded literals → `UPPER_SNAKE_CASE` at module level |
| Docstrings | Google style, English; translate Japanese math comments |
| Logging vs. print | `logging.info` for progress; `print` for result tables/metrics |
| Reproducibility | `RANDOM_SEED` constant; `np.random.default_rng(seed)` in functions |
| `main()` | High-level narrative only — delegate all logic to named functions |

## Magic Command Translation

| Notebook magic | Script replacement |
|---|---|
| `%matplotlib inline` | `import matplotlib; matplotlib.use("Agg")` |
| `%time expr` | `t0 = time.perf_counter(); ...; logging.info("%.2fs", time.perf_counter()-t0)` |
| `!pip install pkg` | Move to `# /// script` dependencies block |
| `!shell_cmd` | `subprocess.run(["shell_cmd", ...], check=True)` |
| `display(fig)` | `fig.savefig(_out("name.png")); plt.close(fig)` |
| `plt.show()` | `fig.savefig(_out("name.png")); plt.close(fig)` |
