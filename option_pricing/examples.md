# Option Pricing Examples

Analytical solutions and Greeks calculation using JAX.

## 1. Black-Scholes-Merton & Greeks
Calculates European option prices and Greeks using JAX auto-diff.

**Source:** [scripts/bsm.py](scripts/bsm.py)
```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax", "jaxlib", "scipy"]
# ///

import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.scipy.stats import norm

# 1. Black-Scholes-Merton Formula
@jit
def black_scholes_call(S, K, T, r, sigma):
    """
    Price of a European Call Option.
    S: Spot Price
    K: Strike Price
    T: Time to Maturity (years)
    r: Risk-free rate
    sigma: Volatility
    """
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)

# ... (See script for Put and Greeks)
```
