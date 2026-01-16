# /// script
# requires-python = ">=3.12"
# dependencies = ["jax", "jaxlib"]
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

@jit
def black_scholes_put(S, K, T, r, sigma):
    """Price of a European Put Option."""
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return K * jnp.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# 2. Greeks Calculation using Auto-Differentiation
def calculate_greeks(S, K, T, r, sigma):
    # Delta: dV/dS
    delta_call = grad(black_scholes_call, argnums=0)(S, K, T, r, sigma)
    
    # Gamma: d^2V/dS^2
    gamma_call = grad(grad(black_scholes_call, argnums=0), argnums=0)(S, K, T, r, sigma)
    
    # Vega: dV/dSigma
    vega_call = grad(black_scholes_call, argnums=4)(S, K, T, r, sigma)
    
    # Theta: dV/dT (Time decay is usually negative of derivative wrt T)
    theta_call = -grad(black_scholes_call, argnums=2)(S, K, T, r, sigma)
    
    # Rho: dV/dr
    rho_call = grad(black_scholes_call, argnums=3)(S, K, T, r, sigma)

    return {
        "Delta": delta_call,
        "Gamma": gamma_call,
        "Vega": vega_call,
        "Theta": theta_call,
        "Rho": rho_call
    }

def main():
    print("Running Option Pricing Demo (JAX)")
    
    S0 = 100.0   # Spot
    K = 100.0    # Strike
    T = 1.0      # Maturity
    r = 0.05     # Risk-free rate
    sigma = 0.2  # Volatility

    call_price = black_scholes_call(S0, K, T, r, sigma)
    put_price = black_scholes_put(S0, K, T, r, sigma)
    
    print(f"Call Price: {call_price:.4f}")
    print(f"Put Price:  {put_price:.4f}")

    print("\n--- Greeks (Call) ---")
    greeks = calculate_greeks(S0, K, T, r, sigma)
    for k, v in greeks.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
