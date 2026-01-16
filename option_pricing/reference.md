# Option Pricing Reference

## 1. Fundamentals

### GBM & Ito's Lemma
*   **SDE**: $dS_t = \mu S_t dt + \sigma S_t dW_t$
*   **Ito**: $df = (\frac{\partial f}{\partial t} + \dots) dt + \sigma \dots dW_t$

## 2. Pricing Framework

*   **Risk-Neutral Measure ($\mathbb{Q}$)**: Discounted price is a martingale.
*   **Feynman-Kac**: Link between PDE and Expectation.

## 3. Black-Scholes Model

**PDE**:
$$ \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r-q) S \frac{\partial V}{\partial S} - rV = 0 $$

## 4. Analytical Solutions & Greeks

### Vanilla European
*   Call: $S e^{-q\tau} N(d_1) - K e^{-r\tau} N(d_2)$
*   Put: $K e^{-r\tau} N(-d_2) - S e^{-q\tau} N(-d_1)$

### Greeks
*   **Delta**: Sensitivity to Price.
*   **Gamma**: Sensitivity to Delta.
*   **Vega**: Sensitivity to Volatility.
*   **Theta**: Time decay.
*   **Rho**: Sensitivity to rates.

### Digital Options (Binary)
*   Cash-or-nothing, Asset-or-nothing solutions provided in [original cheatsheet content - preserved in Sections 4 & 5 of full text].

## 5. Exotic Options (Path Dependent)

### Barrier Options
Single Barrier (Up/Down + In/Out).
*   Formulas involve reflection principle terms ($ (H/S)^{2\lambda} $).

### Lookback Options
Floating vs Fixed strike. Depend on Max/Min over period.

### Asian Options
*   **Geometric**: Closed form exists (volatility adjustment).
*   **Arithmetic**: No closed form (approx needed).

## 6. American Options
*   **Finite**: No closed form (Trees, PDE, BBAW).
*   **Perpetual**: ODE solution ($S^\gamma$).

## 7. Delta Hedging & PnL
*   **Market Completeness**: Risk elimination via replication.
*   **PnL**: $\approx \frac{1}{2} S^2 \Gamma (\sigma_{imp}^2 - \sigma_{real}^2) dt$. (Gamma trading).
