# /// script
# requires-python = ">=3.12"
# dependencies = ["jax", "jaxlib"]
# ///

import jax
import jax.lax
import jax.numpy as jnp

def main():
    print("Running Control Flow Demo")

    # Cond
    print("\n--- Cond ---")
    def true_fun(x): return x * 2
    def false_fun(x): return x / 2
    pred = jnp.array(True) # Traced value in real use
    operand = jnp.array(10.0)
    res = jax.lax.cond(pred, true_fun, false_fun, operand)
    print(f"Cond result (True): {res}")

    # Scan
    print("\n--- Scan ---")
    def step(carry, x):
        new_carry = carry + x
        output = new_carry ** 2
        return new_carry, output

    init_carry = 0.0
    xs = jnp.array([1.0, 2.0, 3.0])
    final_carry, outputs = jax.lax.scan(step, init_carry, xs)
    print(f"Scan final carry: {final_carry}")
    print(f"Scan outputs: {outputs}")

    # While Loop
    print("\n--- While Loop ---")
    cond_fun = lambda x: x < 10
    body_fun = lambda x: x + 1
    init_val = 0
    res = jax.lax.while_loop(cond_fun, body_fun, init_val)
    print(f"While Loop result: {res}")

    # Fori Loop
    print("\n--- Fori Loop ---")
    def body_fun(i, val):
        return val + i
    
    lower = 0
    upper = 10
    init_val = 0
    res = jax.lax.fori_loop(lower, upper, body_fun, init_val)
    print(f"Fori Loop result (sum 0..9): {res}")

if __name__ == "__main__":
    main()
