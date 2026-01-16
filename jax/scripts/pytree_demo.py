# /// script
# requires-python = ">=3.12"
# dependencies = ["jax", "jaxlib"]
# ///

import jax
import jax.numpy as jnp

def main():
    print("Running Pytree Demo")
    
    params = {'W': jnp.ones((2, 2)), 'b': jnp.zeros(2)}
    grads = {'W': jnp.array([[0.1, 0.2], [0.3, 0.4]]), 'b': jnp.array([0.1, 0.1])}

    # Gradient Descent Update
    print("Params before:", params)
    new_params = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)
    print("Params after:", new_params)

if __name__ == "__main__":
    main()
