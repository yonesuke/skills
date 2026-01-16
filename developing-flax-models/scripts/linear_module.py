# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "flax",
#     "jax",
#     "jaxlib",
# ]
# ///
from flax import nnx
import jax
import jax.numpy as jnp

class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        # Initializers take a Key (generated from rngs)
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))

    def __call__(self, x):
        return x @ self.w + self.b

# Usage
if __name__ == "__main__":
    rngs = nnx.Rngs(0) # Seed 0
    model = Linear(10, 5, rngs=rngs)
    x = jnp.ones((1, 10))
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Output: {y}")
