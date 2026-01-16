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

# Simple Linear model for demonstration
class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))

    def __call__(self, x):
        return x @ self.w + self.b

if __name__ == "__main__":
    rngs = nnx.Rngs(0)
    model = Linear(1, 1, rngs=rngs)

    # 1. Split
    graphdef, state = nnx.split(model)

    @jax.jit
    def pure_train_step(graphdef, state, x, y):
        # 2. Merge inside JIT
        model = nnx.merge(graphdef, state)
        
        def loss_fn(model):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        grad = nnx.grad(loss_fn)(model)
        # Update logic would go here...
        # For demo purposes, we just return the gradients or same state if no update implemented
        
        # 3. Split again to return new state
        _, new_state = nnx.split(model)
        return new_state, loss_fn(model)

    x = jnp.ones((16, 1))
    y = x * 2.0
    
    state, loss = pure_train_step(graphdef, state, x, y)
    print(f"Loss: {loss}")
    
    # Merge back for interactive use
    nnx.update(model, state)
