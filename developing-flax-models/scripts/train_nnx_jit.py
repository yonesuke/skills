# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "flax",
#     "jax",
#     "jaxlib",
#     "optax",
# ]
# ///
from flax import nnx
import jax
import jax.numpy as jnp
import optax

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
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer, x, y):
        def loss_fn(model):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)
        
        grad = nnx.grad(loss_fn)(model)
        optimizer.update(model, grad)
        return loss_fn(model)

    # Usage
    x_batch = jnp.ones((16, 1))
    y_batch = x_batch * 2.0 # Target function is y = 2x
    
    for i in range(10):
        loss = train_step(model, optimizer, x_batch, y_batch)
        print(f"Step {i}, Loss: {loss}")
