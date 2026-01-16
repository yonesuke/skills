# /// script
# requires-python = ">=3.12"
# dependencies = ["jax", "jaxlib"]
# ///

import jax
import jax.numpy as jnp
from functools import partial

def main():
    print("Running JAX Core Transformations Demo")

    # 1. JIT
    print("\n--- JAX JIT ---")
    @jax.jit
    def selu(x, alpha=1.67, lmbda=1.05):
        return lmbda * jax.nn.elu(x, alpha=alpha)

    x = jnp.arange(5.0)
    print(f"SELU output: {selu(x)}")

    # Static Arguments
    @partial(jax.jit, static_argnames=['mode'])
    def f(x, mode):
        if mode == 'train': return x
        else: return 0
    
    print(f"Static arg 'train': {f(10, 'train')}")
    print(f"Static arg 'test': {f(10, 'test')}")

    # 2. Grad
    print("\n--- JAX Grad ---")
    def loss_fn(params, x, y):
        pred = params['w'] * x + params['b']
        return jnp.sum((pred - y) ** 2)

    params = {'w': 2.0, 'b': 1.0}
    x_in = jnp.array([1.0, 2.0])
    y_true = jnp.array([3.0, 5.0]) # Perfect fit (2*1+1=3, 2*2+1=5)

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params, x_in, y_true)
    print(f"Gradients: {grads}")

    loss, grads = jax.value_and_grad(loss_fn)(params, x_in, y_true)
    print(f"Loss: {loss}, Grads: {grads}")

    # Higher Order (Hessian of a simple function)
    print("\n--- Higher Order ---")
    def cube(x): return x ** 3
    hessian = jax.jacfwd(jax.jacrev(cube))(2.0) # 3x^2 -> 6x -> 12
    print(f"Hessian of x^3 at x=2: {hessian}")

    # 3. Vmap
    print("\n--- JAX Vmap ---")
    def f_simple(x, param):
        return x * param

    batch_x = jnp.array([1.0, 2.0, 3.0])
    # param is not batched (None), x is batched at axis 0
    batch_f = jax.vmap(f_simple, in_axes=(0, None), out_axes=0)
    preds = batch_f(batch_x, 2.0)
    print(f"Vnames output: {preds}")

    # 4. Random
    print("\n--- JAX Random ---")
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    noise = jax.random.normal(key1, shape=(3, 3))
    print(f"Random noise shape: {noise.shape}")

if __name__ == "__main__":
    main()
