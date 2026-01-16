# JAX Examples
Usage patterns for JAX.

## 1. Core Transformations
Demonstrates `jit`, `grad`, `vmap`, and `random`.

**Source:** [scripts/core_transformations.py](scripts/core_transformations.py)
```python
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
```

## 2. ML Specifics (Pytrees)
JAX works on Pytrees (nested dicts/lists/tuples/dataclasses of arrays).

**Source:** [scripts/pytree_demo.py](scripts/pytree_demo.py)
```python
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
```

## 3. Control Flow
`cond`, `scan` (Efficient Loop), and `while_loop`.

**Source:** [scripts/control_flow_demo.py](scripts/control_flow_demo.py)
```python
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
```

## 4. Parallelism
Distributed arrays across devices (Mesh).

**Source:** [scripts/sharding_demo.py](scripts/sharding_demo.py)
```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax", "jaxlib"]
# ///

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import os

def main():
    print("Running Sharding Demo")
    
    # Check if we have enough devices, otherwise simulate or warn
    n_devices = jax.device_count()
    print(f"Available devices: {n_devices}")
    
    # We will use what's available
    mesh_shape = (1, 1)

    try:
        # 1. Create Mesh
        devices = mesh_utils.create_device_mesh(mesh_shape) 
        mesh = Mesh(devices, axis_names=('data', 'model'))

        # 2. Define Sharding
        sharding = NamedSharding(mesh, PartitionSpec('data', 'model'))

        # 3. Create/Place Array
        x = jax.device_put(jnp.zeros((128, 64)), sharding)
        print(f"Sharded array shape: {x.shape}")
        print("Sharding successful (simulated on available devices)")
        
    except Exception as e:
        print(f"Sharding demo failed: {e}")

if __name__ == "__main__":
    main()
```
