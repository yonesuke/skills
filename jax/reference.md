# JAX Reference

## 1. Core Philosophy
*   **Immutability**: JAX arrays are immutable. ``x[0] = 1`` raises error. Use ``x.at[0].set(1)``.
*   **Functional**: Functions should be **Pure** (Output depends only on input, no side effects).
*   **Accelerated**: Runs on XLA (Accelerated Linear Algebra) for GPU/TPU.

## 2. The Four Transformations
1.  **`jit`**: Compiles `f(x)` -> `xla_f(x)`. Speedup.
2.  **`grad`**: Differentiates `f(x)` -> `f'(x)`. Autograd.
3.  **`vmap`**: Vectorizes `f(x)` -> `f(batch_x)`. Automatic Batching.
4.  **`pmap`** (Legacy) / **`sharding`**: Parallelizes across devices.

## 3. Control Flow
*   **Python Control Flow** (`if x > 0`): Logic is baked into the graph during tracing. `x` must be static (value known at compile time).
*   **JAX Control Flow** (`lax.cond`, `lax.scan`, `lax.fori_loop`): Logic remains in the graph. `x` can be dynamic (tracer).

## 4. Pytrees
JAX generic concept for "container of arrays". JAX ops work over pytrees seamlessly.
*   Supported: `list`, `tuple`, `dict`, `NamedTuple`, custom classes registered as nodes.
*   Common util: `jax.tree.map(fn, tree)` applies function to leaves.
