---
name: developing-flax-models
description: A comprehensive guide for developing, training, and managing neural networks using Flax NNX. Use when defining models, managing state, or writing training loops.
---

# Developing Flax Models

This skill helps you develop neural networks using **Flax NNX**, the object-oriented module system for JAX. Use this skill when you need to define models, handle state/randomness, or implement training loops.

## Workflows

### 1. Implement a New Model
1.  **Define Module**: Subclass `nnx.Module`. Define layers/parameters in `__init__`.
2.  **Handle Randomness**: Pass `nnx.Rngs` to `__init__` for weight initialization. Pass `rngs` to `__call__` for stochastic operations (e.g., Dropout).
3.  **Sanity Check**: Add a `if __name__ == "__main__":` block to instantiate the model and run a dummy forward pass.

### 2. Implement a Training Loop
1.  **Choose Strategy**:
    *   **Automatic (`nnx.jit`)**: Easiest. Use `@nnx.jit` on your update function. Handles mutable state management automatically.
    *   **Functional (`nnx.split`/`nnx.merge`)**: Use for advanced control or when interfacing with pure JAX transformations like `scan` or `vmap` (though `nnx.vmap` exists).
2.  **Define Loss**: Write a loss function `loss_fn(model, batch)`.
3.  **Optimizer**: Wrap the model with `nnx.Optimizer(model, tx, wrt=nnx.Param)`.
    > [!WARNING]
    > **Crucial Change**: As of Flax 0.11.0, the `wrt` argument (e.g., `wrt=nnx.Param`) is **REQUIRED** for `nnx.Optimizer`. Failure to include it will raise a `TypeError`.

## Core Concepts (Reference)

**Flax NNX** (v2.0+) replaces the immutable, functional design of `flax.linen` with standard Python classes and mutable state, while maintaining JAX compatibility.

### Key Differences
*   **Object-Oriented**: Models are standard Python classes. You assign to `self.param`.
*   **Reference Semantics**: Layers hold their parameters directly.
*   **Not Pytrees**: `nnx.Module` objects are **not** Pytrees. You cannot pass them directly to `jax.jit`. You must use `nnx.jit` or manually split/merge state.

### Variable Types
NNX variables allow granular state management via "Collections".
*   `nnx.Param`: Trainable parameters (weights, biases).
*   `nnx.BatchStat`: Batch normalization statistics (running mean/var).
*   `nnx.Rngs`: Random Number Generator streams key management.
*   `nnx.Variable`: Base class for custom state.

### State Management
You can filter and manipulate state sets:
```python
# Get only Parameters
params = nnx.state(model, nnx.Param)

# Get everything EXCEPT BatchStats
state = nnx.state(model, filter=nnx.All - nnx.BatchStat)
```

## Examples
See [examples.md](examples.md) for detailed code patterns mirrored from the `scripts/` directory.
*   **Defining Modules**: Basic layer structure.
*   **Randomness**: Handling Dropout and stochastic layers.
*   **Training**: Comparison of `nnx.jit` vs Pure JAX loops.
*   **Functional API**: Using `vmap` and `split`/`merge`.
*   **MultiLayerPerceptron**: Building complex modules with variable depth and conditional layers.
