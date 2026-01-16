# Examples

## MultiLayerPerceptron

A configurable MLP with support for different normalization types.

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "flax",
#     "jax",
#     "jaxlib",
# ]
# ///
import jax.numpy as jnp
from flax import nnx
from typing import Callable, Optional

class MultiLayerPerceptron(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: int,
        hidden_units: int,
        *,
        rngs: nnx.Rngs,
        activation: Callable = nnx.relu,
        dropout_rate: float = 0.0,
        normalization: Optional[str] = None,
    ):
        layers = []
        self.dropout_rate = dropout_rate
        
        # Build hidden layers
        current_dim = in_features
        for _ in range(hidden_layers):
            # Dense Layer
            layers.append(nnx.Linear(current_dim, hidden_units, rngs=rngs))
            current_dim = hidden_units
            
            # Normalization (optional)
            if normalization == "layernorm":
                layers.append(nnx.LayerNorm(current_dim, rngs=rngs))
            elif normalization == "rmsnorm":
                layers.append(nnx.RMSNorm(current_dim, rngs=rngs))
            elif normalization is not None:
                raise ValueError(f"Unknown normalization: {normalization}")
            
            # Activation
            layers.append(activation)
            
            # Dropout (if rate > 0)
            if dropout_rate > 0:
                layers.append(nnx.Dropout(dropout_rate, rngs=rngs))
        
        # Output Layer
        layers.append(nnx.Linear(current_dim, out_features, rngs=rngs))
        
        # Store as nnx.List to allow iteration and module tracking
        self.layers = nnx.List(layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    import jax
    
    # Setup
    key = jax.random.key(0)
    rngs = nnx.Rngs(0)
    
    # Parameters
    IN_DIM = 10
    OUT_DIM = 2
    HIDDEN_LAYERS = 2
    HIDDEN_UNITS = 32
    DROPOUT = 0.5
    
    # Test 1: LayerNorm
    print("Initializing MultiLayerPerceptron (LayerNorm)...")
    model_ln = MultiLayerPerceptron(
        in_features=IN_DIM,
        out_features=OUT_DIM,
        hidden_layers=HIDDEN_LAYERS,
        hidden_units=HIDDEN_UNITS,
        dropout_rate=DROPOUT,
        normalization="layernorm",
        rngs=rngs
    )
    
    dummy_input = jnp.ones((1, IN_DIM))
    print(f"LayerNorm Output: {model_ln(dummy_input)}")

    # Test 2: RMSNorm
    print("\nInitializing MultiLayerPerceptron (RMSNorm)...")
    model_rms = MultiLayerPerceptron(
        in_features=IN_DIM,
        out_features=OUT_DIM,
        hidden_layers=HIDDEN_LAYERS,
        hidden_units=HIDDEN_UNITS,
        dropout_rate=DROPOUT,
        normalization="rmsnorm",
        rngs=rngs
    )
    print(f"RMSNorm Output: {model_rms(dummy_input)}")
    
    # Validation
    model_ln.train()
    out_train_1 = model_ln(dummy_input)
    out_train_2 = model_ln(dummy_input)
    is_diff = not jnp.allclose(out_train_1, out_train_2)
    print(f"\nDropout Check (LayerNorm): {is_diff}")

    print("\nSuccess! MultiLayerPerceptron created and verified with normalization switching.")
```

**Source:** [scripts/mlp.py](scripts/mlp.py)
