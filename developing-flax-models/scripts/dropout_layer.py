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

class DropoutUsage(nnx.Module):
    def __init__(self, rate: float, *, rngs: nnx.Rngs):
        # nnx.Dropout is state-aware. It uses 'dropout' collection by default.
        self.dropout = nnx.Dropout(rate, rngs=rngs)

    def __call__(self, x):
        # No conditional logic needed here!
        # The Layer itself checks self._graph_node.is_deterministic (handled by .eval()/.train())
        return self.dropout(x)

if __name__ == "__main__":
    rngs = nnx.Rngs(0)
    model = DropoutUsage(0.5, rngs=rngs)
    x = jnp.ones((1, 10))
    
    # 1. Default Mode (Usually Eval/Deterministic dependent on initialization?)
    # Validating default behavior. 
    # Usually modules start in Train mode? Or Indeterminate?
    # Best practice: Explicitly set mode.
    
    print("--- Train Mode ---")
    model.train() # Enable Dropout
    y_train = model(x)
    print(f"Output (Dropout active): {y_train}")
    
    print("\n--- Eval Mode ---")
    model.eval() # Disable Dropout
    y_eval = model(x)
    print(f"Output (Dropout inactive): {y_eval}")
    
    # Validation
    assert jnp.array_equal(y_eval, x), "Eval mode should be identity for Dropout"
    assert not jnp.array_equal(y_train, x), "Train mode should apply mask (with high prob)"
