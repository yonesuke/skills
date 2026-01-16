# mHC Examples

Usage patterns for Manifold-Constrained Hyper-Connections.

## 1. Minimal Implementation
Shows the core `sinkhorn_knopp` algorithm and the `mhc_layer_forward` pass.

**Source:** [scripts/implementation.py](scripts/implementation.py)
```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax", "jaxlib"]
# ///

import jax
import jax.numpy as jnp

def rms_norm(x, eps=1e-6):
    """Simple RMSNorm implementation."""
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)

def sinkhorn_knopp(log_matrix, n_iters=20):
    """
    Projects a matrix onto the Doubly Stochastic Matrix manifold using Sinkhorn-Knopp algorithm.
    Input is assumed to be in log-space or pre-exp space for numerical stability, 
    starting from exp(M) according to Eq.(9) of the paper.
    """
    # Apply exp to get positive values (Eq.9 M(0) = exp(H_res_tilde))
    M = jnp.exp(log_matrix)
    
    def body_fun(i, mat):
        # Row Normalization
        mat = mat / jnp.sum(mat, axis=1, keepdims=True)
        # Column Normalization
        mat = mat / jnp.sum(mat, axis=0, keepdims=True)
        return mat

    # Iterate (t_max=20 in the paper)
    doubly_stochastic_matrix = jax.lax.fori_loop(0, n_iters, body_fun, M)
    return doubly_stochastic_matrix

def mhc_layer_forward(x_l, params, layer_function):
    """
    Forward pass of the mHC (Manifold-Constrained Hyper-Connection) layer.
    
    Args:
        x_l: Input hidden state [Batch, n_streams, C]
        params: Trainable parameters dictionary
        layer_function: Function F(x) representing the layer (e.g. TransformerBlock)
    """
    n_streams = x_l.shape[-2]
    
    # 1. Pre-processing: Flatten & RMSNorm (Eq. 7)
    x_flat = x_l.reshape(x_l.shape[0], -1) 
    x_normed = rms_norm(x_flat)

    # 2. Compute Mapping Coefficients (Eq. 7)
    # phi_pre, phi_post, phi_res are parameter matrices
    h_tilde_pre = jnp.dot(x_normed, params['phi_pre']) + params['b_pre']
    h_tilde_post = jnp.dot(x_normed, params['phi_post']) + params['b_post']
    
    h_tilde_res_flat = jnp.dot(x_normed, params['phi_res']) + params['b_res']
    h_tilde_res = h_tilde_res_flat.reshape(-1, n_streams, n_streams)

    # 3. Apply Manifold Constraints (Eq. 8, 17-19)
    # Pre/Post use Sigmoid (Post is scaled by 2.0)
    # Res uses Sinkhorn-Knopp
    
    H_pre = jax.nn.sigmoid(h_tilde_pre)              # [Batch, n]
    H_post = 2.0 * jax.nn.sigmoid(h_tilde_post)      # [Batch, n]
    
    # Apply Sinkhorn-Knopp per batch element
    H_res = jax.vmap(sinkhorn_knopp)(h_tilde_res)    # [Batch, n, n]

    # 4. Application and Residual Connection (Eq. 3 mHC version)
    
    # Pre-mapping: H_pre * x_l -> Input to layer F
    # Aggregate streams: [Batch, n, 1] * [Batch, n, C] -> sum -> [Batch, C]
    layer_input = jnp.einsum('bn,bnc->bc', H_pre, x_l)
    
    # Apply Layer Function F
    layer_output = layer_function(layer_input) # [Batch, C]
    
    # Post-mapping: Distribute F(x) back to streams
    # [Batch, n, 1] * [Batch, 1, C] -> [Batch, n, C]
    post_term = jnp.einsum('bn,bc->bnc', H_post, layer_output)
    
    # Residual mapping: Mix streams using Doubly Stochastic Matrix H_res
    # [Batch, n, n] * [Batch, n, C] -> [Batch, n, C]
    res_term = jnp.einsum('bnm,bmc->bnc', H_res, x_l)
    
    # Next layer input x_{l+1}
    x_next = res_term + post_term
    
    return x_next

def main():
    print("Running mHC Implementation Demo")
    
    # Simulation Parameters
    B, N, C = 2, 4, 8 # Batch, Streams, Channels
    key = jax.random.PRNGKey(42)
    
    # Dummy Input
    x_l = jax.random.normal(key, (B, N, C))
    
    # Dummy Params
    # Input to coefficient generators is flattened x_l (size N*C)
    flat_dim = N * C
    
    def init_linear(k, in_d, out_d):
        w = jax.random.normal(k, (in_d, out_d)) * 0.02
        b = jnp.zeros(out_d)
        return w, b

    k1, k2, k3, k4 = jax.random.split(key, 4)
    phi_pre, b_pre = init_linear(k1, flat_dim, N)
    phi_post, b_post = init_linear(k2, flat_dim, N)
    phi_res, b_res = init_linear(k3, flat_dim, N * N)
    
    params = {
        'phi_pre': phi_pre, 'b_pre': b_pre,
        'phi_post': phi_post, 'b_post': b_post,
        'phi_res': phi_res, 'b_res': b_res
    }
    
    # Dummy Layer Function (Identity for simplicity)
    def layer_fn(x):
        return x * 2.0
    
    # Forward Pass
    x_next = mhc_layer_forward(x_l, params, layer_fn)
    
    print(f"Input shape: {x_l.shape}")
    print(f"Output shape: {x_next.shape}")
    print("Execution successful.")

if __name__ == "__main__":
    main()
```
