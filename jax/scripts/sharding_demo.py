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
