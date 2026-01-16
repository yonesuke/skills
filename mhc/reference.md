# mHC Technical Reference

## 1. Motivation for Introduction of HC (Hyper-Connections) and its Issues

### Motivation: Breaking through the Limits of ResNet

Conventional Residual Connections were the key to the success of deep learning, but they suffered from a structural problem: a **seesaw game between "Gradient Vanishing" and "Representation Collapse"**.

*   **HC Approach**: By expanding the "width" of the residual stream by a factor of $M$ and controlling the connection strength between layers with a dynamically learnable matrix ($B$), it made the flow of information flexible and improved performance.

### Issues with HC

While HC brought performance improvements, fatal flaws were revealed in large-scale models and deep models.

1.  **Loss of Identity Mapping**:
    *   The core property of residual connections, "if nothing is done, the signal is transmitted as is (identity mapping)," is destroyed by the learnable matrix $B$.
    *   As a result, as layers are stacked, signals explode or vanish, making learning unstable (occurrence of Loss spikes).

2.  **Memory and Communication Overhead**:
    *   Since the residual stream becomes $M$ times larger, memory bandwidth (I/O) consumption increases drastically (Memory Wall problem).
    *   Communication costs increase during distributed learning such as pipeline parallelism.

---

## 2. Advantages of mHC (Manifold-Constrained HC)

mHC solves the problems by introducing **"Manifold Constraints"** while maintaining the expressiveness of HC.

*   **Stability of Learning**: By projecting $B$ onto a specific manifold (doubly stochastic matrices), it regains properties close to identity mapping and preserves the norm of the signal.

*   **Scalability**: Even in large-scale models (e.g., 27B parameters), it does not cause instability like HC and can be trained with stability equivalent to ResNet.

*   **Practical Efficiency**: Through system optimizations such as dedicated Kernel Fusion, Recomputing, and Communication Overlap (DualPipe), the increase in calculation and memory costs is minimized (e.g., about +6.7% time increase).

---

## 3. Theoretical Explanation: Why "Doubly Stochastic Matrix"?

In mHC, the residual mapping matrix $B$ is constrained to the set of **Doubly Stochastic Matrices** (Birkhoff polytope).

Theoretical Characteristics

1.  **Norm Preservation**:
    *   The spectral norm of a doubly stochastic matrix is 1 or less ($||B||_2 \le 1$). This theoretically prevents signal explosion (gradient explosion).

2.  **Compositional Closure**:
    *   The product of doubly stochastic matrices is also a doubly stochastic matrix. This ensures that the property is maintained even if many layers are stacked ($B_1 B_2 \dots B_L$), stabilizing deep models.

3.  **Geometric Interpretation**:
    *   This can be interpreted as a "convex combination of permutation matrices," playing the role of appropriately "mixing" information between multiple streams while preserving the total amount of information (average) as a whole.

---

## Appendix: Implementation Notes

*   **Kernel Fusion**: In the paper, custom kernels fusing these operations (Norm, Projection, Sinkhorn, Update) are created using TileLang etc. to speed up processing.

*   **Mixed Precision**: Proper use of operation precision (float32/bfloat16) is also considered important for stability and speed.
