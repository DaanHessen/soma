# SOMA Backend Architecture Mapping

Based on the research and sequence diagram, this document maps out precisely how the SOMA (Semantic Optimized Masked Adaptation) training pipeline will work in Rust.

## Process Flow (Refined Sequence)

The standard diffusion sequence diagram missed the distinction between DDPM (noise prediction) and Conditional Flow Matching (velocity prediction used in Flux.1). The pipeline logic is mapped as follows:

1. **Initialization:**
   - CLI parses arguments and loads `Config`.
   - `Dataset` loads image-caption pairs into memory streams.

2. **Data Preparation (Forward Pass Part 1):**
   - **Latent Encoding:** Images are passed through a VAE (via `candle` or `ort`) to produce latent tensors ($x_1$).
   - **Segmentation:** Images are simultaneously passed to YOLOv8-seg/SAM 2 via ONNX (`ort`). Masks are generated.
   - **Mask Transformation:** Masks are downsampled (nearest-neighbor) and broadcasted to match the latent space dimensions ($64 \times 64$).

3. **Flow Matching Setup:**
   - Sample noise $x_0 \sim \mathcal{N}(0, I)$.
   - Sample timestep $t \sim \mathcal{U}[0, 1]$.
   - Compute interpolated active latent: $x_t = (1 - t)x_0 + t x_1$.
   - The "Ground Truth" velocity is calculated as: $v_{target} = x_1 - x_0$.

4. **Model Forward Pass:**
   - The Base Model (e.g., Flux.1) with T-LoRA adapters takes $x_t, t$ and text conditioning (from Text Encoders).
   - T-LoRA adapters apply $r(t)$ dynamically: reducing the active rank as $t \to 0$ (noisier space).
   - Output is predicted velocity: $v_\theta(x_t, t)$.

5. **Semantic Loss Calculation:**
   - Compute base squared error: $|| v_{target} - v_\theta(x_t, t) ||^2$.
   - Spatial weights from the `MaskUtils` (semantic masks) are multiplied pixel-by-pixel against the error matrix.
   - Calculate Cross-Attention Penalty using attention maps to ensure token grounding.

6. **Tiered Backward Pass:**
   - Burn's Autodiff computes gradients via `.backward()`.
   - Before passing to the optimizer, iterating through parameters:
     - Use `grad_replace` inside `Burn`'s AutodiffBackend to intercept layer-specific gradients.
     - Scale/Zero background gradients based on the semantic tiered logic.

7. **Optimization (Prodigy):**
   - *Note: Since Prodigy has no native Rust implementation, it will need to be ported manually via Burn's `Optimizer` trait.*
   - Apply adaptive step size based on distance-to-optimal estimations.
   - Update weights and log to Terminal/Metrics.
