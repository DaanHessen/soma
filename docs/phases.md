# SOMA Development Phases

To ensure perfection and reliability, development is strictly parallelized into small, testable phases. We will not proceed to the next phase until the current one is mathematically and programmatically verified.

## CHECKLIST (check an item when it is done AND tested, before continuing to the next step/phase)

- [x] phase 1
- [x] phase 2
- [ ] phase 3
- [ ] phase 4
- [ ] phase 5
- [ ] phase 6
- [ ] phase 7

## Phase 1: Core Foundation & Mock Flow Matching ✅
**Goal:** Establish the Rust project, the `burn` ML backend, and the core loss mathematics.
- **Steps:**
  - [x] Setup cargo with `burn` (CUDA/WGPU features) and `ort`.
  - [x] Implement the Flow Matching (Velocity Prediction) loss function.
  - [x] Build a dummy MLP/Linear model to verify that the Burn Autodiff calculates gradients correctly using the loss function.
- **Verification:**
  - [x] Unit tests confirming loss converges to 0 on synthetic $x_0$ and $x_1$ tensors.

## Phase 2: Segmentation Engine Integration ✅
**Goal:** Get ONNX-based segmentation models running purely in Rust and transform their outputs.
- **Steps:**
  - [x] Initialize ONNX Runtime (`ort`) and load a tiny segmentation model (e.g., YOLOv8n-seg).
  - [x] Feed an image through the module and extract mask tensors.
  - [x] Implement Mask shape transformations (downsampling, broadcasting to 4 channels).
- **Verification:**
  - [x] Feed a test image, save the output mask as an image/array, and visually/programmatically confirm the downsampling aligns properly.

## Phase 3: SOMA Tiered Backward Pass
**Goal:** Prove we can intercept and zero-out specific semantic gradients in Burn.
- **Steps:**
  - Combine Phase 1 (Loss) and Phase 2 (Masks) into a `SemanticWeightedLoss` function.
  - Implement the `grad_replace` logic during the backward pass to scale specific spatial gradients.
- **Verification:** Check tensor gradients manually. Confirm that parameters responsible for "background" paths in a dummy model receive exactly 0 gradient update.

## Phase 4: VAE & Latent Pipeline
**Goal:** Process real images into latent space.
- **Steps:**
  - Load a VAE (via `candle` or `ort` with safetensors).
  - Encode standard $512 \times 512$ images into $64 \times 64$ latent tensors.
  - Decode them back to ensure fidelity.
- **Verification:** Round-trip test: encode an image and decode it. The result must visibly match the input.

## Phase 5: T-LoRA & The Model Architecture
**Goal:** Embed the Timestep-Dependent LoRA (T-LoRA) adapters into a Transformer structure.
- **Steps:**
  - Define the `TLoraLayer` wrapping standard linear layers via Burn modules.
  - Implement the dynamic rank function $r(t)$ to apply masking to SVD-initialized matrices during the forward pass.
- **Verification:** Assert that the number of active parameters in the LoRA adapters correctly scales down as simulated timestep $t$ decreases.

## Phase 6: Prodigy Optimizer Port
**Goal:** Implement the adaptive Prodigy optimizer in Rust.
- **Steps:**
  - Port the update logic from PyTorch/Optax into a Burn `Optimizer` trait.
  - Implement safeguard warmup and dynamic learning rate distance calculations.
- **Verification:** Train a reference model (e.g., MNIST on Burn) with our custom Prodigy and confirm convergence comparable to AdamW.

## Phase 7: End-to-End Orchestration
**Goal:** Pull everything together into the final CLI training loop.
- **Steps:**
  - Connect text encoders, latents, diffusion UNet/Flux, Masks, Loss, and Optimizer.
  - Add Safetensors checkpoint saving.
  - Integrate terminal logging (tqdm-like progress, VRAM metrics).
- **Verification:** Run an end-to-end dry training loop on 1 image for 10 steps. Verify memory doesn't leak and checkpoints generate valid safetensor bins.
