# Worked Example (Generic): W1 epilogue fusion

**Goal**: Fuse W1 GEMM epilogue: activation (SiLU/SwiGLU) + W2‑input quantization (FP8) into the W1 kernel so you delete real kernel work and avoid large intermediate writes/reads.


## W1 Triton epilogue fusion (generic)

### Target to fuse

Baseline often does:
1) W1 expert GEMM writes `[M*top_k, 2*N_local]` to global
2) activation kernel computes `[M*top_k, N_local]` (SiLU/SwiGLU)
3) quant kernel converts activations to FP8 for W2 input (+ scales)
4) W2 expert GEMM consumes FP8 activations

**Hybrid fusion goal**:
- Keep the baseline expert GEMM structure and grid shape
- In the W1 kernel, after producing the W1 output tile:
  - apply activation in registers
  - quantize to FP8 for W2 input
  - write only the FP8 activations (+ scale metadata)

This removes at least one large intermediate write/read and deletes 1–2 kernels.

### Preconditions / constraints

- You can compute activation from W1 outputs without needing a global sync.
- You can produce the same quantization metadata expected by W2:
  - per-tensor scales (simpler; common in FP8 MoE)
  - per-block / grouped scales (harder; requires matching baseline layout exactly)
- Register pressure must not explode; epilogue work should be “light” compared to GEMM.

### Triton implementation sketch (high-level)

1) Locate the Triton expert GEMM kernel used by the baseline (often called from `fused_experts`).
2) Add a kernel variant flag (or a separate kernel) that:
   - computes W1 tile
   - performs activation:
     - **SwiGLU**: `y = silu(gate) * up` (gate/up are halves of the `2*N_local` output)
     - **SiLU+mul**: same idea (depends on model)
   - quantizes `y` to FP8:
     - compute `amax` reduction for the quantization group
     - compute scale and apply FP8 cast
     - write FP8 and any needed scale tensor
3) Wire the new kernel into the fused MoE call path under an env var gate.

### Make it generic (don’t bake in one model)

To keep W1 epilogue fusion generic across MoE models, parameterize only what actually varies:
- activation type (SiLU / SwiGLU / GELU-family)
- quantization format (per-tensor vs per-block scales, dtype)
- where routing weights are applied (pre/post activation), and whether weights can be folded safely

Everything else (grid shape, expert grouping, weight layouts) should be inherited from the baseline expert GEMM call path for that model/config.

### What “success” must look like (profiling)

Under CUDA graphs (production parity):
- Nsight Systems kernel list should show activation/quant kernels removed or reduced.
- Kernel-level timing should improve **without** reducing GEMM concurrency (check achieved occupancy / spills).

### What can go wrong (and how to catch it)

- **Already fused baseline**: you implemented work that was already in the GEMM; no win.
  - Catch with nsys kernel list before coding.
- **Spills / lower occupancy**: epilogue increases registers, drops occupancy, kernel slows.
  - Catch with NCU (registers/spills, achieved occupancy, stall reasons).
- **Quantization mismatch**: scale semantics differ → output mismatch or silent accuracy loss.
  - Catch with dequant+compare vs baseline and end-to-end tests.
