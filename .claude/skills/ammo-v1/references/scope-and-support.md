# Scope and Support (Planner-Level)

## Contents
- Supported data types
- Target components

## Supported data types

Use this as a planning constraint only; always confirm the actual model's weight/activation/scale dtypes in Stage 1.

| Type | Element size | Typical MMA path | Notes |
|------|--------------|------------------|-------|
| FP8 E4M3 | 1 byte | FP8 Tensor Core MMA (sm_89+) | Common inference target on Ada/Hopper-class GPUs |
| BF16 | 2 bytes | BF16 Tensor Core MMA (sm_80+) | Higher SMEM/register footprint than FP8 |
| FP16 | 2 bytes | FP16 Tensor Core MMA (sm_80+) | Compatibility fallback |
| MXFP4 | 0.5 bytes | Experimental | Treat as out-of-scope unless explicitly required |

For exact intrinsics/PTX patterns, see `references/code-templates.md` (MMA templates section).

## Target components

This workflow targets **any vLLM kernel component** including but not limited to:

| Component | Example kernels | Typical vLLM code path |
|-----------|----------------|----------------------|
| MoE | `fused_moe`, `fused_experts`, `topk_softmax` | `vllm/model_executor/layers/fused_moe/` |
| Attention | `flash_attn`, `paged_attention`, FlashInfer kernels | `vllm/v1/attention/backends/` |
| KV Cache | `reshape_and_cache`, page table ops | `vllm/attention/` |
| Sampling | `topk`, `softmax`, sampling kernels | `vllm/model_executor/layers/sampler.py` |
| Quantization | `marlin`, `gptq`, `awq` kernels | `vllm/model_executor/layers/quantization/` |
| FFN | Linear/GEMM kernels, activation fusions | `vllm/model_executor/layers/` |
| Custom | Any user-identified kernel | Varies |

The optimization approach and validation methodology are the same regardless of target component. The key difference is which kernel times and correctness invariants are tracked.
