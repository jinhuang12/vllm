# Scope and Support (Planner-Level)

## Contents
- Supported data types
- Validated examples

## Supported data types

Use this as a planning constraint only; always confirm the actual model’s weight/activation/scale dtypes in Phase 1.

| Type | Element size | Typical MMA path | Notes |
|------|--------------|------------------|-------|
| FP8 E4M3 | 1 byte | FP8 Tensor Core MMA (sm_89+) | Common inference target on Ada/Hopper-class GPUs |
| BF16 | 2 bytes | BF16 Tensor Core MMA (sm_80+) | Higher SMEM/register footprint than FP8 |
| FP16 | 2 bytes | FP16 Tensor Core MMA (sm_80+) | Compatibility fallback |
| MXFP4 | 0.5 bytes | Experimental | Treat as out-of-scope unless explicitly required |

For exact intrinsics/PTX patterns, see `references/code-templates.md` (MMA templates section).

## Validated examples

Validated examples exist for these model architectures (the workflow targets any vLLM MoE that matches the `fused_moe` interface):

| Model | `top_k` | Hardware | Quantization | Key patterns |
|-------|--------:|----------|--------------|--------------|
| **Llama-4-Scout** | 1 | H100 (sm_90a) | Per-tensor FP8 | Direct write, TMA prefetch |
| **Qwen3-Coder-30B-A3B** | 8 | L40S (sm_89) | 128×128 block FP8 | FP32 accumulator, Split-H, `cp.async` |

See `examples/MODELS_COMPARISON.md` for detailed pattern notes.
