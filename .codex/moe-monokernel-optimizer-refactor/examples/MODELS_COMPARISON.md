# MoE Monokernel Model Comparison (Concise)

This document compares **high-leverage implementation decisions** across MoE models, without inlining large kernel dumps. Use it to generalize patterns and avoid semantic mistakes.

## Search anchors
model comparison, top_k, routing semantics, accumulation/reduction, ownership, FP8 block scales, scratchpad, split-h, EP.

## Reference implementations

| Model | Location | Hardware | Key characteristics |
|-------|----------|----------|--------------------|
| **Llama 4** | `assets/LLAMA4_MONOKERNEL_PATCH.md` | H100 (sm_90a) | `top_k=1`, BF16 activations, per-tensor/per-channel scales (reference patch) |
| **Qwen3-Coder-30B-A3B** | `csrc/moe/moe_monokernel/` | L40S (sm_89) | `top_k=8`, FP8 W8A8, block scales (example productionized path) |

## Architectural decisions comparison (reference runs)

| Model | E_local | EP enabled | Notes |
|-------|---------|------------|-------|
| Llama 4 | 16 | No | TP=8 reference |
| Qwen3 | 128 | No | TP=1 reference |

| Decision | Llama 4 (`top_k=1`) | Qwen3 (`top_k=8`) | When to use |
|----------|----------------------|-------------------|-------------|
| **Routing** | top-1 selection (often no normalize) | multi-expert + normalize | `top_k` drives selection + output semantics |
| **Accumulation** | direct write (no overlap) | accumulation required (many-to-1) | `top_k>1` usually needs accumulation or token-major ownership |
| **Weight application** | often foldable (model-dependent) | commonly post-activation | must match model semantics |
| **Scale layout** | simple scales (patch-specific) | block scales (e.g., 128×128) | must match quant format exactly |
| **Ownership** | token-major friendly | hybrid/expert-major common | depends on EP, average tokens/expert, and overlap |
| **Atomics required?** | typically no | only if output overlaps | avoid atomics if possible |

---

## Pattern 1: Routing (Top‑K selection + weights)

What varies most across MoE models:
- **Selection math**: softmax vs sigmoid (+bias) vs grouped routing; some models add correction bias before scoring.
- **Weights math**: softmax over selected logits vs sigmoid weights with optional renorm.
- **Tie-break**: must be deterministic (typically `(score desc, expert_id asc)`).

Where to look:
- Llama4 top-1 routing: `assets/LLAMA4_MONOKERNEL_PATCH.md`
- Qwen3 top-8 routing: `csrc/moe/moe_monokernel/src/moe_routing.cu`
- Canonical k-way merge guidance: `references/router-design.md`
- Planning/dispatch/kills: `references/optimization-techniques.md` → **T10**
- “Don’t guess semantics” checklist: `references/router-design.md`

Practical guidance:
- Treat “grouped routing” (expert groups) as a **separate semantics variant**; gate it with explicit shape/flag checks.
- If you change routing, validate:
  - ids determinism on equal logits
  - weights agreement vs baseline (within tolerance)
  - end-to-end MoE output agreement

---

## Pattern 2: Output accumulation and ownership

Core rule:
- If multiple expert contributions can land on the same output element, you need:
  - token-major ownership (avoid overlap), or
  - a safe accumulation strategy (FP32 scratch + reduce), or
  - atomics (last resort; often slow and/or numerically risky).

Where to look:
- Llama4 “direct write” style: `assets/LLAMA4_MONOKERNEL_PATCH.md`
- Qwen3 “accumulate then convert” style: `csrc/moe/moe_monokernel/src/moe_down_projection.cu`
- Decision logic for overlap/atomics: `references/algorithmic-branching.md`
- Structural templates: `references/code-templates.md`

Practical guidance:
- Prefer token-major accumulation when EP makes `E_local` small enough and the schedule can keep SMs busy.
- If you must use a scratch accumulator, keep the contract explicit (`pair_idx -> token_idx` mapping, zero-init rules, conversion point).

---

## Pattern 3: Scale layout (FP8 quant formats)

Scale layouts are model-format-specific; mismatches silently break correctness.

Common shapes:
- Per-tensor/per-channel scales (simpler, patch-specific): often indexed like `[expert, k]` or `[expert, channel]`.
- Block scales (harder): indexed like `[expert, k_block, n_block]` (e.g., 128×128 blocks).

Where to look:
- Qwen3 block scales handling: `csrc/moe/moe_monokernel/src/moe_down_projection.cu`
- Patch-specific scale handling: `assets/LLAMA4_MONOKERNEL_PATCH.md`
- Guardrails + patterns: `references/code-templates.md`

Practical guidance:
- Only “optimize” scale loading after you lock down exact indexing and broadcast rules.
- Keep a micro-test that dequantizes and compares against a reference path.

---

## Pattern 4: MMA path (what tensor-core instructions you’re really using)

What matters:
- Whether activations arrive as FP8/BF16/FP16/TF32, and what conversion chain your kernel uses.
- K-chunking and register pressure trade-offs (often dominate after you remove obvious extra kernels).

Where to look:
- Qwen3 FP8×FP8 path: `csrc/moe/moe_monokernel/src/ptx_utils.h`
- Llama4 patch MMA utilities: `assets/LLAMA4_MONOKERNEL_PATCH.md`
- Tiling and buffering heuristics: `references/tiling-config.md`

---

## Pattern 5: Underutilization fixes (Split‑H and grid shaping)

When `batch_size * top_k` is small relative to SM count, one block per (pair, tile) can underfill the GPU.

Where to look:
- Split-H heuristics/config: `csrc/moe/moe_monokernel/src/moe_interface.h`
- GPU count + occupancy expectations: `references/gpu-configs.md`

Practical guidance:
- Make Split-H (or another grid multiplier) a **shape-gated fast path**; never apply blindly.

---

## Takeaways

- Routing changes are correctness-fragile: gate strictly and encode semantics in code.
- For `top_k>1`, “ownership + accumulation” is usually the first hard decision; avoid atomics if possible.
- FP8 scale layout is non-negotiable; treat it as part of the semantic contract, not an implementation detail.
- If batch is small, fix grid underutilization before micro-tuning MMA.
