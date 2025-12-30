# MoE Parameters & Semantics Template

## Contents
- Required fields
- How to extract from vLLM
- Output template (copy/paste)

## Search anchors
moe parameters, scoring_func, norm_topk_prob, routed_scaling_factor, shared experts, tie-break, TP, EP, E_local, quantization_config

## Required fields (do not guess)
Fill these by reading the model’s vLLM implementation (`vllm/model_executor/models/...`) and config.

### A) Model + environment
- Model ID:
- vLLM commit hash:
- GPU (SM arch) + driver + CUDA:
- Production knobs (must match baseline):
  - vLLM `-O` optimization level:
  - vLLM `-cc/--compilation-config` string (cudagraph_mode + capture sizes):
  - torch.compile enabled? (vLLM v1 defaults to enabled)

### B) Geometry + parallelism
- hidden_size (K):
- intermediate_size (N):
- num_experts (E_global):
- top_k:
- num_shared_experts (0 if none):
- TP:
- EP:
- E_local (after TP/EP dispatch):
- Prefill vs decode: which regimes are in-scope for optimization?

### C) Routing semantics (correctness-critical)
- scoring_func: softmax / sigmoid / other
- norm_topk_prob: True/False (renormalize weights?)
- bias placement (if any): before scoring / after scoring / none
- routed_scaling_factor (if present): value + where applied
- tie-break rule for equal scores: (score desc, expert_id asc) OR “match baseline exactly”
- output of router: logits vs probs; dtype used for gating math

### D) Weight timing & accumulation semantics
- Are routing weights applied:
  - before activation?
  - after activation?
  - folded into W2?
- Output accumulation:
  - token-major unique ownership (no atomics) OR overlap (atomics/reduction kernel)
- Accumulation dtype:
  - FP16/BF16 accumulate? FP32 accumulate? per-stage?

### E) Quantization / scaling formats
- Weight dtype: FP16/BF16/FP8/int8/etc
- Activation dtype:
- For FP8:
  - per-tensor vs block quant (e.g., 128x128)
  - scale tensor shapes + layout (row-major? per-block?)
  - where activation quantization happens (pre-W1 / post-act / pre-W2)

## Output template (paste into `{artifact_dir}/constraints.md`)
```markdown
## MoE Parameters (verified from vLLM)
- Model file: `...`
- MoE class: `...`
- Router function: `...`
- scoring_func: ...
- norm_topk_prob: ...
- routed_scaling_factor: ...
- tie-break: ...
- shared experts: ...
- weight timing: ...
- accumulation: ...
- quantization: ...
- TP/EP/E_local: ...
