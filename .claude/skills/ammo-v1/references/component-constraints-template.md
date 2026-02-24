# Component Constraints Template

Use this template to document the target component's constraints in `{artifact_dir}/constraints.md`.

## A: Target Envelope

- Model: {model_id}
- Hardware: {hardware}
- Dtype / quant format: {dtype}
- TP / PP: tp={tp}
- Max model len: {max_model_len}
- Decode buckets (batch sizes): {batch_sizes}
- E2E workload: input_len={input_len}, output_len={output_len}

## B: Component Identification

- Target component: (e.g., MoE layer, attention, sampling, KV cache, FFN)
- vLLM code path: (e.g., `vllm/model_executor/layers/fused_moe/fused_moe.py`)
- Production kernel name(s) from nsys: (e.g., `fused_moe`, `flash_attn_with_kvcache`)
- Kernel patterns for verification: (regex patterns to match in nsys output)

## C: Forward Path

Document the data flow through the target component:

1. **Inputs**: shape, dtype, source
2. **Intermediate allocations**: shape, dtype, purpose
3. **Control flow**: branching, dispatch conditions
4. **Outputs**: shape, dtype, destination

Example for MoE:
```
Input: x[BS, K] bf16 -> router_logits[BS, E] -> top_k selection
-> expert dispatch -> W1 GEMM -> activation -> W2 GEMM -> output[BS, K] bf16
```

Example for Attention:
```
Input: Q[BS, H, D], K/V from paged cache
-> QK^T -> softmax -> attention weights -> V multiply -> output[BS, H, D]
```

## D: Correctness Invariants

- **Numerical tolerance**: atol, rtol requirements (from existing vLLM tests or dtype defaults)
- **Shape constraints**: must maintain exact input/output shapes
- **Determinism**: is deterministic output required?
- **Edge cases**: (e.g., empty batches, single-token, max sequence length)
- **Semantic invariants**: (e.g., softmax must sum to 1, attention weights non-negative)

## E: Baseline Truth Snapshot

**ACTUAL NUMBERS required** -- not commands.

- Production settings: CUDA graphs = {on/off}, torch.compile = {level}, V1 = {yes/no}
- Profile source: {artifact_dir}/runs/baseline.nsys-rep

| Batch Size | Target Kernel Time (us) | E2E Latency (s) | Component Share f |
|-----------|------------------------|-----------------|------------------|
| 8 | XXX | XXX | XXX |
| 64 | XXX | XXX | XXX |

## F: "Already Optimized?" Checklist

Before proposing optimizations, check what the baseline already does:

- [ ] Is torch.compile already fusing adjacent kernels for this component?
- [ ] Are CUDA graphs already capturing this component's kernel sequence?
- [ ] Does vLLM already have a specialized fast-path for this configuration?
- [ ] Is there an existing Triton/CUTLASS kernel that's already near-optimal?
- [ ] What kernels appear between the target component's main operations? (potential fusion targets)
