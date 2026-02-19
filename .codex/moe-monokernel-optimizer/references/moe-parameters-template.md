# MoE Parameters & Baseline Truth Template (Phase 1)

Use this to write `{artifact_dir}/constraints.md`. Fill it by reading the model’s vLLM implementation (`vllm/model_executor/models/...`) and by profiling **production parity** (CUDA graphs / torch.compile / real TP/EP and buckets).

## Contents
- A) Model + environment
- B) MoE topology + routing semantics (do not guess)
- C) Expert compute + epilogue semantics
- D) Distributed semantics (TP/EP, reductions)
- E) Quantization / scaling formats
- F) Baseline Truth Snapshot (production parity, required for route selection)
- Output template (copy/paste)

## Search anchors
scoring_func, norm_topk_prob, routed_scaling_factor, shared_experts, tie-break, TP, EP, E_local, E_global, fused_moe, fused_experts, fp8 scales, per-block quant

---

## A) Model + environment
- Model ID:
- vLLM commit hash / version:
- GPU(s) + arch (sm_80/sm_89/sm_90a):
- CUDA / driver:
- Dtype path: BF16/FP16/FP8/int8/etc

## B) MoE topology + routing semantics (do not guess)
- Model file: `vllm/model_executor/models/...`
- MoE module / class:
- Router function name + file:
- `E_global`:
- `E_local` (per rank):
- `top_k`:
- Routing score function: softmax / sigmoid / other
- `norm_topk_prob`: true/false (or equivalent)
- Any extra router scaling (temperature, bias, z-loss, etc.):
- Tie-break / stable ordering rules (if any):
- Shared experts? (yes/no; how applied):
- Routing weight placement: pre-activation / post-activation (or both)
- Activation function: SiLU / SwiGLU / GeGLU / other

## C) Expert compute + epilogue semantics
- Expert weights layout (per expert): shapes for W1/W2 (or up/down)
- Gating: W1 output is 2N? (for SwiGLU) or N?
- Output accumulation rule (sum over top_k, scaling order):
- Any per-expert/per-token scaling after W2:

## D) Distributed semantics (TP/EP, reductions)
- TP size:
- EP size:
- Is expert parallel (EP) pre-dispatch done outside the kernel? (yes/no)
- Where reduction happens: inside fused kernel / separate kernel / NCCL / other
- Any cross-rank all-reduce in the MoE path:

## E) Quantization / scaling formats
- Weight dtype per matrix (W1/W2):
- Activation dtype in/out:
- If FP8:
  - Per-tensor vs block quant (e.g., 128×128)
  - Scale tensor shapes + layout (per-row/per-col/per-block)
  - Where activation quantization happens (pre-W1 / post-act / pre-W2)
  - Are per-token scales materialized to global? (yes/no; shape)

---

## F) Baseline Truth Snapshot (production parity)

This section is the **input to route selection**. Collect it under the *same serving mode* you will ship (CUDA graphs, torch.compile, TP/EP, bucketed shapes). Use `references/profiling-launch-vs-kernel.md` if launch time vs kernel time is confusing.

### F0) Full-model E2E baseline (required)

You must record at least one **full-model** E2E latency baseline in Phase 1 (to compute MoE share `f = T_moe / T_total` and sanity-check that MoE is worth optimizing).

- If the model is not cached locally: **download the weights** (do not skip E2E because the cache is empty).
- If you cannot download due to gating/auth/terms, or due to network/disk constraints:
  - mark the target `blocked` in `{artifact_dir}/state.json`,
  - document the blocker in `{artifact_dir}/constraints.md`,
  - and ask the user explicitly whether they want to waive the E2E requirement.

Record:
- the exact `vllm bench latency` command(s),
- the bucket(s) used (BS, input_len, output_len),
- the serving parity knobs (CUDA graphs / torch.compile / TP/EP),
- the resulting E2E latency numbers.

### F1) Buckets + modes
- Buckets measured (must match production): `BS = {...}` (and sequence length regime)
- CUDA graphs mode: on/off + capture details (static shapes/buckets?)
- torch.compile mode: on/off + backend (if relevant)
- Evidence files: record trace paths + exact commands (prefer `nsys` with `--cuda-graph-trace=node`; see `references/nsys-profiling-guide.md`).

### F2) Kernel-time breakdown (GPU time)
For 1–2 representative buckets (and then for all “in-envelope” buckets later), record:
- Total MoE GPU time per step (µs):
- Top 5 MoE kernels by GPU time (name + µs + % of MoE):
- Stage breakdown if you can map it:
  - routing / prepare / W1 / activation / quant / W2 / reduce

### F3) Baseline “already fused?” checklist
Answer per your baseline build/config (yes/no):
- Routing is fused CUDA op (vs Python fallback)
- Routed-weight multiply folded into expert GEMM epilogue
- Activation fused into W1 epilogue
- W2-input quant fused into W1 epilogue (single write to FP8)
- Reduce (`moe_sum` / top_k accumulate) fused into W2 (or fused elsewhere)
- Any explicit sort/prefix-sum kernels present (token↔expert transforms)

### F4) Dominant GEMM concurrency facts (sanity)
For the dominant GEMM-like kernel(s), note:
- Approx blocks launched / waves (is the GPU already saturated?)
- Occupancy / regs / spills (coarse)
- Any obvious barrier/SMEM bottlenecks (if known)

### F5) Derived triage numbers (per bucket, per rank)
- `P = BS * top_k` (token–expert pairs on this rank)
- `M_avg ≈ BS * top_k / E_local` (uniform routing proxy; use real routing stats if you have them)
- If your serving buckets vary sequence lengths, note if the MoE path changes (prefill vs decode).

---

## Output template (paste into `{artifact_dir}/constraints.md`)

```markdown
## A) Model + environment
- Model ID:
- vLLM commit hash / version:
- GPU(s) + arch:
- CUDA / driver:
- Dtype path:

## B) MoE topology + routing semantics (verified from vLLM)
- Model file:
- MoE module / class:
- Router function:
- E_global:
- E_local:
- top_k:
- Routing score function:
- norm_topk_prob:
- Any extra router scaling:
- Tie-break rules:
- Shared experts:
- Routing weight placement:
- Activation:

## C) Expert compute + epilogue semantics
- W1/W2 shapes + layouts:
- Gating (2N?):
- Output accumulation rule:
- Any post-W2 scaling:

## D) Distributed semantics (TP/EP, reductions)
- TP size:
- EP size:
- EP pre-dispatch outside kernel:
- Where reduction happens:
- Any cross-rank all-reduce:

## E) Quantization / scaling formats
- Weight dtype(s):
- Activation dtype in/out:
- FP8 details (if applicable):
  - Quant granularity:
  - Scale shapes/layout:
  - Quant placement:
  - Per-token scales materialized:

## F) Baseline Truth Snapshot (production parity)
### Full-model E2E baseline (required)
- E2E command(s):
- Buckets and workload:
- Parity knobs (CUDA graphs / torch.compile / TP/EP):
- E2E results (avg/p50/p90 etc):

### Buckets + modes
- Buckets measured:
- CUDA graphs mode:
- torch.compile mode:

### Kernel-time breakdown (GPU time)
- Representative bucket 1: BS=...
  - Total MoE GPU time (µs):
  - Top kernels:
  - Stage breakdown (if known):
- Representative bucket 2: BS=...
  - ...

### Baseline “already fused?” checklist
- Routing fused CUDA op:
- Routed-weight multiply in GEMM epilogue:
- Activation fused into W1:
- W2-input quant fused into W1:
- Reduce fused:
- Sort/prefix-sum present:

### Dominant GEMM concurrency facts
- Blocks / waves:
- Occupancy/reg/spills:
- Notes:

### Derived triage numbers
- P (BS*top_k) per bucket:
- M_avg (uniform proxy) per bucket:
```
