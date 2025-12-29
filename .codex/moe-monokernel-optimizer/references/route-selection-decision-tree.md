# Route Selection Decision Tree (Generalizable)

This is the quickest way to choose **full cooperative monokernel** vs **hybrid large‑grid fusion** vs **split kernels** across vLLM MoE models.

## Phase 1 required artifact: Baseline Truth Snapshot

Write this in `{artifact_dir}/constraints.md` (copy/paste and fill).

### Deployment + geometry

- Model:
- Hardware:
- Dtype / quant format:
- TP / EP:
- `E_global`:
- `E_local`:
- `top_k`:
- Hidden dims: `K` (in), `N` (ffn):
- Activation:
- Routing weight placement (pre/post activation):

### Batch buckets (decode)

- Buckets measured (must match production): `BS = { ... }`
- CUDA graphs mode:
- torch.compile mode:

### Routing distribution

- Uniform proxy: `M_avg = BS * top_k / E_local` (per bucket)
- Measured histogram (if available): p50/p95 per‑expert tokens

### Baseline “already fused?” checklist (per your config)

- Routing is fused CUDA op (vs Python fallback): yes/no
- Routed‑weight multiply folded into expert GEMM epilogue: yes/no
- Activation kernel separate from W1: yes/no
- W2 input quantization separate from W1: yes/no
- Reduce (`moe_sum`) separate from W2: yes/no

### CUDA‑graphs baseline kernel-time breakdown (GPU time)

For a representative bucket (and optionally 2–3 buckets):

| Stage | Kernel(s) | GPU time (µs) | Share |
|------:|-----------|---------------|-------|
| routing | … | … | … |
| prepare/align/sort | … | … | … |
| W1 GEMM | … | … | … |
| activation | … | … | … |
| quantize (W2 input) | … | … | … |
| W2 GEMM | … | … | … |
| reduce | … | … | … |
| **total** |  | … | 100% |

Derived:
- `share_gemm = (W1 + W2) / total`
- `share_non_gemm = 1 - share_gemm`

### Baseline concurrency facts (dominant GEMM kernel)

From NCU (or a comparable profiler view):
- Grid size (CTAs launched):
- Achieved occupancy:
- Regs / thread:
- Static + dynamic SMEM / CTA:
- Top stall reasons (1–3 bullets):

### MoE share of end-to-end (stop-condition input)

- `T_total` (end‑to‑end) and `T_moe` (MoE subgraph) under production settings
- `f = T_moe / T_total`
- Use `references/e2e-delta-math.md` to bound expected system impact.

## Phase 2 required artifact: Route Decision

Write this in `{artifact_dir}/optimization_plan.md` (copy/paste and fill).

### Route (pick one)

- Chosen route: **cooperative monokernel** / **hybrid large‑grid fusion** / **split kernels**

### Why this route (tie to Baseline Truth Snapshot)

- 3–6 bullets referencing the snapshot numbers (shares, dominant kernels, concurrency facts).

### Why not the other routes

- Cooperative monokernel: …
- Hybrid large‑grid fusion: …
- Split kernels: …

### Kill criteria (pivot trigger)

Write at least one “stop if…” statement measurable via NCU/nsys, e.g.:
- “Stop cooperative work if dynamic SMEM forces 1 CTA/SM and baseline GEMM already achieves high occupancy/concurrency.”
- “Stop routing optimization if routing+prepare is <X% under CUDA graphs.”

## Decision Tree (A/B/C)

This maps the snapshot to a route.

### A) Baseline GEMMs dominate and baseline concurrency is strong → **Hybrid large‑grid fusion**

Choose **hybrid** when both are true:
- `share_gemm` is high (typ. ≥ ~70%)
- Dominant GEMM kernel uses **large grids** (grid >> SM count) and has decent achieved occupancy/utilization

Reason:
- A cooperative design caps parallelism near SM count and often adds `grid.sync` + higher SMEM/regs.
- Even if it reduces some memory traffic, it can lose on latency hiding/concurrency.

Primary opportunities:
1. Fuse **W1 epilogue**: activation + W2‑input quant into W1 GEMM (delete large intermediate writes/reads).
2. Fuse/optimize routing+prepare only if it is material under CUDA graphs.
3. Defer W2+reduce fusion unless you accept atomics or a major remap.

### B) Non‑GEMM stages are a big slice → **Hybrid** or **Split kernels**

Choose **hybrid** when the non‑GEMM slice is driven by large materializations:
- activation/quant kernels move big tensors (e.g., `[BS*top_k, N]`)

Choose **split kernels** when:
- ownership is token‑major/hybrid and you can avoid expensive global barriers
- the “front end” (routing/prepare) is large and can be improved independently

### C) Cooperative monokernel viable → **Cooperative**, but only after the viability test

Consider cooperative only if most are true:
- `share_non_gemm` is large enough that fusing can plausibly win
- You can keep `grid.sync` ≤ 1–2
- Your SMEM+regs budget does not collapse occupancy below what’s needed (or the per‑CTA work is enormous)
- Routing distribution is not extremely skewed (imbalance amplifies barrier tax)

#### Cooperative viability test (do before committing to the design)

1. **Barrier tax**: if >2 grid‑wide sync points are required → default away from cooperative.
2. **SMEM/reg budget**: estimate per‑CTA SMEM and expected regs; if this forces 1 CTA/SM and the baseline relies on latency hiding → assume loss.
3. **Concurrency comparison**:
   - If baseline grid is thousands of CTAs and cooperative would be ~O(SM count), cooperative must win by large per‑output cost reduction (not “fewer kernels”).

