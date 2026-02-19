# Route Selection Decision Tree (Generalizable)

This is the quickest way to choose **full cooperative monokernel** vs **hybrid large‑grid fusion** vs **split kernels** across vLLM MoE models.

## Contents
- Baseline Truth Snapshot (required)
- Phase 2 Route Decision template (required)
- Kill criteria examples
- Decision Tree (A/B/C)

## Phase 1 required artifact: Baseline Truth Snapshot

Write this in `{artifact_dir}/constraints.md` (copy/paste and fill).

### Baseline normalization (do before route selection if applicable)

If the baseline warns it is using a **default** config due to a missing tuned file (common for Triton MoE kernels):
- Generate a tuned baseline config (bounded tuner is acceptable).
- Re-run the baseline snapshot under CUDA graphs.
- Use the *normalized* baseline for route selection. Otherwise you may optimize the wrong target.

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

### Fusion feasibility heuristics (required when proposing W1-epilogue or full monokernel)

If your Phase 2 plan proposes:
- W1-epilogue fusion (W1→activation→W2-input-quant), or
- full monokernel (…→W2→reduce)

Then compute and record:
- `P = M * top_k` (token–expert pairs on this rank for the bucket)
- `bytes_saved` and `time_saved_max_us` using `references/fusion-feasibility-heuristics.md`
- Compare `time_saved_max_us` to the **required µs savings** to beat baseline (from Phase 2 Baseline delta gate)

Conclusion (required): {plausible / low-probability} and why.

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

**If you can’t name the kernels** (e.g., you only have a single wrapper timing like `fused_experts`): treat this section as incomplete and run Nsight Systems to list the kernel nodes inside that wrapper before claiming “no fusion opportunities”.

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

### Hybrid deliverable (required when route == hybrid large‑grid fusion)

Write one of:
- Fusion target(s) for Phase 3: {e.g., W1 epilogue fusion; routing+prepare fusion}, or
- “No fusion opportunities” proof: {nsys per-kernel breakdown + justification why only tuning/config improvements are pursued}.

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

## Route Feasibility Triage (REQUIRED in Phase 2)
Goal: before committing to a route, compute an objective feasibility bound for ALL THREE routes:
(A) cooperative monokernel, (B) hybrid large-grid fusion, (C) split kernels + CUDA graphs.

This triage must be completed using:
- Baseline Truth Snapshot (per-stage kernel breakdown under production parity CUDA graphs)
- One NCU report for the dominant GEMM kernel(s) (grid size, achieved occupancy, regs, SMEM, DRAM/L2 bytes)
- Routing distribution summary (uniform M_avg + measured histogram if available)

### Common inputs (fill once per representative bucket)
For each representative bucket (at least: tiny decode, medium, large/prefill if applicable):
- M = tokens in bucket (decode batch size for this step)
- top_k
- E_local (post TP/EP)
- K (hidden)
- N_local (ffn intermediate per rank)
- P = M * top_k   # token–expert pairs

From baseline nsys (CUDA graphs):
- T_total_us
- T_route_us
- T_prepare_us
- T_w1_us
- T_act_us
- T_quant_us
- T_w2_us
- T_reduce_us

From baseline NCU (dominant GEMM kernel):
- gemm_grid_ctas
- achieved_occupancy
- regs_per_thread
- smem_static_bytes, smem_dynamic_bytes
- dram_bytes_read+write (sum)
- l2_bytes (sum)

Compute:
- share_gemm = (T_w1_us + T_w2_us) / T_total_us
- share_non_gemm = 1 - share_gemm
- non_gemm_us = T_total_us - (T_w1_us + T_w2_us)
- Headroom_non_gemm_us = T_route_us + T_prepare_us + T_act_us + T_quant_us + T_reduce_us
  (Upper bound on savings if GEMMs do not slow down.)

Optional constants for byte bounds (override if your baseline differs):
- bytes_act = 2   # bf16/fp16 intermediates
- bytes_fp8 = 1
- bytes_scale = 4 # per-pair scale (if per-token/per-pair scale differs, state it)

---

## A) Cooperative monokernel feasibility scorecard
Use this scorecard IN ADDITION to the existing cooperative viability test.

### A1. Hard gates (fail-fast)
[ ] Barrier budget: requires <= 2 grid.sync points total.
    - If >2 => FAIL cooperative (default to hybrid/split).
[ ] Output ownership: can write output without atomics (token-major K-slice or equivalent).
    - If requires FP32 atomicAdd into [M, K] at top_k>1 => FAIL cooperative unless M is huge and atomics are proven acceptable.
[ ] Cooperative launch feasible under resources:
    - If dynamic SMEM forces 1 CTA/SM AND baseline relies on large-grid latency hiding => HIGH RISK (usually FAIL).

### A2. Concurrency penalty test
Compute:
- coop_grid_ctas ≈ O(SM_count)  (cooperative caps you near SM count)
- R = gemm_grid_ctas / coop_grid_ctas

Rule:
- If R >= 8 and share_gemm is high => cooperative must reduce per-output cost by a LARGE margin, not “fewer kernels”.

### A3. Memory-hop upper bound (bytes/BW)
This is an *upper bound* on time saved by removing intermediate store+load hops.

Bytes saved if you fuse W1 + activation + quant (no BF16 intermediates mm1_out/act_out):
- bytes_saved_w1ep =
    2 * (P * (2*N_local) * bytes_act)   # write+read mm1_out
  + 2 * (P * (N_local)   * bytes_act)   # write+read act_out
  = 6 * P * N_local * bytes_act

Extra bytes saved if you ALSO fuse away FP8 quant_out into W2:
- bytes_saved_w2hop =
    2 * (P * N_local * bytes_fp8)       # write+read quant_out
  + 2 * (P * bytes_scale)              # write+read per-pair scale (if applicable)

Total bytes saved upper bound:
- bytes_saved_full = bytes_saved_w1ep + bytes_saved_w2hop

Convert to time bound:
- BW_eff = (dram_bytes_read+write) / (T_w1_us + T_w2_us)   # from NCU baseline GEMM kernel
- t_saved_max_us ≈ bytes_saved_full / BW_eff

Interpretation:
- If t_saved_max_us is single-digit µs and baseline GEMM grid is large, cooperative is very unlikely to beat baseline under CUDA graphs.

### A4. Decide PASS/FAIL (per bucket)
PASS cooperative only if ALL are true:
- Hard gates pass
- R is not huge OR headroom_non_gemm_us is large (>= ~15–20% of T_total_us)
- Routing distribution is not extremely skewed (skew amplifies barrier tax)

If PASS on at least one target bucket, cooperative remains a candidate; otherwise prefer hybrid/split.

---

## B) Hybrid large-grid fusion feasibility scorecard
Hybrid is chosen when baseline GEMMs dominate and use large grids (grid >> SM count). The key question is:
“Is there at least ONE material fusion target around GEMM that can save meaningful GPU time WITHOUT slowing GEMMs?”

### B1. Entry gates
[ ] share_gemm >= ~70% AND baseline GEMM grid is large (grid >> SM count).
    - If false, hybrid may not be the best default; consider split kernels.

### B2. Candidate fusion inventory (must list at least 2 or prove none)
Candidate types (pick those applicable):
- (H1) routing/prepare fusion or routing algorithm improvement
- (H2) W1 epilogue fusion (activation + W2-input quant inside W1 kernel)
- (H3) fuse activation+quant as ONE kernel (separate from W1 GEMM; preserves GEMM)
- (H4) reduce fusion (usually NO unless you accept atomics/remap)

### B3. Headroom test per candidate (stage-time bound)
For each candidate compute the *max removable time* under CUDA graphs:

- H1 routing/prepare: headroom_us = T_route_us + T_prepare_us
- H2 W1-epilogue-in-GEMM: headroom_us = T_act_us + T_quant_us
- H3 fuse act+quant kernel: headroom_us = min(T_act_us + T_quant_us, non_gemm_us)
- H4 reduce: headroom_us = T_reduce_us

Rules:
- If headroom_us < 5% of T_total_us on the buckets you care about, it’s probably not worth risking GEMM regression.
- Under CUDA graphs, “kernel count reduction” alone is rarely enough; demand real GPU-time savings.

### B4. Risk test: “will this perturb the GEMM?”
- H2 (epilogue inside GEMM) is HIGH RISK when:
  - baseline GEMM already has strong occupancy/utilization, and
  - headroom_us is small (fusion must be nearly free), and/or
  - the epilogue requires reductions (e.g., per-token FP8 quant amax/scale) that inflate regs/SMEM.

Default rule:
- If headroom_us is small => prefer H3 (fuse act+quant as separate kernel) over H2.

### B5. Decide PASS/FAIL (per bucket)
Hybrid is the best route when:
- Entry gates pass, AND
- At least one candidate has headroom_us >= required savings target, AND
- The candidate does not require cooperative barriers and does not obviously tank GEMM occupancy.

If hybrid passes only with “tuning config files” and you cannot identify a fusion candidate, you must provide a “no fusion opportunities” proof (nsys kernel breakdown shows already-fused/negligible stages).

---

## C) Split kernels + CUDA graphs feasibility scorecard
Split-kernel route means 2–3 graph-captured kernels that preserve GEMM performance while improving the “front end” and/or removing a material intermediate hop.

Typical split graph:
- Router/Prepare  →  W1 GEMM  →  (Act+Quant fused)  →  W2 GEMM  →  Reduce (if needed)

### C1. When split is a serious candidate (entry conditions)
Split becomes attractive when ANY are true:
- Cooperative fails the viability test (barriers, SMEM/regs, concurrency).
- Hybrid’s best candidate is H2 (epilogue-in-GEMM) but risk is high and headroom is small.
- Ownership is token-major/hybrid and you can avoid expensive grid.sync and atomics.

### C2. Underfill / utilization heuristic (token distribution driven)
Use these to judge whether changing ownership or staging can help:
- M_avg = (M * top_k) / E_local
  - If M_avg < 8: expert-major grouped GEMM often underutilized; token-major or hybrid ownership may win.
  - If 8–16: benchmark; hybrid ownership (expert-major up, token-major down) often wins.
  - If >= 16: expert-major can be viable.

Also look at baseline GEMM grid:
- If gemm_grid_ctas is not >> SM count, you are underfilling; split/token-major strategies can create more CTAs.

### C3. Split fusion headroom (what can you actually remove?)
Core “safe” split fusion target is H3:
- Fuse activation + quant into ONE kernel (do NOT touch GEMM).
This removes:
- one kernel launch node,
- and the act_out intermediate store+load hop.

Bytes saved bound (act_out hop only):
- bytes_saved_actquant = 2 * (P * N_local * bytes_act)

Rule of thumb:
- If (T_act_us + T_quant_us) is tiny, this fusion won’t matter under CUDA graphs.
- If (T_act_us + T_quant_us) is material, H3 is a low-risk win compared to H2.

### C4. Decide PASS/FAIL (per bucket)
PASS split if:
- Cooperative is high risk / fails gates, AND
- There exists at least one split-stage improvement with headroom_us >= required savings target,
  while keeping baseline GEMMs intact.

If split is chosen, explicitly document:
- which kernel boundaries are preserved (to avoid GEMM regression),
- what intermediate(s) you are eliminating,
- and what CUDA-graph capture constraints apply.

---

## How to choose a route after triage
For each target bucket, record:
- PASS/FAIL for A/B/C
- best candidate headroom_us for each PASS route
- key risk flags (barriers, occupancy collapse, atomics)

Pick the route that:
1) PASSes on the most important bucket(s), and
2) has the highest headroom_us with the lowest risk flags.

If two routes tie, choose the lower-risk one (hybrid/split) unless cooperative is clearly justified by large headroom and low barrier/occupancy risk.
