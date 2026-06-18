# Optimization Category Taxonomy

Canonical reference for AMMO's optimization-category catalog — a **non-binding, non-exhaustive descriptor**, NOT the eligibility gate. The gate is the principle in `SKILL.md` NN#8 (authored mechanism logic/structure + profiled bottleneck + measured parity-safe E2E win). Each Phase 0 proposal MUST still self-classify via the `## Category` block (spec §5.3): the category determines projection formula, Phase 0 evidence requirements, and validation-gate routing. But **catalog membership never decides eligibility** — a novel mechanism that passes the gate yet fits no listed class is eligible; it names a new descriptor and maps onto the projection slice it attacks.

This file is referenced from:
- `SKILL.md` — NN#7 (`f_e2e` as Amdahl input) and NN#8 (authored-mechanism mandate)
- `references/e2e-delta-math.md` — projection formulas, four-slice composition model
- `references/technology-selection.md` — eligible technologies per category
- `references/validation-defaults.md` — default Gate 5.2 form
- `references/debate-scoring-rubric.md` — per-category projection-formula scoring penalty
- `agents/ammo-champion.md` — Category field requirement in Phase 0 template
- `agents/ammo-impl-champion.md` — per-category validation gate routing + scaffolds (the impl-champion runs the kernel correctness & speedup checks)
- `orchestration/debate-protocol.md` — Category eligibility gate, champion spawn context

---

## Schema-Version Guard (Legacy Campaigns)

The Category requirement only applies to campaigns with `state.json.campaign.schema_version >= "4.1"`. Legacy campaigns (`schema_version < "4.1"` or absent) use the pre-existing Phase 0 eligibility rules (Authored-Mechanism Mandate + Technology Selection + Precision Classification only). The lead reads `state.json` and skips Category-related gates when the schema version is older than 4.1, preserving the resume path for paused/in-flight campaigns.

This mirrors the §3.4 / §5.3 schema-version pattern used by `verify_stage2_gate.py`, `ammo-validate-researcher-dilution.sh`, and the new_target.py default bump.

---

## When the Category Is Chosen (Phase 0)

The category is a conclusion drawn from data — whether a fusable seam exists, whether a kernel can be replaced by a faster one, or whether host-side dispatch gaps dominate. These become clear from the existing profiling data for the component. The category is selected in **Phase 0**, after the champion analyzes the existing profiling data for its assigned component; the Phase -1 target claim is **component-only**. See `agents/ammo-champion.md` § Target Claim Phase and `orchestration/debate-protocol.md` § Phase -1.

This shapes exhaustion filtering: the claim-time waterfall keys on the component, so it rejects a *fully-exhausted component* (every category exhausted). The precise per-technology exhausted check runs as a soft check at the **Phase 0 Diversity Check (Lead)** in `orchestration/debate-protocol.md` (item 2), once the proposal declares a concrete `## Category` + technology. This matches the `exhausted_technologies[]` schema, which keys on `technology_class` + `applies_to_component` + `applies_to_shape_bucket`.

## The Category Catalog (non-binding, non-exhaustive descriptor)

Eligibility is decided by the NN#8 principle, not by this table. The catalog exists to (a) select the projection formula and (b) route validation. Each class is grounded in verified vLLM in-tree precedent — the mechanism is real engineering vLLM itself ships, not a novel risk. The "Slice targeted" column is what binds a class to its projection formula and gate routing (see § Validation Routing — by slice, not by enum).

**Decode-kernel-slice classes** (route to Standard / chain Gate 5.2):

| Category | Scope / what's authored | Slice targeted | Gate set |
|----------|-------------------------|----------------|----------|
| `kernel_replacement` | Single kernel → faster alternative | `f_e2e(kernel)` | 5.1a, 5.1b, **5.2**, 5.3a, 5.3b |
| `kernel_fusion` | N kernels → 1 fused kernel | `f_e2e(chain)` | 5.1a, 5.1b, **5.2**, 5.3a, 5.3b |
| `custom_kernel` | New/rewritten kernel compute (Triton/CuTeDSL/CUTLASS/CUDA C++) — legacy alias of `kernel_replacement` | `f_e2e(kernel)` | 5.1a, 5.1b, **5.2**, 5.3a, 5.3b |
| `weight_layout_transform` | Load-time weight concat/repack/requant + slice, library does the matmul (e.g. QKV/gate-up weight-merge) | `f_e2e(chain)` | 5.1a, 5.1b, **5.2**, 5.3a, 5.3b |
| `attention_kv_layout` | Authored KV-cache layout / new AttentionBackend / metadata builder | `f_e2e(kernel)` | 5.1a, 5.1b, **5.2**, 5.3a, 5.3b |

**Inter-kernel-slice classes** (inherit the `dispatch_optimization` Gate 5.2 SKIPPED carve-out):

| Category | Scope / what's authored | Slice targeted | Gate set |
|----------|-------------------------|----------------|----------|
| `dispatch_optimization` | CPU↔GPU pipelining, dispatch elimination | host portion of `inter_kernel_share` | 5.1b, 5.3a, 5.3b |
| `execution_pipeline_restructuring` | Authored scheduling / H2D-D2H overlap / async-stream / CUDA-graph capture-boundary code — legacy alias of `dispatch_optimization` | host portion of `inter_kernel_share` | 5.1b, 5.3a, 5.3b |
| `communication_strategy` | Authored collective-comm algorithm / EP dispatch / EPLB / comm-compute overlap | host portion of `inter_kernel_share` | 5.1b, 5.3a, 5.3b |
| `compute_graph_pass` | Authored Inductor/FX pattern-matcher pass that rewrites the compiled graph | host portion of `inter_kernel_share` | 5.1b, 5.3a, 5.3b |

**Legacy aliases (retained, never deleted):** `kernel_replacement` conceptually folds into `custom_kernel`; `dispatch_optimization` conceptually folds into `execution_pipeline_restructuring`. Both old values **remain valid and route to their existing gate set** so paused campaigns resume cleanly (see `SKILL.md` NN#8 and the schema's add-only enum). **Floor-reject twins** (stay out — retuned constant, not authored mechanism): selecting an existing kernel via a backend flag; toggling `--enforce-eager` / a `custom_ops` list; `--max-num-batched-tokens` / `--cudagraph-capture-sizes`; `--tensor-parallel-size` / `--all-reduce-backend`; `--kv-cache-dtype` / `--block-size`.

---

## Per-Category Projection Formulas (spec §3.6)

Champions MUST use the formula matching their declared category. Wrong formula → **0/10 for E2E impact** in the debate scoring rubric (regardless of how good the underlying evidence is).

| Category | Projection formula | Variables |
|----------|-------------------|-----------|
| `kernel_replacement` | `f_e2e(kernel) × (1 - 1/s)` | `s` = kernel speedup ratio (new vs baseline at target shape, under CUDA graphs + torch.compile) |
| `kernel_fusion` | `f_e2e(chain) × (1 - 1/s_fused)` | `f_e2e(chain)` = sum of `f_e2e` over the fused chain. `s_fused` = `chain_time / fused_time` |
| `dispatch_optimization` | `inter_kernel_share × host_fraction × elimination_fraction` | `host_fraction` = fraction of `inter_kernel_share` attributable to CPU-side dispatch/scheduling (measured by comparing nsys CPU-activity vs GPU-idle regions). `elimination_fraction` = measured reduction in host-side latency (dispatches removed / total dispatches, or component wall-time reduction ratio). |
| `custom_kernel` | `f_e2e(kernel) × (1 - 1/s)` | Same as `kernel_replacement` (its alias). |
| `weight_layout_transform` | `f_e2e(chain) × (1 - 1/s_fused)` | Same form as `kernel_fusion`: `f_e2e(chain)` = sum of `f_e2e` over the merged GEMMs; `s_fused` = `chain_time / merged_time`. |
| `attention_kv_layout` | `f_e2e(kernel) × (1 - 1/s)` | Same form as `kernel_replacement`: `s` = attention-kernel speedup at target shape. |
| `execution_pipeline_restructuring` | `inter_kernel_share × host_fraction × elimination_fraction` | Same as `dispatch_optimization` (its alias). |
| `communication_strategy` | `inter_kernel_share × host_fraction × elimination_fraction` | Same form: `elimination_fraction` = measured reduction in collective-comm / EP-dispatch host latency. |
| `compute_graph_pass` | `inter_kernel_share × host_fraction × elimination_fraction` | Same form: `elimination_fraction` = measured reduction in dispatched-op count / host wall time from the graph rewrite. |

**Nearest-analogue fallback (novel descriptor):** a mechanism that fits no listed class uses the formula for **whichever of the two regimes its declared Slice targeted matches** — `f_e2e(slice) × (1 - 1/s)` if it attacks decode-kernel compute, or `inter_kernel_share × host_fraction × elimination_fraction` if it attacks the inter-kernel/host slice. Gate 4 matches the proposal's formula to the slice it claims, not to a fixed enum row.

**`f_e2e` definition** (canonical, see `e2e-delta-math.md`): `f_e2e = f_decode × decode_busy × decode_share_of_e2e`. The conversion is **mandatory** whenever any workload-dilution red flag fires (`decode_busy < 0.85`, `prefill_share_of_e2e ≥ 0.10`, or `input_len ≥ 512`).

---

## Per-Category Phase 0 Evidence Requirements (spec §5.2)

Every Phase 0 proposal MUST attach an empirical micro-experiment. The required form depends on the **slice** the declared category targets (not on the specific enum value):

### decode-kernel-slice categories (`kernel_replacement`, `kernel_fusion`, `custom_kernel`, `weight_layout_transform`, `attention_kv_layout`)

- **Phase 0 evidence**: New/merged kernel (or restructured kernel chain) beats production baseline at the target shape, **under CUDA graphs + torch.compile**, with cold + warm cache timings. Roofline / PyTorch reference / proxy kernels are insufficient (see anti-regression rule). For `weight_layout_transform`, the baseline is the **separate** GEMMs and the optimized side is the single merged GEMM — measure both under parity.
- **Tier**: 2+ (per `references/debate-rules.md` Evidence Tiers)
- **Theoretical-only cap**: 3/10 if no empirical evidence.

### inter-kernel-slice categories (`dispatch_optimization`, `execution_pipeline_restructuring`, `communication_strategy`, `compute_graph_pass`)

- **Phase 0 evidence**: Demonstrate on the production model with a representative decode step:
  - The new dispatch / scheduling / comm / graph-rewrite path is exercised on the workload's distribution of inputs (not a mock)
  - The chosen path is faster than the previous default (measured under production parity)
  - Quantify: number of dispatches eliminated, host wall-time reduction, or graph-op-count reduction (nsys trace comparison)
- The micro-experiment exercises the actual production code path; mocks of the scheduler/preamble/collective are insufficient.
- **Gate 5.2**: SKIPPED by default. Runs only if a new kernel is introduced (e.g., a Triton kernel that replaces the dispatch sequence). When Gate 5.2 runs, the binding metric is component wall-time drop, not kernel-vs-kernel speedup ratio.
- **Anti-regression**: applies to the **E2E metric only** — there is no library-baseline ranking concept for dispatch/pipelining/comm/graph-pass code. Champions still owe a justification for technology choice (e.g., why Triton over Python for the fused dispatch).

---

## Per-Category Validation Gate Routing (spec §5.1)

The validation gates consult the proposal's declared category to select scaffolds. **Routing is keyed on the declared projection-slice, NOT on a per-category row** — the two slices below cover all catalog classes and any novel descriptor, so future descriptors need zero routing edits.

| Gate | decode-kernel-slice (`kernel_replacement`, `kernel_fusion`, `custom_kernel`, `weight_layout_transform`, `attention_kv_layout`) | inter-kernel-slice (`dispatch_optimization`, `execution_pipeline_restructuring`, `communication_strategy`, `compute_graph_pass`) |
|------|--------------------|------------------------|
| 5.1a (kernel correctness) | Standard (whole chain for fusion/merge) | Optional (only if a new kernel is introduced) |
| 5.1b (E2E correctness) | Sweep GSM8K | Sweep GSM8K |
| **5.2 (kernel speedup)** | Standard; chain-time vs fused-time for `kernel_fusion`/`weight_layout_transform` | **SKIPPED** by default; runs if a new kernel is introduced |
| 5.3a (kernel proof) | Standard nsys trace | **Trace inspection** — verify the new dispatch path executes and dispatches are eliminated |
| 5.3b (E2E latency) | Standard sweep | Standard sweep |

The **inter-kernel-slice Gate 5.2 carve-out** (originally the `dispatch_optimization` carve-out, now generalized to its slice): these categories' wins come from reducing inter-kernel gap time (host-side dispatch overhead), not from faster kernel compute. Forcing the standard kernel-vs-kernel speedup gate would erroneously fail valid optimizations. When a new kernel IS introduced (e.g., a Triton kernel replacing N dispatches), Gate 5.2 runs with component wall-time as the binding metric. The inter-kernel-slice routing is **byte-identical** to the legacy `dispatch_optimization` routing — a resumed paused `dispatch_optimization` track is never silently re-routed to the Standard Gate 5.2 path (which would force a spurious kernel-speedup FAIL).

---

## Technology Eligibility Per Category (spec §5.5)

Each category's eligible technologies are the set the impl-champion may use without justifying why they reached outside the canonical set. See `references/technology-selection.md` for selection signals and the anti-regression rule.

| Category | Eligible technologies |
|----------|---------------------|
| `kernel_replacement`, `kernel_fusion` | Triton, CuTeDSL, CUTLASS, CUDA C++ |
| `dispatch_optimization` | Python or C++ targeting vLLM internals (scheduler, preamble, executor paths); Triton if the optimization introduces a fused dispatch kernel |
| `custom_kernel`, `attention_kv_layout` | Triton, CuTeDSL, CUTLASS, CUDA C++ (kernel-authoring regime — same as `kernel_replacement`) |
| `weight_layout_transform` | Python `weight_loader` / load-time concat-repack-requant + the library GEMM it feeds; Triton/CUDA C++ only if a custom kernel is introduced for the merged op |
| `execution_pipeline_restructuring`, `communication_strategy`, `compute_graph_pass` | Python or C++ targeting vLLM internals (scheduler, executor, collective-comm, Inductor/FX passes); Triton if a fused kernel is introduced |

**Route-by-regime fallback (novel descriptor):** a class not listed uses the eligible-technology set of **whichever regime its declared Slice targeted matches** — the kernel-authoring set (Triton/CuTeDSL/CUTLASS/CUDA C++) for decode-kernel-slice mechanisms, or the vLLM-internals set (Python/C++, Triton if a kernel is introduced) for inter-kernel-slice mechanisms.

The standard rank-based anti-regression rule (Triton > CuTeDSL ≈ CUTLASS > CUDA C++) applies to the **kernel-authoring regime** (`kernel_replacement`, `kernel_fusion`, `custom_kernel`, `attention_kv_layout`, and the merged-kernel case of `weight_layout_transform`). For the **inter-kernel-slice regime** (`dispatch_optimization`, `execution_pipeline_restructuring`, `communication_strategy`, `compute_graph_pass`) and for structure-only `weight_layout_transform` (no custom kernel), anti-regression applies to the **E2E metric only** — there is no library-baseline ranking concept for dispatch/pipelining/comm/graph-pass/host-side-structure code.

---

## Disambiguating Examples

The taxonomy is meant to be self-classifying with low ambiguity.

### Example 1 — kernel_replacement (accepted)

> Replace the production DeepGEMM gate_up with a Triton kernel tuned for the target shape.

- **Category: `kernel_replacement`**. Single kernel → faster alternative.
- **Projection**: `f_e2e(kernel) × (1 - 1/s)`.
- **Gate 5.2**: standard kernel speedup vs production baseline.

### Example 2 — kernel_fusion (accepted)

> Replace the `silu_and_mul + dynamic_quant` two-kernel sequence with a single fused Triton kernel.

- **Category: `kernel_fusion`**. Two kernels → one. Stays within a single decode-step block.
- **Projection**: `f_e2e(silu+quant chain) × (1 - 1/s_fused)`.
- **Gate 5.2**: chain-time vs fused-time.

### Example 3 — dispatch_optimization (accepted)

> Fuse the MTP target-model preamble (28 eager kernel launches: torch.compile op + H2D copies + elementwise updates) into a single Triton `delta_advance` kernel that eliminates 27 dispatches and caches structurally-constant H2D uploads.

- **Category: `dispatch_optimization`**. The win is dispatch elimination — the 28 kernels collectively compute ~107 µs but incur ~6800 µs of inter-kernel gap time.
- **Projection**: `inter_kernel_share × host_fraction × elimination_fraction`.
- **Gate 5.2**: runs (new Triton kernel introduced), but binding metric is component wall-time drop (not kernel-vs-kernel speedup). The 9.96× kernel speedup is decorative; the 1.418× preamble wall-time reduction drives E2E.
- **Technology**: Triton (fused dispatch kernel) + Python (H2D caching, sync deferral).

### Example 4 — Rejected proposal: env-var flip (no Category will save it)

> Set `VLLM_USE_FLASHINFER_SAMPLER=1` in production.

- **Rejected at NN#8 (authored-mechanism mandate)**. The env var selects an existing code path — the cubin that runs is byte-identical to one the baseline could already produce; no mechanism is authored. No category applies. (Also fails production-parity provenance.)

### Example 4b — Rejected proposal: tuning constant in `.py` (no Category will save it)

> Change `num_stages=2` → `num_stages=3` (or `num_warps`, `BLOCK_SIZE_M`, an `@autotune` config tuple, or a TRTLLM tactic-table entry) in a Triton/kernel source file. Measured +0.69% E2E.

- **Rejected at NN#8** even though it edits a `.py` file and shows a real measured win. The **kernel/cubin body is byte-identical**; the compiler/autotuner/library tactic-table emits the difference, not authored logic. *A constant in `.py` is still config.* Contrast with Example 4c.

### Example 4c — Eligible: authored logic that achieves the same effect

> Hand-write a software-pipelined prefetch loop in the kernel body that achieves what `num_stages` would, with an explicit double-buffer and `cp.async` schedule you wrote.

- **Eligible at NN#8** (gate). The mechanism logic is authored — the kernel body changes, not a meta-parameter. (Magnitude still decided by Gate 5.2 / projection; eligibility ≠ ship.)

### Example 5 — Rejected: missing Category field (schema_version ≥ 4.1)

> Phase 0 proposal targeting DeepGEMM gate_up replacement with full Technology Selection block, but no `## Category` block.

- **Rejected at the Phase 0 eligibility gate** under campaigns with `schema_version >= "4.1"`. Same severity as missing Technology Selection — eligible for one revision; if the champion does not provide a compliant `## Category` block, the candidate is eliminated before debate rounds begin.
- On legacy campaigns (`schema_version < "4.1"`), this rule does not fire.

---

## Cross-Category Anti-Regression

If a proposal crosses category boundaries between rounds (e.g., Round 1 was a `kernel_replacement` that EXHAUSTED, Round 2 proposes a `kernel_fusion`), the **technology-axis anti-regression rule does NOT carry over** — they target different slices and use different evidence forms. The campaign-level forward-progress check (in `references/technology-selection.md` § Technology diversity across rounds) still applies, but framed in terms of *category* rather than *technology class*: the lead's Diversity Check verifies that the new category targets a distinct slice from the prior round's failed category, which is automatic when the categories themselves differ.

---

## Verifying a Proposal's Category Block

The lead's eligibility-gate check in `orchestration/debate-protocol.md § Proposal Eligibility Gate` validates the Category block on schema_version ≥ 4.1 campaigns. Required fields (verbatim from `agents/ammo-champion.md`):

```markdown
## Category
- Selected: <a catalog class (custom_kernel | kernel_fusion | weight_layout_transform | compute_graph_pass | execution_pipeline_restructuring | communication_strategy | attention_kv_layout) | a legacy alias (kernel_replacement | dispatch_optimization) | a novel descriptor name (if no class fits)>
- Slice targeted: <f_e2e(component) | f_e2e(chain) | host portion of inter_kernel_share>
- Projection formula: <exact formula from § Per-Category Projection Formulas matching the declared Slice, with numeric values filled — or the nearest-analogue regime formula for a novel descriptor>
- Justification: <1-2 sentences citing the champion's analysis of the existing profiling data for the assigned component (the seam/op-character that makes this the right mechanism), with workload-composition data from bottleneck_analysis.md as support>
- Expected validation gates: <list per § Per-Category Validation Gate Routing>
```

Block missing → rejected at eligibility gate (after one revision opportunity). Wrong formula for the declared **slice** → 0/10 E2E impact in scoring (per `debate-scoring-rubric.md`). `Selected` may name any catalog class, a legacy alias, or a novel descriptor — eligibility is decided by the NN#8 principle (authored mechanism + profiled bottleneck + measured parity-safe win), not by membership in the catalog.

---

## References

- `e2e-delta-math.md` — `f_e2e` definition, four-slice composition, projection formulas (single source of truth)
- `technology-selection.md` — eligible technologies per category (§5.5)
- `validation-defaults.md` — default Gate 5.2 form
- `debate-scoring-rubric.md` — projection-formula penalty (0/10 for wrong formula)
- `audit-invariants.md` — Stage 2 invariants on category fields
- `agents/ammo-champion.md` — Phase 0 Category block template
- `orchestration/debate-protocol.md` — eligibility gate, Diversity Check (Lead) per-technology exhausted check, champion spawn context (component axis only)
