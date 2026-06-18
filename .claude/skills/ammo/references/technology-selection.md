# Technology Selection for Custom Kernel Proposals

Canonical reference for how AMMO champions pick the **authoring technology** for a proposed kernel. Every other skill file (SKILL.md, ammo-champion.md, debate-scoring-rubric.md, ammo-researcher.md) defers to this document for the ranking, the selection signals, and the anti-regression rule.

## Why this exists

A kernel speedup proposal has two orthogonal axes: *what* the kernel does (the algorithm, the fusion, the memory pattern) and *what tool it is written in* (Triton, CUDA C++, CUTLASS, CuTeDSL, or a library kernel). The debate scores the algorithm on evidence. This document governs the tool choice, because the wrong tool choice can burn an entire round:

- Proposing a Triton rewrite of a Hopper CUTLASS GEMM is almost guaranteed to lose — the baseline already uses TMA + WGMMA + warp-specialized pipelines that Triton cannot match on SM90+.
- Proposing a CuTeDSL replacement on SM80/SM89 is a maturity bet — the DSL's Ampere/Hopper support is experimental per NVIDIA's own docs.
- Proposing a CUDA-graph-incompatible kernel when the campaign requires production parity is a dead end regardless of how fast the kernel is in eager mode.

The selection function below is deterministic. Champions run it in Phase 0 and report the answer in their proposal.

## Technology classes (authoring ladder)

The four classes AMMO recognizes, ranked by **abstraction height** (higher = more of the compiler's job is done for you, lower = more explicit control over the hardware). **CuTeDSL and CUTLASS share rank 2** — both sit on top of the same CuTe layout algebra and expose TMA/WGMMA/UMMA/TMEM at similar levels of abstraction. The Python vs C++ surface is a tooling choice, not an abstraction difference.

| Rank | Technology | Abstraction | Typical wins |
|------|-----------|-------------|--------------|
| 1 (highest) | **Triton** | Python DSL; compiler owns register allocation, SMEM layout, async pipelining | Irregular fusion, dynamic-shape/control-flow kernels, fast-iteration research, SM80/SM89 targets without mature CUTLASS coverage |
| 2 (tie) | **CuTeDSL** (CUTLASS 4 Python DSL) | Python over CuTe layout algebra; TMA/WGMMA/UMMA/TMEM/cluster launch exposed directly | Tensor-core-heavy ops on SM100/SM120/SM121 (Blackwell); within 2% of handwritten C++ per NVIDIA docs. In vLLM: GDN decode path (PR #36111) delivers ~1.6x vs the prior Triton implementation; FP4 MoE GEMM (PR #40082) and FA4 attention (PR #40110) are production examples. |
| 2 (tie) | **CUTLASS (C++)** | Template metaprogramming over CuTe; shipped kernel collections | Dense/MoE GEMM on Hopper+ when a template is already close; CUDA-graph-safe production kernels |
| 3 (lowest) | **CUDA C++** | Hand-written kernels with manual SMEM/register management | Novel algorithms, cluster-launch coordination, integration into existing vLLM csrc/ code paths, paths where CUDA-graph capture is mandatory today |

**For anti-regression purposes, CuTeDSL ↔ CUTLASS is treated as a peer (same-rank) replacement** — moving between them does NOT trigger the cap. Consistent shorthand used in the scoring rubric: `Triton > CuTeDSL ≈ CUTLASS > CUDA C++`.

**Library kernels are not a fifth class** — they are the *default baseline* (cuBLAS, FlashAttn, FlashInfer, DeepGEMM, torch.compile-generated Triton, etc.). They are treated separately by the anti-regression rule; see § Anti-regression rule below.

### The abstraction ranking is used for one specific purpose: the anti-regression rule

See below. For all other purposes (which tool is "best"), the class-fit signals govern.

## The four selection signals

The champion evaluates the proposal against four signals, drawn from data the ammo-researcher emits in `bottleneck_analysis.md` § Technology Landscape:

### 1. Baseline technology

What is the current production kernel written in? Traced from nsys kernel names (e.g., `sm90_xmma_...` → CUTLASS/cuBLAS, `triton_...` → Triton, `void <anonymous>::...` with CuTe layout symbols → CUTLASS or CuTeDSL) and from the vLLM source path the kernel dispatches from.

### 2. Hardware generation (SM capabilities)

What SM-level features are available to a new kernel on this hardware?

| SM | Arch | TMA | WGMMA | Cluster launch | TMEM | CuTeDSL tier |
|----|------|-----|-------|----------------|------|--------------|
| SM80 | A100 | No | No | No | No | Experimental (DSL) |
| SM89 | L40S, L40 | No | No | No | No | Experimental (DSL) |
| SM90 | H100, H200 | Yes | Yes | Yes | No | Experimental (DSL) |
| SM100 | B200 | Yes | Yes (+ UMMA/`tcgen05`) | Yes | Yes | First-class |
| SM120/SM121 | (DGX Spark, consumer Blackwell) | Yes | Yes (+ UMMA) | Yes | Yes | First-class |

### 3. Operation character

Where does the kernel sit on the structured ↔ irregular axis?

| Character | Example ops | Strong fit |
|-----------|-------------|------------|
| Structured tensor-core | Dense GEMM, attention (flash-style), grouped/MoE GEMM, FP8/FP4/INT4 quant-GEMM with fused dequant | CUTLASS, CuTeDSL (SM100+), CUDA C++ |
| Irregular / control-flow-heavy | Token/expert permute, top-k routing with dynamic shapes, paged-KV gather, silu+quant fusion across non-MMA ops | Triton, CUDA C++ |

### 4. Existing library coverage

Does a mature library kernel already exist for this op+shape+dtype? If so, the proposal should be evaluated as an *extension or adaptation* (e.g., adding an epilogue variant, tuning a schedule) rather than a from-scratch rewrite. Missing library coverage is information — not a blank check to pick any tool.

## Selection function (run this in Phase 0)

Given the four signals above, the champion picks a technology by answering:

1. **Match hardware × op character.** Consult the class-fit table below.
2. **Cross-check against baseline technology** (anti-regression rule — see next section).
3. **Check CUDA-graph capture compatibility** (production parity self-check — see further below).
4. **Consider library coverage.** If a mature library kernel exists for the op+shape, justify why writing a custom replacement wins.

### Class-fit table

| Hardware × op character | First pick | Second pick | Avoid |
|-------------------------|-----------|-------------|-------|
| SM100+ × structured tensor-core | **CuTeDSL** or **CUTLASS** | CUDA C++ | Triton (rarely matches tensor-core pipelines on Blackwell) |
| SM90 × structured tensor-core | **CUTLASS** | CUDA C++, CuTeDSL (with caveats — DSL on SM90 is experimental per NVIDIA docs) | Triton for the tensor-core inner loop |
| SM80/SM89 × structured tensor-core | **CUTLASS** or hand-tuned **Triton** | CUDA C++ | CuTeDSL (SM80/SM89 DSL support is experimental) |
| Any SM × irregular / dynamic-shape | **Triton** | CUDA C++ | CUTLASS, CuTeDSL (static typing + layout algebra don't fit dynamic control flow) |
| Any SM × novel algorithm (cluster launches, custom scheduling) | **CUDA C++** | CUTLASS (if templates cover) | Triton, CuTeDSL |
| Any SM × extending a library kernel | Same technology as the library (e.g., CUTLASS template specialization, CuTeDSL epilogue) | CUDA C++ | Rewriting in a different class just to rewrite |

### Anti-regression rule (BLOCKING at Phase 0)

If the proposed technology is **strictly higher in the abstraction ranking above** than the baseline kernel's technology, the proposal requires **Tier 2+ empirical evidence** that the new implementation beats the *actual production kernel* at the target shape. Not a PyTorch reference. Not a roofline bound. Not a proxy kernel. The evidence must use production-parity methodology (CUDA graphs + torch.compile).

"Strictly higher" means the proposed class is at a smaller rank number than the baseline's, per the table above. CuTeDSL and CUTLASS are co-rank 2, so moving between them does not trigger the rule.

Examples:
- Baseline is `sm90_xmma_...` (CUTLASS Hopper GEMM, rank 2). Proposal is a Triton rewrite (rank 1). → Need micro-experiment evidence that Triton beats the CUTLASS kernel at the target shape under CUDA graphs + torch.compile. Otherwise self-reject at Phase 0; if submitted, rubric caps Feasibility at **3/10**.
- Baseline is CUTLASS (rank 2). Proposal is CuTeDSL (rank 2). → Peer replacement. Anti-regression rule does NOT apply. Champion still owes an honest justification of why CuTeDSL is the right pick on this (hardware, op) combo, and the CuTeDSL caveats (CUDA-graph self-check, SM maturity) apply.
- Baseline is hand-written CUDA C++ (rank 3, e.g., `silu_and_mul_quant_kernel`). Proposal is a Triton replacement (rank 1). → Rule applies. Need empirical evidence beats-baseline.
- Baseline is Triton (rank 1). Proposal is CUDA C++ or CUTLASS (rank 3 or 2). → Rule does NOT apply (moving down the ranking / gaining hardware control is not a regression). Champion still has to justify the complexity cost, but not with hard evidence beating the baseline.

#### Library baselines (cuBLAS / FlashAttn / FlashInfer / DeepGEMM / …)

Library kernels are the dominant baseline class in vLLM and are **not members of the custom-authoring ranking**. They are treated as follows:

- **A library kernel is assumed to be at-or-near the performance ceiling for its (op, shape, dtype)** — vendor and community maintainers have spent years tuning these kernels. Treat a library baseline as **rank 0 (below all custom authoring classes)** for anti-regression purposes.
- **Replacing a library kernel with ANY custom authoring class triggers the anti-regression rule** (Tier 2+ beats-baseline evidence required, else Feasibility capped at 3/10). This applies equally to a CUDA C++ rewrite, a CUTLASS specialization, a CuTeDSL kernel, and a Triton rewrite.
- **Extending a library kernel** (adding an epilogue variant, tuning a schedule within the library's extension API, writing a specialization that re-uses the library's core) is NOT the same as replacing it. Treat extensions as same-class — the anti-regression rule does not apply.
- **Special case: torch.compile-generated Triton.** A `triton_poi_fused_*` / `triton_red_fused_*` / `triton_*_fused_*` kernel in the trace is inductor codegen output — treat it as a library baseline (rank 0). A hand-written Triton replacement triggers the anti-regression rule; "same tool class" is a false peer. Common legitimate reasons a hand-written rewrite can beat inductor: custom scheduling, op-specific shmem layout, warp-specialization inductor doesn't reach. Absent one of these, the proposal is likely to lose to the compiler.

Why this exists: without this rule, a round that targets a CUTLASS or library baseline can spend the entire implementation budget on a Triton rewrite that was never going to win. The rule is a forcing function for *proposing the right tool first*, not a free pass to stay at the lowest abstraction forever.

#### Scope: kernel-authoring regime only

The abstraction-ranking anti-regression rule applies **only to the kernel-authoring regime** — `kernel_replacement` / `kernel_fusion` / `custom_kernel` / `attention_kv_layout`, and the merged-kernel case of `weight_layout_transform` (see `references/optimization-categories.md`). **Host-side-structure and inter-kernel-slice proposals have no library baseline to regress against**: a `weight_layout_transform` that does a load-time concat-repack + feeds the *same* library GEMM, or a `dispatch_optimization` / `execution_pipeline_restructuring` / `communication_strategy` / `compute_graph_pass` that authors host-side scheduling/comm/graph-rewrite code, is not "rewriting a library kernel in a higher-abstraction tool" — it changes *which* kernels run / *how* they are coordinated, not the kernel that does the matmul. For these, anti-regression is the E2E metric only. The library-extension exemption above (extending vs replacing) is the same principle: restructuring the wiring around a library kernel, or extending it, is not a regression; flipping a tactic/flag that retunes it without authoring anything is not eligible at all (it fails NN#8, not the anti-regression cap).

## CuTeDSL caveats (explicit warnings)

CuTeDSL is a legitimate first-class peer — but not a universal peer. Champions proposing CuTeDSL include these checks:

1. **CUDA-graph capture compatibility.** FlashAttention-4 (a CuTeDSL kernel) currently requires `enforce_eager=True` in vLLM because CUDA-graph capture of JIT-compiled CuTe kernels is not yet fully supported (vLLM PR #40110). Non-Negotiable #1 requires production parity with CUDA graphs + torch.compile — so a CuTeDSL proposal whose kernel cannot be captured inside a CUDA graph will fail Stage 5 even if the kernel itself is fast.

   **Champion self-check — required acceptance criteria.** The Phase 0 micro-experiment MUST exercise CUDA-graph capture for the proposed kernel and report all four of the following. A three-line log that shows `capture_begin/capture_end returned without exception` is NOT sufficient — Stage 5 will reveal silent replay bugs.

   | # | Check | How to verify | Pass condition |
   |---|-------|---------------|----------------|
   | 1 | Capture succeeds | Call the kernel once inside `torch.cuda.graph(g)` context (or `torch.cuda.CUDAGraph().capture_begin()/capture_end()`) | No exception raised; `g` is populated |
   | 2 | Replay produces correct output | Run a reference (eager) forward once, replay the graph ≥3 times, compare each replay output to the eager reference with `torch.allclose(atol=1e-3, rtol=1e-3)` (tighten for fp32/bf16 as needed) | All replays match eager reference within tolerance |
   | 3 | Replay is deterministic across iterations | Compare replay outputs pairwise across the ≥3 iterations | All replay outputs match each other bitwise |
   | 4 | No re-capture during replay | Profile the replay with `nsys` or count JIT events via `cute.compile` cache hits; verify zero `cudaGraphLaunch` → kernel recompile paths, zero CuTe JIT cache misses during replay | No JIT recompilation on the replay path |

   If any of these fails: either pivot to the `cute.compile` AOT path (if applicable), or disqualify CuTeDSL for this target and fall back to CUTLASS/CUDA C++/library extension. A reference self-test scaffold lives at `scripts/cutedsl_cudagraph_selftest.py` (template — the champion fills in the kernel launch and inputs).
2. **SM generation.** First-class support is Blackwell (SM100/SM120/SM121). Hopper (SM90) and Ampere (SM80) are marked experimental by NVIDIA. Proposing CuTeDSL on SM80/SM89 needs explicit justification.
3. **Version stability.** vLLM pins `nvidia-cutlass-dsl==4.4.2` — version 4.5.0 is known to generate bad PTX on SM121 (vLLM PR #40082). Champions should not assume latest-DSL-version behavior.
4. **Static typing constraints.** CuTeDSL rejects: type changes inside loop bodies, dependent types, dynamic-dtype or dynamic-shape branching inside inline if/else. If the proposed kernel needs any of these, CuTeDSL is the wrong tool — pick Triton or CUDA C++.
5. **Known correctness bugs.** NVFP4 grouped MoE GEMM via CUTLASS DSL has produced garbage output in certain configurations (NVIDIA/cutlass issues). Champions proposing this op in CuTeDSL should validate against a CUTLASS C++ reference.

## Required proposal fields (Phase 0)

Every Phase 0 proposal MUST include a **Technology Selection** block with:

```markdown
## Technology Selection
- Baseline technology: <Triton | CuTeDSL | CUTLASS | CUDA C++ | library:cuBLAS | library:FlashAttn | library:FlashInfer | ...>
- Proposed technology: <Triton | CuTeDSL | CUTLASS | CUDA C++>
- Hardware: <SM80 | SM89 | SM90 | SM100 | SM120 | SM121>
- Op character: <structured tensor-core | irregular / dynamic-shape | novel algorithm | library extension>
- Library coverage: <name of nearest mature kernel, or "none found" with 1-2 sentences of search evidence>
- Justification: <1-3 sentences explaining why the proposed technology is the right fit given the four signals>
- Anti-regression check: <"Not applicable — same or lower abstraction than baseline" | "Applicable — evidence attached: <path to micro-experiment log showing beats-baseline result>">
- CUDA-graph capture self-check (CuTeDSL only): <"Demonstrated — evidence at <path>" | "N/A (not CuTeDSL)">
```

Proposals missing this block are rejected at the Phase 0 eligibility gate. The orchestrator verifies these fields before debate rounds begin.

## Exhaustion handling (between rounds)

When a round EXHAUSTED (all tracks failed), the orchestrator does NOT prescribe a specific pivot. Exhaustion state is recorded in `state.json.round.exhausted_technologies` as a structured array — each entry has `technology_class`, `failure_mode`, `applies_to_component`, `applies_to_shape_bucket`, `evidence_refs`, and `expires_after_reprofile`. Champions filter this array by their current op and shape bucket; they do NOT read free-form exhaustion prose, and the orchestrator does not author any.

Exhaustion is consulted at **two granularities** at two different times (see `orchestration/debate-protocol.md`): **component-level at claim time** (Phase -1 — a component is skipped only if ALL its viable mechanisms are exhausted with no residual surface, since the mechanism is not yet chosen), and **technology-level at proposal time** (the soft Phase 0 Diversity Check (Lead) item 2 — the `component × technology_class × failure_mode` tuple is matched once the champion has analyzed its assigned component's existing profiling data and declared a mechanism).

Champions retain full freedom to re-propose technology X with a different algorithm — but if they do, they must explain what changed versus the prior attempt (recorded in their proposal's `rationale_refs`, not as free-form exhaustion-note narrative). The debate still scores against the selection function. The orchestrator does not pre-commit to a ladder because the right next step depends on *why* the prior round failed (compile failure, correctness miss, regression vs library baseline, shape coverage gap — each is a `failure_mode` enum value) and the champions are better positioned than the orchestrator to reason about that.

### Technology diversity across rounds (soft requirement)

Replacing the old forced-pivot ladder with champion autonomy removes a crude but useful forward-progress guard. To prevent the debate from looping on the same failed technology class across rounds without making progress, the **Diversity Check** (in `debate-protocol.md`) applies an additional soft check when the previous round was EXHAUSTED:

- If round N targeting component C failed with technology class T, and all proposals in round N+1 targeting component C are in class T, the lead REQUIRES at least one proposal to either (a) explore a different class (with justification), or (b) explicitly document why re-attempting T is warranted — e.g., "the algorithm changed fundamentally: prior attempt was a naive tile, this attempt uses warp-specialization," or "the class-fit table shows no viable alternative for this (hardware, op)."
- This is NOT a hard block — champions who can justify re-attempting the same class proceed. It is a "document the loop" requirement so reviewers can see the debate is advancing, not spinning.
- The check only applies after an EXHAUSTED round. If round N succeeded or is still in-flight, no diversity requirement fires.

## References

- `debate-scoring-rubric.md` — how technology fit feeds into Feasibility scoring
- `cudagraph-safety.md` — CUDA graph capture checklist (used by CuTeDSL self-check)
- `gpu-configs.md` — per-GPU SMEM/register/TMA availability (signal 2)
- `optimization-techniques.md` — technique catalog, many of which cross-reference this doc for tool choice
